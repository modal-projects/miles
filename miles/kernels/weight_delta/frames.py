from collections.abc import Sequence
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

TILE_NBYTES = 1 << 16
BLOCK_NBYTES = 1 << 12
GAP_NBYTES = 3


@triton.jit(do_not_specialize=["nframes"])
def _xor_frames_kernel(table, nframes, tile_counts, tile_last_positions, TILE: tl.constexpr, BLOCK: tl.constexpr):
    frame = tl.program_id(0)
    tile = tl.program_id(1)
    target = tl.load(table + frame).to(tl.pointer_type(tl.uint8))
    length = tl.load(table + nframes + frame).to(tl.int32)
    other = tl.load(table + 2 * nframes + frame).to(tl.pointer_type(tl.uint8))
    lanes = tl.arange(0, BLOCK)
    count = tl.full((), 0, tl.int32)
    last = tl.full((), -1, tl.int32)
    for start in range(tile * TILE, tl.minimum((tile + 1) * TILE, length), BLOCK):
        offsets = start + lanes
        valid = offsets < length
        xor = tl.load(target + offsets, valid, other=0) ^ tl.load(other + offsets, valid, other=0)
        tl.store(target + offsets, xor, valid)
        changed = xor != 0
        count += tl.sum(changed.to(tl.int32), axis=0)
        last = tl.maximum(last, tl.max(tl.where(changed, offsets, -1), axis=0))
    tl.store(tile_counts + frame * tl.num_programs(1) + tile, count)
    tl.store(tile_last_positions + frame * tl.num_programs(1) + tile, last)


@dataclass(frozen=True)
class FrameChanges:
    """Per frame of a batch, tensor after tensor and ceil(nbytes / frame_nbytes) per tensor: ``table``
    holds its target address, length and other address; per tile of TILE_NBYTES, ``tile_counts`` holds
    the changed bytes and ``tile_last_positions`` the last one's position in the frame, or -1."""

    frame_nbytes: int
    table: torch.Tensor
    tile_counts: torch.Tensor
    tile_last_positions: torch.Tensor

    @property
    def counts(self) -> torch.Tensor:
        """Changed bytes per frame."""
        return self.tile_counts.sum(dim=1, dtype=torch.int64)


def xor_frames_(targets: Sequence[torch.Tensor], others: Sequence[torch.Tensor], frame_nbytes: int) -> FrameChanges:
    """XORs each of ``others`` into its target in place and finds what changed in each frame of
    ``frame_nbytes``; queued on the current stream."""
    if frame_nbytes < TILE_NBYTES or frame_nbytes & (frame_nbytes - 1):
        raise ValueError(f"frame_nbytes must be a power of two of at least {TILE_NBYTES}, got {frame_nbytes}")
    if len(targets) != len(others):
        raise ValueError(f"{len(targets)} targets for {len(others)} others")
    device = torch.device("cuda", torch.cuda.current_device())
    for index, (target, other) in enumerate(zip(targets, others, strict=True)):
        for tensor in (target, other):
            if (
                tensor.dtype != torch.uint8
                or tensor.ndim != 1
                or not tensor.is_contiguous()
                or tensor.device != device
            ):
                raise ValueError(
                    f"pair {index}: {tensor.dtype} {tuple(tensor.shape)} on {tensor.device} is not contiguous 1-D "
                    f"uint8 on {device}"
                )
        if target.numel() != other.numel():
            raise ValueError(f"pair {index}: {target.numel()} target bytes for {other.numel()} other bytes")
    rows = [[], [], []]
    for target, other in zip(targets, others, strict=True):
        for offset in range(0, target.numel(), frame_nbytes):
            rows[0].append(target.data_ptr() + offset)
            rows[1].append(min(frame_nbytes, target.numel() - offset))
            rows[2].append(other.data_ptr() + offset)
    nframes, tiles = len(rows[0]), frame_nbytes // TILE_NBYTES
    table = torch.tensor(rows, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
    tile_counts = torch.empty((nframes, tiles), dtype=torch.int32, device=device)
    tile_last_positions = torch.empty((nframes, tiles), dtype=torch.int32, device=device)
    if nframes:
        _xor_frames_kernel[(nframes, tiles)](
            table, nframes, tile_counts, tile_last_positions, TILE=TILE_NBYTES, BLOCK=BLOCK_NBYTES, num_warps=4
        )
    return FrameChanges(frame_nbytes, table, tile_counts, tile_last_positions)


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b)


@triton.jit(do_not_specialize=["nframes"])
def _encode_gap_frames_kernel(
    table,
    nframes,
    tile_counts,
    tile_last_positions,
    outputs,
    TILES: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    frame = tl.program_id(0)
    tile = tl.program_id(1)
    tiles = tl.arange(0, TILES)
    counts = tl.load(tile_counts + frame * TILES + tiles)
    if tl.sum(tl.where(tiles == tile, counts, 0), axis=0) > 0:
        target = tl.load(table + frame).to(tl.pointer_type(tl.uint8))
        length = tl.load(table + nframes + frame).to(tl.int32)
        output = tl.load(outputs + frame).to(tl.pointer_type(tl.uint8))
        frame_count = tl.sum(counts, axis=0)
        rank = tl.sum(tl.where(tiles < tile, counts, 0), axis=0)
        last = tl.max(tl.where(tiles < tile, tl.load(tile_last_positions + frame * TILES + tiles), 0), axis=0)
        lanes = tl.arange(0, BLOCK)
        for start in range(tile * TILE, tl.minimum((tile + 1) * TILE, length), BLOCK):
            offsets = start + lanes
            valid = offsets < length
            xor = tl.load(target + offsets, valid, other=0)
            changed = xor != 0
            changed_before = tl.load(target + offsets - 1, valid & (lanes > 0), other=0) != 0
            previous = tl.maximum(tl.associative_scan(tl.where(changed_before, offsets - 1, -1), 0, _maximum), last)
            gaps = offsets - previous
            ranks = rank + tl.cumsum(changed.to(tl.int32), 0) - 1
            tl.store(output + ranks, (gaps & 0xFF).to(tl.uint8), changed)
            tl.store(output + frame_count + ranks, ((gaps >> 8) & 0xFF).to(tl.uint8), changed)
            tl.store(output + 2 * frame_count + ranks, (gaps >> 16).to(tl.uint8), changed)
            tl.store(output + 3 * frame_count + ranks, xor, changed)
            rank += tl.sum(changed.to(tl.int32), axis=0)
            last = tl.maximum(last, tl.max(tl.where(changed, offsets, -1), axis=0))


def encode_gap_frames(changes: FrameChanges, output: torch.Tensor, offsets: Sequence[int]) -> None:
    """Writes each changed frame's XOR, now in its target, at its offset in ``output`` as codec gap:
    n changed bytes become (GAP_NBYTES + 1) * n, each position's gap from the previous one (the first
    from 0) as a little-endian u24 in three byte planes, then the values. One offset per frame; an
    unchanged frame's is not used. Queued on the current stream."""
    if changes.frame_nbytes > 1 << (8 * GAP_NBYTES):
        raise ValueError(f"a gap frame is at most 16 MiB, got {changes.frame_nbytes} bytes")
    nframes, tiles = changes.tile_counts.shape
    if output.dtype != torch.uint8 or output.ndim != 1 or not output.is_contiguous() or len(offsets) != nframes:
        raise ValueError(
            f"output must be contiguous 1-D uint8 and offsets one per frame, got {output.dtype} "
            f"{tuple(output.shape)} and {len(offsets)} offsets for {nframes} frames"
        )
    if not nframes:
        return
    outputs = torch.tensor([output.data_ptr() + offset for offset in offsets], dtype=torch.int64, pin_memory=True)
    _encode_gap_frames_kernel[(nframes, tiles)](
        changes.table,
        nframes,
        changes.tile_counts,
        changes.tile_last_positions,
        outputs.to(output.device, non_blocking=True),
        TILES=tiles,
        TILE=TILE_NBYTES,
        BLOCK=BLOCK_NBYTES,
        num_warps=4,
    )
