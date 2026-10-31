from collections.abc import Sequence
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

TILE_NBYTES = 1 << 16
BLOCK_NBYTES = 1 << 12
GAP_NBYTES = 3


@triton.jit
def _nonzero_bytes(words):
    """0x80 in each byte of the words that is not zero, 0 elsewhere."""
    low = tl.full((), 0x7F7F7F7F, tl.int64).to(tl.uint32)
    high = tl.full((), 0x80808080, tl.int64).to(tl.uint32)
    return (((words & low) + low) | words) & high


@triton.jit
def _checksum_terms(word_indices, words):
    """(murmur3's 32-bit finalizer of the word's index, made odd) x the word, as a full 64-bit product."""
    h = word_indices.to(tl.uint32)
    h ^= h >> 16
    h *= tl.full((), 0x85EBCA6B, tl.int64).to(tl.uint32)
    h ^= h >> 13
    h *= tl.full((), 0xC2B2AE35, tl.int64).to(tl.uint32)
    h ^= h >> 16
    h |= 1
    return (tl.umulhi(h, words).to(tl.uint64) << 32) | (h * words).to(tl.uint64)


@triton.jit(do_not_specialize=["nframes"])
def _xor_frames_kernel(
    table, nframes, tile_counts, tile_last_positions, tile_checksums, TILE: tl.constexpr, BLOCK_WORDS: tl.constexpr
):
    frame = tl.program_id(0)
    tile = tl.program_id(1)
    target = tl.load(table + frame).to(tl.pointer_type(tl.uint32))
    length = tl.load(table + nframes + frame).to(tl.int32)
    other = tl.load(table + 2 * nframes + frame).to(tl.pointer_type(tl.uint32))
    first_word = tl.load(table + 3 * nframes + frame)
    words = tl.cdiv(length, 4)
    lanes = tl.arange(0, BLOCK_WORDS)
    all_bytes = tl.full((), 0xFFFFFFFF, tl.int64).to(tl.uint32)
    count = tl.full((), 0, tl.int32)
    last = tl.full((), -1, tl.int32)
    checksum = tl.full((), 0, tl.uint64)
    for start in range(tile * (TILE // 4), tl.minimum((tile + 1) * (TILE // 4), words), BLOCK_WORDS):
        offsets = start + lanes
        valid = offsets < words
        # A tensor's last word may run past its end: those bytes are read, then masked out.
        kept = all_bytes >> (8 * (4 - tl.minimum(tl.maximum(length - offsets * 4, 1), 4))).to(tl.uint32)
        new = tl.load(other + offsets, valid, other=0)
        xor = (tl.load(target + offsets, valid, other=0) ^ new) & kept
        tl.store(target + offsets, xor, valid)
        changed = _nonzero_bytes(xor)
        bytes_changed = ((changed >> 7) & 1) + ((changed >> 15) & 1) + ((changed >> 23) & 1) + (changed >> 31)
        count += tl.sum(bytes_changed.to(tl.int32), axis=0)
        top = tl.where((xor >> 24) != 0, 3, tl.where((xor >> 16) != 0, 2, tl.where((xor >> 8) != 0, 1, 0)))
        last = tl.maximum(last, tl.max(tl.where(xor != 0, offsets * 4 + top, -1), axis=0))
        checksum += tl.sum(_checksum_terms(first_word + offsets, new & ((changed >> 7) * 255)), axis=0)
    tl.store(tile_counts + frame * tl.num_programs(1) + tile, count)
    tl.store(tile_last_positions + frame * tl.num_programs(1) + tile, last)
    tl.store(tile_checksums + frame * tl.num_programs(1) + tile, checksum.to(tl.int64, bitcast=True))


@dataclass(frozen=True)
class FrameChanges:
    """Per frame of a batch, tensor after tensor and ceil(nbytes / frame_nbytes) per tensor: ``table``
    holds its target address, length, other address and first word in its tensor; per tile of
    TILE_NBYTES, ``tile_counts`` holds the changed bytes, ``tile_last_positions`` the last one's
    position in the frame, or -1, and ``tile_checksums`` the checksum of the changed bytes' new values
    (publication.checksum), as int64 bits."""

    frame_nbytes: int
    table: torch.Tensor
    tile_counts: torch.Tensor
    tile_last_positions: torch.Tensor
    tile_checksums: torch.Tensor

    @property
    def counts(self) -> torch.Tensor:
        """Changed bytes per frame."""
        return self.tile_counts.sum(dim=1, dtype=torch.int64)


def xor_frames_(targets: Sequence[torch.Tensor], others: Sequence[torch.Tensor], frame_nbytes: int) -> FrameChanges:
    """XORs each of ``others`` into its target in place and finds what changed in each frame of
    ``frame_nbytes``; queued on the current stream. Both start 4-byte aligned; a target also owns
    the bytes up to its next multiple of 4."""
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
                or tensor.data_ptr() % 4
            ):
                raise ValueError(
                    f"pair {index}: {tensor.dtype} {tuple(tensor.shape)} on {tensor.device} is not contiguous 1-D "
                    f"uint8 on {device}, 4-byte aligned"
                )
        if target.numel() != other.numel():
            raise ValueError(f"pair {index}: {target.numel()} target bytes for {other.numel()} other bytes")
    rows = [[], [], [], []]
    for target, other in zip(targets, others, strict=True):
        for offset in range(0, target.numel(), frame_nbytes):
            rows[0].append(target.data_ptr() + offset)
            rows[1].append(min(frame_nbytes, target.numel() - offset))
            rows[2].append(other.data_ptr() + offset)
            rows[3].append(offset // 4)
    nframes, tiles = len(rows[0]), frame_nbytes // TILE_NBYTES
    table = torch.tensor(rows, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
    tile_counts = torch.empty((nframes, tiles), dtype=torch.int32, device=device)
    tile_last_positions = torch.empty((nframes, tiles), dtype=torch.int32, device=device)
    tile_checksums = torch.empty((nframes, tiles), dtype=torch.int64, device=device)
    if nframes:
        _xor_frames_kernel[(nframes, tiles)](
            table,
            nframes,
            tile_counts,
            tile_last_positions,
            tile_checksums,
            TILE=TILE_NBYTES,
            BLOCK_WORDS=BLOCK_NBYTES // 4,
            num_warps=4,
        )
    return FrameChanges(frame_nbytes, table, tile_counts, tile_last_positions, tile_checksums)


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
