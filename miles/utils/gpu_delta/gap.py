"""Gap frames: the changed bytes of an XOR frame, each placed by its distance from the previous change.

A frame with n changed bytes encodes to 4n bytes: the changes' gaps as three little-endian byte planes
(the first change's gap is its position), then the n changed values. An unchanged frame encodes to
nothing; a changed frame decodes to its whole dense XOR length. Both directions run on CUDA SMs:
``GapCompressor`` on trainer GPUs, ``codec.GapDecoder`` on receivers.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
import triton
import triton.language as tl

from miles.utils.gpu_delta.nvcomp import CompressionBatch, _Alignments

GAP_BYTES = 3
CHANGE_BYTES = GAP_BYTES + 1
# The largest frame whose gaps fit their three bytes.
MAX_FRAME_BYTES = 1 << (8 * GAP_BYTES)
# Frames start 16-byte aligned on both sides: decoders zero them in 16-byte vectors.
ALIGNMENT = 16
_TILE = 1 << 16
_BLOCK = 1 << 12
_ZERO_BLOCK = 1 << 13
_DECODE_BLOCK = 1 << 10


@triton.jit(do_not_specialize=["nframes"])
def _count_changes(frames, nframes, counts, last_positions, TILE: tl.constexpr, BLOCK: tl.constexpr):
    """Each tile's changed bytes and its last changed position, -1 in an unchanged tile."""
    frame = tl.program_id(0)
    tile = tl.program_id(1)
    source = tl.load(frames + frame).to(tl.pointer_type(tl.uint8))
    length = tl.load(frames + nframes + frame).to(tl.int32)
    lanes = tl.arange(0, BLOCK)
    count = tl.zeros((), tl.int32)
    last = tl.full((), -1, tl.int32)
    for start in range(tile * TILE, tl.minimum((tile + 1) * TILE, length), BLOCK):
        offsets = start + lanes
        changed = tl.load(source + offsets, offsets < length, other=0) != 0
        count += tl.sum(changed.to(tl.int32), axis=0)
        last = tl.maximum(last, tl.max(tl.where(changed, offsets, -1), axis=0))
    tl.store(counts + frame * tl.num_programs(1) + tile, count)
    tl.store(last_positions + frame * tl.num_programs(1) + tile, last)


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b)


@triton.jit(do_not_specialize=["nframes", "tiles"])
def _encode_frames(
    frames,
    nframes,
    counts,
    last_positions,
    outputs,
    tiles,
    TILES: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Each changed tile writes its changes at their ranks in the frame: the earlier tiles' counts give
    its first rank, their last change its first gap."""
    frame = tl.program_id(0)
    tile = tl.program_id(1)
    indices = tl.arange(0, TILES)
    present = indices < tiles
    tile_counts = tl.load(counts + frame * tiles + indices, present, other=0)
    if tl.sum(tl.where(indices == tile, tile_counts, 0), axis=0) > 0:
        source = tl.load(frames + frame).to(tl.pointer_type(tl.uint8))
        length = tl.load(frames + nframes + frame).to(tl.int32)
        output = tl.load(outputs + frame).to(tl.pointer_type(tl.uint8))
        changes = tl.sum(tile_counts, axis=0)
        rank = tl.sum(tl.where(indices < tile, tile_counts, 0), axis=0)
        earlier = tl.load(last_positions + frame * tiles + indices, present, other=-1)
        # Position 0 when no earlier change: the first change's gap is its position.
        last = tl.max(tl.where(indices < tile, earlier, 0), axis=0)
        lanes = tl.arange(0, BLOCK)
        for start in range(tile * TILE, tl.minimum((tile + 1) * TILE, length), BLOCK):
            offsets = start + lanes
            valid = offsets < length
            values = tl.load(source + offsets, valid, other=0)
            changed = values != 0
            changed_before = tl.load(source + offsets - 1, valid & (lanes > 0), other=0) != 0
            previous = tl.maximum(tl.associative_scan(tl.where(changed_before, offsets - 1, -1), 0, _maximum), last)
            gaps = offsets - previous
            ranks = rank + tl.cumsum(changed.to(tl.int32), 0) - 1
            tl.store(output + ranks, (gaps & 0xFF).to(tl.uint8), changed)
            tl.store(output + changes + ranks, ((gaps >> 8) & 0xFF).to(tl.uint8), changed)
            tl.store(output + 2 * changes + ranks, (gaps >> 16).to(tl.uint8), changed)
            tl.store(output + 3 * changes + ranks, values, changed)
            rank += tl.sum(changed.to(tl.int32), axis=0)
            last = tl.maximum(last, tl.max(tl.where(changed, offsets, -1), axis=0))


class GapCompressor:
    """``NvcompCompressor.compress`` for gap frames.

    A count pass sizes each frame's output, one host readback of the counts lays out a compact output,
    then an encode pass fills it. An unchanged frame gets an empty output; no frame fails.
    """

    codec = "gap"

    def __init__(self, device: torch.device):
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("GPU delta compressor requires an explicit CUDA device")
        self.alignments = _Alignments(input=ALIGNMENT, output=ALIGNMENT, temp=1)

    def compress(self, frames: list[torch.Tensor], stream: torch.cuda.Stream) -> CompressionBatch:
        if stream.device != self.device:
            raise ValueError("Compression stream/device mismatch")
        for frame in frames:
            if (
                frame.device != self.device
                or frame.dtype != torch.uint8
                or not frame.is_contiguous()
                or not 0 < frame.numel() <= MAX_FRAME_BYTES
                or frame.data_ptr() % ALIGNMENT
            ):
                raise ValueError("Gap frames must be 16-byte aligned contiguous CUDA uint8, with 1..16 MiB bytes")
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            return self._encode(frames, stream)

    def _encode(self, frames, stream):
        count = len(frames)
        statuses = torch.zeros(count, dtype=torch.int32, device=self.device)
        if not count:
            return CompressionBatch([], torch.empty(0, dtype=torch.int64, device=self.device), statuses, ())
        frame_bytes = [frame.numel() for frame in frames]
        tiles = triton.cdiv(max(frame_bytes), _TILE)
        table = torch.empty((2, count), dtype=torch.int64, device="cpu", pin_memory=True)
        table.numpy()[:] = [[frame.data_ptr() for frame in frames], frame_bytes]
        device_table = table.to(self.device, non_blocking=True)
        tile_counts = torch.empty((count, tiles), dtype=torch.int32, device=self.device)
        last_positions = torch.empty_like(tile_counts)
        _count_changes[(count, tiles)](
            device_table, count, tile_counts, last_positions, TILE=_TILE, BLOCK=_BLOCK, num_warps=4
        )
        sizes = tile_counts.sum(dim=1, dtype=torch.int64).mul_(CHANGE_BYTES)
        host_sizes = torch.empty(count, dtype=torch.int64, device="cpu", pin_memory=True)
        host_sizes.copy_(sizes, non_blocking=True)
        counted = torch.cuda.Event()
        counted.record(stream)
        counted.synchronize()
        encoded = host_sizes.numpy()
        aligned = (encoded + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        starts = np.cumsum(aligned) - aligned
        output = torch.empty(int(aligned.sum()), dtype=torch.uint8, device=self.device)
        outputs = [output[start : start + size] for start, size in zip(starts.tolist(), encoded.tolist(), strict=True)]
        pointers = torch.empty(count, dtype=torch.int64, device="cpu", pin_memory=True)
        np.add(starts, output.data_ptr(), out=pointers.numpy())
        device_pointers = pointers.to(self.device, non_blocking=True)
        if output.numel():
            _encode_frames[(count, tiles)](
                device_table,
                count,
                tile_counts,
                last_positions,
                device_pointers,
                tiles,
                TILES=triton.next_power_of_2(tiles),
                TILE=_TILE,
                BLOCK=_BLOCK,
                num_warps=4,
            )
        keepalive = (
            frames,
            table,
            device_table,
            tile_counts,
            last_positions,
            host_sizes,
            pointers,
            device_pointers,
            output,
        )
        return CompressionBatch(outputs, sizes, statuses, keepalive)


# Every argument is a device address, as in nvCOMP's batched decompression call.
@triton.jit(do_not_specialize=["inputs", "encoded_sizes", "decoded_sizes", "actual_sizes", "outputs", "statuses"])
def _decode_frames(
    inputs,
    encoded_sizes,
    decoded_sizes,
    actual_sizes,
    outputs,
    statuses,
    ZERO_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per frame zeroes the frame's output, then stores each change's value at its position.

    A change past the frame, a repeated position or a zero value sets the frame's status; no store
    leaves the frame.
    """
    frame = tl.program_id(0)
    source = tl.load(inputs.to(tl.pointer_type(tl.int64)) + frame).to(tl.pointer_type(tl.uint8))
    changes = tl.load(encoded_sizes.to(tl.pointer_type(tl.int64)) + frame) // 4
    length = tl.load(decoded_sizes.to(tl.pointer_type(tl.int64)) + frame)
    target = tl.load(outputs.to(tl.pointer_type(tl.int64)) + frame).to(tl.pointer_type(tl.uint8))
    target = tl.multiple_of(target, 16)
    zero_lanes = tl.arange(0, ZERO_BLOCK)
    zeros = tl.zeros((ZERO_BLOCK,), tl.uint8)
    whole = length // ZERO_BLOCK * ZERO_BLOCK
    for start in range(0, whole, ZERO_BLOCK):
        tl.store(target + start + zero_lanes, zeros)
    tl.store(target + whole + zero_lanes, zeros, whole + zero_lanes < length)
    # Every thread's zeros land before any thread's values.
    tl.debug_barrier()
    lanes = tl.arange(0, BLOCK)
    base = tl.zeros((), tl.int64)
    corrupt = tl.zeros((), tl.int32)
    for first in range(0, changes, BLOCK):
        index = first + lanes
        live = index < changes
        gaps = tl.load(source + index, live, other=0).to(tl.int64)
        gaps |= tl.load(source + changes + index, live, other=0).to(tl.int64) << 8
        gaps |= tl.load(source + 2 * changes + index, live, other=0).to(tl.int64) << 16
        values = tl.load(source + 3 * changes + index, live, other=0)
        positions = base + tl.cumsum(gaps, 0)
        inside = positions < length
        wrong = ~inside | ((index > 0) & (gaps == 0)) | (values == 0)
        corrupt |= tl.max((live & wrong).to(tl.int32), axis=0)
        tl.store(target + positions, values, live & inside)
        base += tl.sum(gaps, axis=0)
    tl.store(statuses.to(tl.pointer_type(tl.int32)) + frame, corrupt)
    tl.store(actual_sizes.to(tl.pointer_type(tl.int64)) + frame, length)


@lru_cache(maxsize=None, typed=True)
def compiled_decode(device_index):
    """Compile and load the decode kernel once per device, before any pause."""
    # Any 64-bit address selects the kernel's i64 argument types.
    address = 1 << 40
    with torch.cuda.device(device_index):
        kernel = _decode_frames.warmup(
            *[address] * 6, ZERO_BLOCK=_ZERO_BLOCK, BLOCK=_DECODE_BLOCK, grid=(1, 1, 1), num_warps=4
        )
        _ = kernel.run
        return kernel
