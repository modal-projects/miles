"""XOR tensors encoded on the GPU into weight-delta/v1 records: each batch's changed frames and their
inner encodings, then every changed tensor's outer Zstd chunks."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from miles.backends.training_utils.weight_update.protocols.weight_delta.publication import (
    ALIGNMENT,
    CODECS,
    FRAME_NBYTES,
)
from miles.kernels.weight_delta.frames import GAP_NBYTES, FrameChanges, encode_gap_frames, xor_frames_
from miles.utils.nvcomp import CompressedBatch, NvcompCompressor

CHUNK_NBYTES = 1 << 20
BITS_SLICE_NBYTES = 1 << 26


@dataclass(frozen=True)
class EncodedTensor:
    """A changed tensor's frame records, how many of its bytes changed, and which when the encoder was
    asked (changed_bits). Its payload and Zstd record come from finish."""

    frames: list[dict]
    changed_nbytes: int
    changed_bits: torch.Tensor | None


class DeltaEncoder:
    """Works on the device's current stream and uploads previous on its own: ``encode`` returns once it
    no longer reads the pairs, and keeps each changed tensor's inner on the GPU until max_pending_nbytes
    of them wait for the outer Zstd; ``finish`` returns once every payload is on the host."""

    def __init__(self, codec: str, device: torch.device, max_pending_nbytes: int, with_changed_bits: bool = False):
        if codec not in CODECS:
            raise ValueError(f"weight-delta codec must be one of {CODECS}, got {codec!r}")
        self.device = torch.device(device)
        self._max_pending_nbytes = max_pending_nbytes
        self._with_changed_bits = with_changed_bits
        self._snappy = NvcompCompressor("snappy", self.device) if codec == "snappy-zstd" else None
        self._zstd = NvcompCompressor("zstd", self.device)
        self._padding = torch.zeros(ALIGNMENT - 1, dtype=torch.uint8, device=self.device)
        self._upload_stream = torch.cuda.Stream(device=self.device)
        self._pending: list[torch.Tensor] = []
        self._pending_nbytes = 0
        self._payloads: list[memoryview] = []
        self._zstds: list[dict] = []

    def encode(self, pairs: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> list[EncodedTensor | None]:
        """Each pair is a tensor's previous bytes (pinned host) and current bytes (on this device,
        complete on its current stream), contiguous 1-D uint8 of one length. None for a tensor that
        did not change."""
        for index, (previous, current) in enumerate(pairs):
            # Pinned host bytes are always copied to the device, so the in-place XOR leaves them intact.
            if (
                previous.dtype != torch.uint8
                or current.dtype != torch.uint8
                or previous.ndim != 1
                or previous.shape != current.shape
                or not previous.is_contiguous()
                or not current.is_contiguous()
                or (previous.numel() and not previous.is_pinned())
                or current.device != self.device
            ):
                raise ValueError(
                    f"pair {index}: previous {previous.dtype} {tuple(previous.shape)} on {previous.device} and "
                    f"current {current.dtype} {tuple(current.shape)} on {current.device} must be contiguous 1-D "
                    f"uint8 of one length, previous pinned on the host and current on {self.device}"
                )
        encoded: list[EncodedTensor | None] = [None] * len(pairs)
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream()
            try:
                xors = [torch.empty(previous.numel(), dtype=torch.uint8, device=self.device) for previous, _ in pairs]
                self._upload_stream.wait_stream(stream)
                with torch.cuda.stream(self._upload_stream):
                    for xor, (previous, _) in zip(xors, pairs, strict=True):
                        xor.copy_(previous, non_blocking=True)
                stream.wait_stream(self._upload_stream)
                changes = xor_frames_(xors, [current for _, current in pairs], FRAME_NBYTES)
                frame_counts = _split(
                    changes.counts.tolist(), [(xor.numel() + FRAME_NBYTES - 1) // FRAME_NBYTES for xor in xors]
                )
                changed_frames = [[i for i, count in enumerate(counts) if count] for counts in frame_counts]
                changed_tensors = [index for index, indices in enumerate(changed_frames) if indices]
                if not changed_tensors:
                    return encoded
                bits = [changed_bits(xors[index]) if self._with_changed_bits else None for index in changed_tensors]
                if self._snappy is None:
                    inners, layouts = self._encode_gap(changes, frame_counts)
                else:
                    inners, layouts = self._encode_snappy(xors, changed_frames, changed_tensors, stream)
                self._pending += inners
                self._pending_nbytes += sum(inner.numel() for inner in inners)
                if self._pending_nbytes >= self._max_pending_nbytes:
                    self._compress_pending(stream)
            except BaseException:
                # Queued work may still read the caller's tensors; let it finish before they are released.
                self._upload_stream.synchronize()
                stream.synchronize()
                raise
        for index, layout, tensor_bits in zip(changed_tensors, layouts, bits, strict=True):
            nbytes = pairs[index][1].numel()
            frame_records = [
                {
                    "offset": i * FRAME_NBYTES,
                    "nbytes": min(FRAME_NBYTES, nbytes - i * FRAME_NBYTES),
                    "inner_offset": inner_offset,
                    "inner_nbytes": inner_nbytes,
                }
                for i, (inner_offset, inner_nbytes) in zip(changed_frames[index], layout, strict=True)
            ]
            encoded[index] = EncodedTensor(frame_records, sum(frame_counts[index]), tensor_bits)
        return encoded

    def finish(self) -> tuple[list[memoryview], list[dict]]:
        """The payload and Zstd record of each changed tensor, in the order encode returned them."""
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream()
            try:
                self._compress_pending(stream)
            except BaseException:
                stream.synchronize()
                raise
        return self._payloads, self._zstds

    def _encode_gap(
        self, changes: FrameChanges, frame_counts: list[list[int]]
    ) -> tuple[list[torch.Tensor], list[list[tuple[int, int]]]]:
        """Each changed tensor's inner, all in one buffer the kernel writes, and its frames'
        (inner_offset, inner_nbytes)."""
        offsets, spans, layouts, end = _layout(
            [[(GAP_NBYTES + 1) * count for count in counts] for counts in frame_counts]
        )
        output = torch.zeros(end, dtype=torch.uint8, device=self.device)
        encode_gap_frames(changes, output, offsets)
        return [output[start:end] for start, end in spans], layouts

    def _encode_snappy(
        self,
        xors: list[torch.Tensor],
        changed_frames: list[list[int]],
        changed_tensors: list[int],
        stream: torch.cuda.Stream,
    ) -> tuple[list[torch.Tensor], list[list[tuple[int, int]]]]:
        """Each changed tensor's inner, all in one buffer copied from one nvCOMP call's outputs, and its
        frames' (inner_offset, inner_nbytes)."""
        frames = [
            [xors[index][i * FRAME_NBYTES : (i + 1) * FRAME_NBYTES] for i in changed_frames[index]]
            for index in changed_tensors
        ]
        batch = self._snappy.compress([frame for tensor_frames in frames for frame in tensor_frames], stream)
        compressed = [slot[:size] for slot, size in zip(batch.slots, _compressed_nbytes(batch, "snappy"), strict=True)]
        offsets, spans, layouts, _ = _layout(
            _split([inner.numel() for inner in compressed], [len(tensor_frames) for tensor_frames in frames])
        )
        pieces, position = [], 0
        for offset, inner in zip(offsets, compressed, strict=True):
            if offset > position:
                pieces.append(self._padding[: offset - position])
            pieces.append(inner)
            position = offset + inner.numel()
        output = torch.cat(pieces)
        return [output[start:end] for start, end in spans], layouts

    def _compress_pending(self, stream: torch.cuda.Stream) -> None:
        if self._pending:
            payloads, zstds = self._compress(self._pending, stream)
            self._payloads += payloads
            self._zstds += zstds
            self._pending, self._pending_nbytes = [], 0

    def _compress(
        self, inners: Sequence[torch.Tensor], stream: torch.cuda.Stream
    ) -> tuple[list[memoryview], list[dict]]:
        """Each tensor's inner in chunks, each chunk one Zstd frame, the chunks back to back as its payload."""
        tensor_chunks = [inner.split(CHUNK_NBYTES) for inner in inners]
        batch = self._zstd.compress([chunk for chunks in tensor_chunks for chunk in chunks], stream)
        nbytes = _compressed_nbytes(batch, "zstd")
        packed = torch.cat([slot[:size] for slot, size in zip(batch.slots, nbytes, strict=True)])
        host = torch.empty(packed.numel(), dtype=torch.uint8, pin_memory=True)
        host.copy_(packed, non_blocking=True)
        stream.synchronize()
        host_bytes = memoryview(host.numpy())
        payloads, zstds, start = [], [], 0
        for inner, chunks, sizes in zip(
            inners, tensor_chunks, _split(nbytes, [len(chunks) for chunks in tensor_chunks]), strict=True
        ):
            chunk_records, offset, inner_offset = [], 0, 0
            for chunk, size in zip(chunks, sizes, strict=True):
                chunk_records.append(
                    {"offset": offset, "nbytes": size, "inner_offset": inner_offset, "inner_nbytes": chunk.numel()}
                )
                offset += size
                inner_offset += chunk.numel()
            payloads.append(host_bytes[start : start + offset])
            zstds.append({"inner_nbytes": inner.numel(), "chunks": chunk_records})
            start += offset
        return payloads, zstds


def changed_bits(xor: torch.Tensor) -> torch.Tensor:
    """One bit per byte of a 1-D uint8 XOR, set where it changed, in numpy.packbits' order; a slice at a
    time to keep the scratch small."""
    return torch.cat([_pack_bits(piece != 0) for piece in xor.split(BITS_SLICE_NBYTES)])


def _pack_bits(mask: torch.Tensor) -> torch.Tensor:
    # Eight 0/1 bytes read as one little-endian int64; this multiplier gathers byte i into bit 63 - i.
    words = torch.nn.functional.pad(mask.view(torch.uint8), (0, -mask.numel() % 8)).view(torch.int64)
    packed = words * -0x7FBFDFEFF7FBFDFF
    packed >>= 56
    packed &= 0xFF
    return packed.to(torch.uint8)


def _layout(
    frame_inner_nbytes: list[list[int]],
) -> tuple[list[int], list[tuple[int, int]], list[list[tuple[int, int]]], int]:
    """Where each frame's inner goes in one buffer of every changed tensor's inner, each tensor and each
    of its frames at a multiple of ALIGNMENT; 0 bytes is an unchanged frame. Returns each frame's offset,
    each changed tensor's (start, end) and its frames' (inner_offset, inner_nbytes), and the end."""
    offsets, spans, layouts, end = [], [], [], 0
    for nbytes in frame_inner_nbytes:
        end += -end % ALIGNMENT
        start, layout = end, []
        for size in nbytes:
            if size:
                end += -end % ALIGNMENT
                layout.append((end - start, size))
            offsets.append(end)
            end += size
        if layout:
            spans.append((start, end))
            layouts.append(layout)
    return offsets, spans, layouts, end


def _compressed_nbytes(batch: CompressedBatch, algorithm: str) -> list[int]:
    """Each chunk's compressed size once nvCOMP is done; a failed chunk refuses the batch."""
    nbytes, statuses = torch.stack((batch.nbytes, batch.statuses.to(torch.int64))).tolist()
    for index, (size, status, slot) in enumerate(zip(nbytes, statuses, batch.slots, strict=True)):
        if status or not 0 < size <= slot.numel():
            raise RuntimeError(
                f"nvCOMP {algorithm} chunk {index}: status {status}, {size} bytes in a {slot.numel()}-byte slot"
            )
    return nbytes


def _split(items: list, lengths: list[int]) -> list[list]:
    groups, start = [], 0
    for length in lengths:
        groups.append(items[start : start + length])
        start += length
    return groups
