"""A batch of XOR tensors encoded on the GPU into weight-delta/v1 records: each tensor's changed
frames, their inner encodings, and the tensor's outer Zstd chunks."""

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
    """What OwnerWriter.add_xor takes: the changed frames, their compressed bytes, and its chunks; and
    how many of the tensor's bytes changed, and which when the encoder was asked (changed_bits)."""

    frames: list[dict]
    payload: memoryview
    zstd: dict
    changed_nbytes: int
    changed_bits: torch.Tensor | None


class DeltaEncoder:
    """``encode`` runs on the device's current stream and returns once its work there is done."""

    def __init__(self, codec: str, device: torch.device, with_changed_bits: bool = False):
        if codec not in CODECS:
            raise ValueError(f"weight-delta codec must be one of {CODECS}, got {codec!r}")
        self.device = torch.device(device)
        self._with_changed_bits = with_changed_bits
        self._snappy = NvcompCompressor("snappy", self.device) if codec == "snappy-zstd" else None
        self._zstd = NvcompCompressor("zstd", self.device)
        self._padding = torch.zeros(ALIGNMENT - 1, dtype=torch.uint8, device=self.device)

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
                xors = [previous.to(self.device, non_blocking=True) for previous, _ in pairs]
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
                payloads, zstds = self._compress(inners, stream)
            except BaseException:
                # Queued work may still read the caller's tensors; let it finish before they are released.
                stream.synchronize()
                raise
        for index, layout, payload, zstd, tensor_bits in zip(
            changed_tensors, layouts, payloads, zstds, bits, strict=True
        ):
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
            encoded[index] = EncodedTensor(frame_records, payload, zstd, sum(frame_counts[index]), tensor_bits)
        return encoded

    def _encode_gap(
        self, changes: FrameChanges, frame_counts: list[list[int]]
    ) -> tuple[list[torch.Tensor], list[list[tuple[int, int]]]]:
        """Each changed tensor's inner, all in one buffer the kernel writes, and its frames'
        (inner_offset, inner_nbytes)."""
        offsets, spans, layouts, end = [], [], [], 0
        for counts in frame_counts:
            end += -end % ALIGNMENT
            start, layout = end, []
            for count in counts:
                if count:
                    end += -end % ALIGNMENT
                    layout.append((end - start, (GAP_NBYTES + 1) * count))
                offsets.append(end)
                end += (GAP_NBYTES + 1) * count
            if layout:
                spans.append((start, end))
                layouts.append(layout)
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
        """Each changed tensor's inner, packed from one nvCOMP call, and its frames' (inner_offset,
        inner_nbytes)."""
        frames = [
            [xors[index][i * FRAME_NBYTES : (i + 1) * FRAME_NBYTES] for i in changed_frames[index]]
            for index in changed_tensors
        ]
        batch = self._snappy.compress([frame for tensor_frames in frames for frame in tensor_frames], stream)
        compressed = [slot[:size] for slot, size in zip(batch.slots, _compressed_nbytes(batch, "snappy"), strict=True)]
        by_tensor = _split(compressed, [len(tensor_frames) for tensor_frames in frames])
        packed = [self._pack(frame_inners) for frame_inners in by_tensor]
        return [inner for inner, _ in packed], [layout for _, layout in packed]

    def _pack(self, frame_inners: list[torch.Tensor]) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """One tensor's frame inners back to back, each at a multiple of ALIGNMENT, and their
        (inner_offset, inner_nbytes)."""
        pieces, layout, total = [], [], 0
        for inner in frame_inners:
            if pad := -total % ALIGNMENT:
                pieces.append(self._padding[:pad])
                total += pad
            layout.append((total, inner.numel()))
            pieces.append(inner)
            total += inner.numel()
        return torch.cat(pieces), layout

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
