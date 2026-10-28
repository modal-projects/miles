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
from miles.utils.nvcomp import CompressedBatch, NvcompCompressor

GAP_NBYTES = 3
CHUNK_NBYTES = 1 << 20


@dataclass(frozen=True)
class EncodedTensor:
    """What OwnerWriter.add_xor takes: the changed frames, their compressed bytes, and its chunks; and
    how many of the tensor's bytes changed."""

    frames: list[dict]
    payload: memoryview
    zstd: dict
    changed_nbytes: int


class DeltaEncoder:
    """``encode`` runs on the device's current stream and returns once its work there is done."""

    def __init__(self, codec: str, device: torch.device):
        if codec not in CODECS:
            raise ValueError(f"weight-delta codec must be one of {CODECS}, got {codec!r}")
        self.device = torch.device(device)
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
                xors = [
                    previous.to(self.device, non_blocking=True).bitwise_xor_(current) for previous, current in pairs
                ]
                frame_counts = _frame_counts(xors)
                changed_frames = [[i for i, count in enumerate(counts) if count] for counts in frame_counts]
                changed_tensors = [index for index, indices in enumerate(changed_frames) if indices]
                if not changed_tensors:
                    return encoded
                frames = [
                    [xors[index][i * FRAME_NBYTES : (i + 1) * FRAME_NBYTES] for i in changed_frames[index]]
                    for index in changed_tensors
                ]
                frame_inners = self._encode_frames(frames, stream)
                inners, inner_offsets = zip(
                    *(self._pack(tensor_inners) for tensor_inners in frame_inners), strict=True
                )
                payloads, zstds = self._compress(inners, stream)
            except BaseException:
                # Queued work may still read the caller's tensors; let it finish before they are released.
                stream.synchronize()
                raise
        for index, tensor_frames, tensor_inners, offsets, payload, zstd in zip(
            changed_tensors, frames, frame_inners, inner_offsets, payloads, zstds, strict=True
        ):
            frame_records = [
                {
                    "offset": i * FRAME_NBYTES,
                    "nbytes": frame.numel(),
                    "inner_offset": offset,
                    "inner_nbytes": inner.numel(),
                }
                for i, frame, inner, offset in zip(
                    changed_frames[index], tensor_frames, tensor_inners, offsets, strict=True
                )
            ]
            encoded[index] = EncodedTensor(frame_records, payload, zstd, sum(frame_counts[index]))
        return encoded

    def _encode_frames(self, frames: list[list[torch.Tensor]], stream: torch.cuda.Stream) -> list[list[torch.Tensor]]:
        if self._snappy is None:
            return [[encode_gap_frame(frame) for frame in tensor_frames] for tensor_frames in frames]
        batch = self._snappy.compress([frame for tensor_frames in frames for frame in tensor_frames], stream)
        inners = [slot[:size] for slot, size in zip(batch.slots, _compressed_nbytes(batch, "snappy"), strict=True)]
        return _split(inners, [len(tensor_frames) for tensor_frames in frames])

    def _pack(self, frame_inners: list[torch.Tensor]) -> tuple[torch.Tensor, list[int]]:
        """One tensor's frame inners back to back, each at a multiple of ALIGNMENT."""
        pieces, offsets, total = [], [], 0
        for inner in frame_inners:
            if pad := -total % ALIGNMENT:
                pieces.append(self._padding[:pad])
                total += pad
            offsets.append(total)
            pieces.append(inner)
            total += inner.numel()
        return torch.cat(pieces), offsets

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


def _frame_counts(xors: list[torch.Tensor]) -> list[list[int]]:
    """Changed bytes in each frame of each tensor, read back to the host at once."""
    per_tensor = []
    for xor in xors:
        full = xor.numel() // FRAME_NBYTES
        counts = [xor[: full * FRAME_NBYTES].view(full, FRAME_NBYTES).count_nonzero(dim=1)]
        if xor.numel() % FRAME_NBYTES:
            counts.append(xor[full * FRAME_NBYTES :].count_nonzero().view(1))
        per_tensor.append(torch.cat(counts))
    flat = torch.cat(per_tensor).tolist() if per_tensor else []
    return _split(flat, [counts.numel() for counts in per_tensor])


def encode_gap_frame(frame: torch.Tensor) -> torch.Tensor:
    """Codec gap's inner encoding of an XOR frame: n changed bytes become 4n, each position's gap
    from the previous one (the first from 0) as a little-endian u24 in three byte planes, then the
    values."""
    if frame.dtype != torch.uint8 or frame.ndim != 1 or frame.numel() > 1 << (8 * GAP_NBYTES):
        raise ValueError(f"a gap frame is 1-D uint8 of at most 16 MiB, got {frame.dtype} {tuple(frame.shape)}")
    positions = torch.nonzero(frame).squeeze(1)
    gaps = torch.diff(positions, prepend=positions.new_zeros(1))
    planes = torch.stack([(gaps >> 8 * plane) & 0xFF for plane in range(GAP_NBYTES)]).to(torch.uint8)
    return torch.cat([planes.reshape(-1), frame[positions]])


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
