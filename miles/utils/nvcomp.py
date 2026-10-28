"""Batched Zstd and Snappy compression on the GPU through nvCOMP's C API."""

import ctypes
import functools
import importlib.metadata
from collections.abc import Sequence
from dataclasses import dataclass

import torch

NVCOMP_VERSION = "5.3.0.16"
# Snappy's limit; Zstd allows larger chunks.
MAX_CHUNK_NBYTES = 1 << 24
_ALGORITHMS = {"zstd": "Zstd", "snappy": "Snappy"}


class _CompressOpts(ctypes.Structure):
    # nvCOMP 5.3's Zstd and Snappy compression options are both reserved bytes only.
    _fields_ = [("reserved", ctypes.c_char * 64)]


class _AlignmentRequirements(ctypes.Structure):
    _fields_ = [("input", ctypes.c_size_t), ("output", ctypes.c_size_t), ("temp", ctypes.c_size_t)]


def nvcomp_distribution() -> importlib.metadata.Distribution:
    """The nvCOMP wheel for this PyTorch's CUDA, or why it cannot be used."""
    if torch.version.cuda is None:
        raise RuntimeError("nvCOMP needs a CUDA build of PyTorch")
    package = f"nvidia-libnvcomp-cu{torch.version.cuda.split('.')[0]}"
    try:
        distribution = importlib.metadata.distribution(package)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(f"nvCOMP is not installed: pip install --no-deps {package}=={NVCOMP_VERSION}") from error
    if not (5, 3) <= tuple(int(part) for part in distribution.version.split(".")[:2]) < (6, 0):
        raise RuntimeError(f"nvCOMP's 5.3 C API is required, found {package} {distribution.version}")
    return distribution


@functools.cache
def _library() -> ctypes.CDLL:
    return ctypes.CDLL(str(nvcomp_distribution().locate_file("nvidia/libnvcomp/lib64/libnvcomp.so.5")))


@dataclass(frozen=True)
class CompressedBatch:
    """Per chunk: a slot whose first ``nbytes[i]`` bytes are the payload, and nvCOMP's status (0
    is success), all on the GPU and valid once the stream reaches them. Keep the batch until then:
    ``keepalive`` holds every buffer the queued work uses."""

    slots: list[torch.Tensor]
    nbytes: torch.Tensor
    statuses: torch.Tensor
    keepalive: tuple


class NvcompCompressor:
    """One algorithm on one device. ``compress`` only queues work on the given stream."""

    def __init__(self, algorithm: str, device: torch.device):
        if algorithm not in _ALGORITHMS:
            raise ValueError(f"nvCOMP algorithm must be one of {sorted(_ALGORITHMS)}, got {algorithm!r}")
        self.algorithm, self.device = algorithm, torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError(f"nvCOMP compression needs an explicit CUDA device, got {self.device}")
        size, pointer = ctypes.c_size_t, ctypes.c_void_p
        self._nvcomp_max_output_chunk_size = self._bind(
            "GetMaxOutputChunkSize", [size, _CompressOpts, ctypes.POINTER(size)]
        )
        self._nvcomp_temp_size = self._bind(
            "GetTempSizeAsync", [size, size, _CompressOpts, ctypes.POINTER(size), size]
        )
        self._nvcomp_compress = self._bind(
            "Async", [pointer, pointer, size, size, pointer, size, pointer, pointer, _CompressOpts, pointer, pointer]
        )
        nvcomp_alignments = self._bind(
            "GetRequiredAlignments", [_CompressOpts, ctypes.POINTER(_AlignmentRequirements)]
        )
        self._opts = _CompressOpts()
        self._alignments = _AlignmentRequirements()
        self._check(nvcomp_alignments(self._opts, ctypes.byref(self._alignments)))

    def _bind(self, suffix: str, argtypes: list):
        function = _library()[f"nvcompBatched{_ALGORITHMS[self.algorithm]}Compress{suffix}"]
        function.argtypes, function.restype = argtypes, ctypes.c_int
        return function

    def _check(self, status: int) -> None:
        if status:
            raise RuntimeError(f"nvCOMP {self.algorithm} compression failed: status {status}")

    def _max_output_nbytes(self, nbytes: int) -> int:
        size = ctypes.c_size_t()
        self._check(self._nvcomp_max_output_chunk_size(nbytes, self._opts, ctypes.byref(size)))
        return size.value

    def _temp_nbytes(self, count: int, max_nbytes: int, total_nbytes: int) -> int:
        size = ctypes.c_size_t()
        self._check(self._nvcomp_temp_size(count, max_nbytes, self._opts, ctypes.byref(size), total_nbytes))
        return size.value

    def compress(self, chunks: Sequence[torch.Tensor], stream: torch.cuda.Stream) -> CompressedBatch:
        """Each chunk (contiguous CUDA uint8, aligned, 1 byte to 16 MiB) becomes one compressed
        chunk in a standard format: a Zstd frame or a raw Snappy block."""
        if stream.device != self.device:
            raise ValueError(f"stream on {stream.device}, compressor on {self.device}")
        for index, chunk in enumerate(chunks):
            if (
                chunk.device != self.device
                or chunk.dtype != torch.uint8
                or not chunk.is_contiguous()
                or not 0 < chunk.numel() <= MAX_CHUNK_NBYTES
                or chunk.data_ptr() % self._alignments.input
            ):
                raise ValueError(
                    f"chunk {index} ({chunk.dtype} {tuple(chunk.shape)} on {chunk.device} at {chunk.data_ptr():#x}) "
                    f"is not contiguous uint8 on {self.device}, {self._alignments.input}-byte aligned, "
                    f"1 byte to {MAX_CHUNK_NBYTES >> 20} MiB"
                )
        chunks = tuple(chunks)
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            nbytes = torch.empty(len(chunks), dtype=torch.int64, device=self.device)
            statuses = torch.empty(len(chunks), dtype=torch.int32, device=self.device)
            if not chunks:
                return CompressedBatch([], nbytes, statuses, ())
            chunk_nbytes = [chunk.numel() for chunk in chunks]
            max_nbytes = max(chunk_nbytes)
            max_output_nbytes = {n: self._max_output_nbytes(n) for n in set(chunk_nbytes)}
            offsets, total = [], 0
            for n in chunk_nbytes:
                total += -total % self._alignments.output
                offsets.append(total)
                total += max_output_nbytes[n]
            output = torch.empty(total, dtype=torch.uint8, device=self.device)
            temp_nbytes = self._temp_nbytes(len(chunks), max_nbytes, sum(chunk_nbytes))
            temp = torch.empty(temp_nbytes, dtype=torch.uint8, device=self.device)
            assert output.data_ptr() % self._alignments.output == 0 and temp.data_ptr() % self._alignments.temp == 0
            slots = [
                output[offset : offset + max_output_nbytes[n]] for offset, n in zip(offsets, chunk_nbytes, strict=True)
            ]
            host_arguments = torch.tensor(
                [[chunk.data_ptr() for chunk in chunks], chunk_nbytes, [slot.data_ptr() for slot in slots]],
                dtype=torch.int64,
                pin_memory=True,
            )
            arguments = host_arguments.to(self.device, non_blocking=True)
            try:
                self._check(
                    self._nvcomp_compress(
                        arguments[0].data_ptr(),
                        arguments[1].data_ptr(),
                        max_nbytes,
                        len(chunks),
                        temp.data_ptr(),
                        temp_nbytes,
                        arguments[2].data_ptr(),
                        nbytes.data_ptr(),
                        self._opts,
                        statuses.data_ptr(),
                        stream.cuda_stream,
                    )
                )
            except Exception:
                # Work may already be queued; let it finish before the caller drops the buffers.
                stream.synchronize()
                raise
        return CompressedBatch(slots, nbytes, statuses, (chunks, output, temp, host_arguments, arguments))
