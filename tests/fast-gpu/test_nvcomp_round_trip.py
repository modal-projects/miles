from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60,
    suite="stage-b-2-gpu-h200",
    labels=["weight-update"],
    hardware=["hopper", "blackwell"],
    num_gpus=1,
)


import pyarrow
import pytest
import torch
import zstandard

from miles.utils.nvcomp import MAX_CHUNK_NBYTES, NvcompCompressor


def _decode(algorithm: str, payload: bytes, nbytes: int) -> bytes:
    # Engines undo the outer Zstd with zstandard; pyarrow's Snappy is Google's raw-format decoder.
    if algorithm == "zstd":
        return zstandard.ZstdDecompressor().decompress(payload, max_output_size=nbytes)
    return pyarrow.Codec("snappy").decompress(payload, decompressed_size=nbytes).to_pybytes()


def _chunks(device: torch.device) -> list[torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(0)
    random = torch.randint(0, 256, (1 << 20,), generator=generator, dtype=torch.uint8, device=device)
    sparse = torch.zeros(1 << 20, dtype=torch.uint8, device=device)
    sparse[::4099] = random[: sparse[::4099].numel()]
    zeros = torch.zeros(1 << 20, dtype=torch.uint8, device=device)
    return [random, sparse, zeros, random[:1].clone(), random[:4099].clone()]


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


@pytest.mark.parametrize("algorithm", ["zstd", "snappy"])
def test_every_chunk_decodes_back_with_a_standard_decoder(device: torch.device, algorithm: str) -> None:
    compressor = NvcompCompressor(algorithm, device)
    chunks = _chunks(device)
    stream = torch.cuda.Stream(device)

    batch = compressor.compress(chunks, stream)
    stream.synchronize()

    assert batch.statuses.tolist() == [0] * len(chunks)
    nbytes = batch.nbytes.tolist()
    for chunk, slot, n in zip(chunks, batch.slots, nbytes, strict=True):
        assert 0 < n <= slot.numel()
        assert _decode(algorithm, slot[:n].cpu().numpy().tobytes(), chunk.numel()) == chunk.cpu().numpy().tobytes()
    assert nbytes[2] < (1 << 20) // 10, "an all-zero MiB must compress"
    spans = sorted((slot.data_ptr(), slot.data_ptr() + slot.numel()) for slot in batch.slots)
    assert all(start % compressor._alignments.output == 0 for start, _ in spans)
    assert all(end <= next_start for (_, end), (next_start, _) in zip(spans, spans[1:]))


@pytest.mark.parametrize("case", ["float", "strided", "empty", "too large", "on the CPU", "misaligned"])
def test_a_chunk_nvcomp_cannot_take_is_refused_by_its_index(device: torch.device, case: str) -> None:
    compressor = NvcompCompressor("zstd", device)
    if case == "misaligned" and compressor._alignments.input == 1:
        pytest.skip("nvCOMP's Zstd takes any input alignment")
    bad = {
        "float": lambda: torch.zeros(16, device=device),
        "strided": lambda: torch.zeros(32, dtype=torch.uint8, device=device)[::2],
        "empty": lambda: torch.zeros(0, dtype=torch.uint8, device=device),
        "too large": lambda: torch.zeros(MAX_CHUNK_NBYTES + 1, dtype=torch.uint8, device=device),
        "on the CPU": lambda: torch.zeros(16, dtype=torch.uint8),
        "misaligned": lambda: torch.zeros(32, dtype=torch.uint8, device=device)[1:17],
    }[case]()

    with pytest.raises(ValueError, match="^chunk 1 "):
        compressor.compress([torch.zeros(16, dtype=torch.uint8, device=device), bad], torch.cuda.Stream(device))


def test_an_empty_batch_queues_nothing(device: torch.device) -> None:
    batch = NvcompCompressor("zstd", device).compress([], torch.cuda.Stream(device))

    assert batch.slots == [] and batch.nbytes.numel() == 0 and batch.statuses.numel() == 0
