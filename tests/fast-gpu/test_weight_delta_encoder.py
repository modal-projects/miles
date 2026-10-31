from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60,
    suite="stage-b-2-gpu-h200",
    labels=["weight-update"],
    hardware=["hopper", "blackwell"],
    num_gpus=1,
)


import numpy as np
import pyarrow
import pytest
import torch
import zstandard

from miles.backends.training_utils.weight_update.protocols.weight_delta.encoder import CHUNK_NBYTES, DeltaEncoder
from miles.backends.training_utils.weight_update.protocols.weight_delta.publication import (
    ALIGNMENT,
    FRAME_NBYTES,
    checksum,
)
from miles.kernels.weight_delta.frames import GAP_NBYTES

MIB = 1 << 20
MAX_PENDING_NBYTES = 512 * MIB
# name -> (bytes, changed positions)
CASES = {
    "frames 0 and 2 of 3, the last short": (2 * MIB + 12345, [5, 70000, 2 * MIB + 7, 2 * MIB + 12344]),
    "unchanged, between changed ones": (3 * MIB // 2, []),
    "dense, its inner over several chunks": (3 * MIB, list(range(0, 3 * MIB, 2))),
    "one byte of a tiny tensor": (100, [99]),
    "empty": (0, []),
}


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _pair(nbytes: int, changed: list[int], seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    previous = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, generator=generator)
    current = previous.clone()
    current[changed] ^= torch.randint(1, 256, (len(changed),), dtype=torch.uint8, generator=generator)
    return (previous.pin_memory() if nbytes else previous), current.to(device)


def _undo_zstd(payload: memoryview, zstd: dict) -> bytes:
    """The tensor's inner, from its chunks in the order and at the places the records give."""
    payload, pieces, offset, inner_offset = bytes(payload), [], 0, 0
    for chunk in zstd["chunks"]:
        assert (chunk["offset"], chunk["inner_offset"]) == (offset, inner_offset)
        assert 0 < chunk["inner_nbytes"] <= CHUNK_NBYTES
        data = zstandard.ZstdDecompressor().decompress(
            payload[offset : offset + chunk["nbytes"]], max_output_size=chunk["inner_nbytes"]
        )
        assert len(data) == chunk["inner_nbytes"]
        pieces.append(data)
        offset += chunk["nbytes"]
        inner_offset += chunk["inner_nbytes"]
    assert (offset, inner_offset) == (len(payload), zstd["inner_nbytes"])
    return b"".join(pieces)


def _undo_gap(frame_inner: bytes, nbytes: int) -> bytes:
    count = len(frame_inner) // (GAP_NBYTES + 1)
    inner = torch.frombuffer(bytearray(frame_inner), dtype=torch.uint8)
    planes = inner[: GAP_NBYTES * count].view(GAP_NBYTES, count).to(torch.int64)
    gaps = sum(planes[plane] << 8 * plane for plane in range(GAP_NBYTES))
    frame = torch.zeros(nbytes, dtype=torch.uint8)
    frame[torch.cumsum(gaps, 0)] = inner[GAP_NBYTES * count :]
    return frame.numpy().tobytes()


@pytest.mark.parametrize("codec", ["gap-zstd", "snappy-zstd"])
def test_exactly_the_changed_frames_decode_back_from_each_payload(device: torch.device, codec: str):
    pairs = [_pair(nbytes, changed, seed, device) for seed, (nbytes, changed) in enumerate(CASES.values())]
    snapshots = [previous.clone() for previous, _ in pairs]

    encoder = DeltaEncoder(codec, device, MAX_PENDING_NBYTES, with_changed_bits=True)
    encoded = encoder.encode(pairs)
    payloads, zstds = encoder.finish()
    torch.cuda.synchronize()

    assert len(payloads) == sum(result is not None for result in encoded)
    compressed = iter(zip(payloads, zstds, strict=True))
    for name, (previous, current), snapshot, result in zip(CASES, pairs, snapshots, encoded, strict=True):
        assert torch.equal(previous, current.cpu()), f"{name}: previous does not hold the current bytes"
        xor = snapshot ^ current.cpu()
        changed_frames = [i for i, frame in enumerate(xor.split(FRAME_NBYTES)) if frame.any()]
        if not changed_frames:
            assert result is None, name
            continue
        assert [record["offset"] for record in result.frames] == [i * FRAME_NBYTES for i in changed_frames], name
        assert result.changed_nbytes == int(xor.count_nonzero()), name
        assert result.checksum == checksum(current.cpu().numpy(), changed=(xor != 0).numpy()), name
        assert torch.equal(result.changed_bits.cpu(), torch.from_numpy(np.packbits(xor.numpy() != 0))), name
        payload, zstd = next(compressed)
        inner = _undo_zstd(payload, zstd)
        for record in result.frames:
            frame = xor[record["offset"] : record["offset"] + FRAME_NBYTES]
            assert record["nbytes"] == frame.numel() and record["inner_offset"] % ALIGNMENT == 0, name
            frame_inner = inner[record["inner_offset"] : record["inner_offset"] + record["inner_nbytes"]]
            if codec == "gap-zstd":
                assert len(frame_inner) == (GAP_NBYTES + 1) * int(frame.count_nonzero()), name
                assert _undo_gap(frame_inner, frame.numel()) == frame.numpy().tobytes(), name
            else:
                decoded = pyarrow.Codec("snappy").decompress(frame_inner, decompressed_size=frame.numel())
                assert decoded.to_pybytes() == frame.numpy().tobytes(), name
    assert len(zstds[1]["chunks"]) > 1


@pytest.mark.parametrize("codec", ["gap-zstd", "snappy-zstd"])
def test_the_payloads_are_the_same_whether_zstd_runs_once_or_after_every_batch(device: torch.device, codec: str):
    # Each encoder gets its own pairs: encoding overwrites previous.
    pairs_once, pairs_every_batch = (
        [_pair(nbytes, changed, seed, device) for seed, (nbytes, changed) in enumerate(CASES.values())]
        for _ in range(2)
    )
    once, every_batch = DeltaEncoder(codec, device, MAX_PENDING_NBYTES), DeltaEncoder(codec, device, 1)

    encoded_once = once.encode(pairs_once)
    encoded_every_batch = [result for pair in pairs_every_batch for result in every_batch.encode([pair])]

    assert [result and result.frames for result in encoded_once] == [
        result and result.frames for result in encoded_every_batch
    ]
    (payloads_once, zstds_once), (payloads_every_batch, zstds_every_batch) = once.finish(), every_batch.finish()
    assert [bytes(payload) for payload in payloads_once] == [bytes(payload) for payload in payloads_every_batch]
    assert zstds_once == zstds_every_batch


@pytest.mark.parametrize(
    "case", ["previous on the GPU", "previous not pinned", "lengths differ", "not uint8", "current on the host"]
)
def test_a_pair_the_encoder_cannot_take_is_refused_by_its_index(device: torch.device, case: str):
    good = (torch.zeros(16, dtype=torch.uint8).pin_memory(), torch.zeros(16, dtype=torch.uint8, device=device))
    bad = {
        "previous on the GPU": lambda: (torch.ones(16, dtype=torch.uint8, device=device), good[1]),
        "previous not pinned": lambda: (torch.ones(16, dtype=torch.uint8), good[1]),
        "lengths differ": lambda: (good[0], torch.zeros(8, dtype=torch.uint8, device=device)),
        "not uint8": lambda: (
            torch.ones(8, dtype=torch.int16).pin_memory(),
            torch.zeros(8, dtype=torch.int16, device=device),
        ),
        "current on the host": lambda: (good[0], torch.zeros(16, dtype=torch.uint8)),
    }[case]()

    with pytest.raises(ValueError, match="^pair 1: "):
        DeltaEncoder("gap-zstd", device, MAX_PENDING_NBYTES).encode([good, bad])
    assert bad[0].eq(bad[0].flatten()[0]).all(), "a refused previous must be left as it was"


def test_a_current_off_a_word_boundary_encodes_as_an_aligned_one(device: torch.device):
    nbytes, changed = 2 * MIB + 3, [1, 70001, 2 * MIB + 2]
    (previous, current), (previous_too, _) = (_pair(nbytes, changed, 0, device) for _ in range(2))
    unaligned = torch.empty(nbytes + 1, dtype=torch.uint8, device=device)[1:].copy_(current)
    aligned, off_word = (DeltaEncoder("gap-zstd", device, MAX_PENDING_NBYTES) for _ in range(2))

    (expected,), (result,) = aligned.encode([(previous, current)]), off_word.encode([(previous_too, unaligned)])

    assert (result.frames, result.changed_nbytes, result.checksum) == (
        expected.frames,
        expected.changed_nbytes,
        expected.checksum,
    )
    assert [bytes(payload) for payload in off_word.finish()[0]] == [bytes(payload) for payload in aligned.finish()[0]]


def test_a_payload_stays_intact_while_other_pinned_memory_is_reused(device: torch.device):
    encoder = DeltaEncoder("gap-zstd", device, MAX_PENDING_NBYTES)
    encoder.encode([_pair(MIB, list(range(0, MIB, 7)), 0, device)])
    (payload,), _ = encoder.finish()
    expected = bytes(payload)

    for _ in range(4):
        torch.empty(len(expected), dtype=torch.uint8, pin_memory=True).fill_(0xAB)

    assert bytes(payload) == expected


def test_an_empty_batch_encodes_to_nothing(device: torch.device):
    encoder = DeltaEncoder("gap-zstd", device, MAX_PENDING_NBYTES)

    assert encoder.encode([]) == []
    assert encoder.finish() == ([], [])
