import pytest
import torch

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-b-cpu", labels=["weight-update"])

from miles.backends.training_utils.weight_update.protocols.weight_delta.encoder import GAP_NBYTES, encode_gap_frame

# Changes at 0, 300 and 70300: gaps 0, 0x00012C and 0x011170, then the values. Engines' gap
# decoder must read exactly these bytes.
GOLDEN_PAYLOAD = bytes.fromhex("002c70" "000111" "000001" "112233")


def _decode(payload: torch.Tensor, nbytes: int) -> torch.Tensor:
    count = payload.numel() // (GAP_NBYTES + 1)
    planes = payload[: GAP_NBYTES * count].view(GAP_NBYTES, count).to(torch.int64)
    gaps = sum(planes[plane] << 8 * plane for plane in range(GAP_NBYTES))
    frame = torch.zeros(nbytes, dtype=torch.uint8)
    frame[torch.cumsum(gaps, 0)] = payload[GAP_NBYTES * count :]
    return frame


def _frame(nbytes: int, positions: list[int], seed: int = 0) -> torch.Tensor:
    frame = torch.zeros(nbytes, dtype=torch.uint8)
    values = torch.randint(1, 256, (len(positions),), generator=torch.Generator().manual_seed(seed), dtype=torch.uint8)
    frame[positions] = values
    return frame


def test_the_payload_is_the_gap_planes_then_the_values():
    frame = torch.zeros(1 << 20, dtype=torch.uint8)
    frame[[0, 300, 70300]] = torch.tensor([0x11, 0x22, 0x33], dtype=torch.uint8)

    assert encode_gap_frame(frame).numpy().tobytes() == GOLDEN_PAYLOAD


@pytest.mark.parametrize(
    "frame",
    [
        _frame(1 << 20, torch.randperm(1 << 20, generator=torch.Generator().manual_seed(1))[:10_000].tolist()),
        _frame(4099, list(range(4099))),
        _frame(4099, [7, 4098]),
        _frame(1 << 20, [(1 << 20) - 1]),
        _frame(1 << 24, [(1 << 24) - 1]),
        _frame(1 << 20, []),
    ],
    ids=["sparse", "every byte", "short last frame", "last byte of a frame", "largest gap", "no change"],
)
def test_a_frame_decodes_back_from_four_bytes_per_change(frame: torch.Tensor):
    payload = encode_gap_frame(frame)

    assert payload.numel() == (GAP_NBYTES + 1) * int(torch.count_nonzero(frame))
    assert torch.equal(_decode(payload, frame.numel()), frame)


@pytest.mark.parametrize(
    "frame",
    [
        torch.zeros((1 << 24) + 1, dtype=torch.uint8),
        torch.zeros(16, dtype=torch.int16),
        torch.zeros(4, 4, dtype=torch.uint8),
    ],
    ids=["over 16 MiB", "not uint8", "not 1-D"],
)
def test_a_frame_whose_gaps_or_values_do_not_fit_is_refused(frame: torch.Tensor):
    with pytest.raises(ValueError, match="1-D uint8 of at most 16 MiB"):
        encode_gap_frame(frame)
