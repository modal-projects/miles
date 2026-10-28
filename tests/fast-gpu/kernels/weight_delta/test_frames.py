"""The weight-delta frame kernels against a torch reference: XOR and changed bytes per frame, and
codec gap's bytes at each frame's offset, including the golden bytes engines decode."""

import pytest
import torch

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60, suite="stage-b-2-gpu-h200", labels=["weight-update"], hardware=["hopper", "blackwell"], num_gpus=1
)

from miles.kernels.weight_delta.frames import GAP_NBYTES, TILE_NBYTES, encode_gap_frames, xor_frames_

MIB = 1 << 20
BLOCK = 1 << 12
FILL = 0xAB
# Changes at 0, 300 and 70300: gaps 0, 0x00012C and 0x011170, then the values. Engines' gap
# decoder must read exactly these bytes.
GOLDEN_PAYLOAD = bytes.fromhex("002c70" "000111" "000001" "112233")
# name -> (bytes, changed positions)
CASES = {
    "a change at 0": (MIB, [0]),
    "around block edges": (MIB, [BLOCK - 1, BLOCK, 2 * BLOCK - 1, 2 * BLOCK]),
    "around tile edges": (MIB, [TILE_NBYTES - 1, TILE_NBYTES, 2 * TILE_NBYTES - 1, 2 * TILE_NBYTES]),
    "empty tiles between changes": (MIB, [10, 3 * TILE_NBYTES + 5]),
    "every byte": (3 * TILE_NBYTES + 100, list(range(3 * TILE_NBYTES + 100))),
    "sparse": (MIB, torch.randperm(MIB, generator=torch.Generator().manual_seed(0))[:10_000].tolist()),
    "short last frame": (2 * MIB + 4099, [7, MIB + 3, 2 * MIB + 4098]),
    "last byte of a frame": (MIB, [MIB - 1]),
    "unchanged": (MIB + 5, []),
    "empty": (0, []),
}


def _pair(nbytes: int, changed: list[int], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    previous = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, generator=generator)
    current = previous.clone()
    current[changed] ^= torch.randint(1, 256, (len(changed),), dtype=torch.uint8, generator=generator)
    return previous.cuda(), current.cuda()


def _gap_frame_reference(frame: torch.Tensor) -> bytes:
    positions = torch.nonzero(frame).squeeze(1)
    gaps = torch.diff(positions, prepend=positions.new_zeros(1))
    planes = torch.stack([(gaps >> 8 * plane) & 0xFF for plane in range(GAP_NBYTES)]).to(torch.uint8)
    return torch.cat([planes.reshape(-1), frame[positions]]).cpu().numpy().tobytes()


def _encode(pairs: list[tuple[torch.Tensor, torch.Tensor]], frame_nbytes: int = MIB):
    """Both kernels over a batch, each frame's encoding 16 bytes past the previous one's end, on a
    buffer filled with FILL. Returns the targets, counts, buffer and offsets."""
    targets = [previous.clone() for previous, _ in pairs]
    changes = xor_frames_(targets, [current for _, current in pairs], frame_nbytes)
    counts = changes.counts.tolist()
    offsets, end = [], 0
    for count in counts:
        offsets.append(end)
        end += (GAP_NBYTES + 1) * count + 16
    output = torch.full((end,), FILL, dtype=torch.uint8, device="cuda")
    encode_gap_frames(changes, output, offsets)
    return targets, counts, output.cpu(), offsets


def _check(pairs, targets, counts, output, offsets, frame_nbytes: int = MIB) -> None:
    written = torch.zeros(output.numel(), dtype=torch.bool)
    frame = 0
    for (previous, current), target in zip(pairs, targets, strict=True):
        xor = previous ^ current
        assert torch.equal(target, xor)
        for start in range(0, xor.numel(), frame_nbytes):
            reference = xor[start : start + frame_nbytes]
            assert counts[frame] == int(reference.count_nonzero())
            nbytes = (GAP_NBYTES + 1) * counts[frame]
            assert output[offsets[frame] : offsets[frame] + nbytes].numpy().tobytes() == _gap_frame_reference(
                reference
            )
            written[offsets[frame] : offsets[frame] + nbytes] = True
            frame += 1
    assert frame == len(counts)
    assert bool((output[~written] == FILL).all()), "a kernel wrote outside a changed frame's encoding"


def test_the_golden_frame_is_the_gap_planes_then_the_values():
    previous = torch.zeros(MIB, dtype=torch.uint8, device="cuda")
    current = previous.clone()
    current[[0, 300, 70300]] = torch.tensor([0x11, 0x22, 0x33], dtype=torch.uint8, device="cuda")

    _, counts, output, offsets = _encode([(previous, current)])

    assert counts == [3]
    assert output[offsets[0] : offsets[0] + len(GOLDEN_PAYLOAD)].numpy().tobytes() == GOLDEN_PAYLOAD


@pytest.mark.parametrize("case", list(CASES))
def test_a_tensor_matches_the_reference(case: str):
    pairs = [_pair(*CASES[case], seed=0)]
    others = [current.clone() for _, current in pairs]

    _check(pairs, *_encode(pairs))
    assert all(torch.equal(current, other) for (_, current), other in zip(pairs, others, strict=True))


def test_a_batch_of_every_case_matches_the_reference():
    pairs = [_pair(nbytes, changed, seed) for seed, (nbytes, changed) in enumerate(CASES.values())]

    _check(pairs, *_encode(pairs))


def test_the_largest_gap_fits_a_16_mib_frame():
    pairs = [_pair(1 << 24, [(1 << 24) - 1], seed=0)]

    targets, counts, output, offsets = _encode(pairs, frame_nbytes=1 << 24)

    assert output[offsets[0] : offsets[0] + GAP_NBYTES].numpy().tobytes() == b"\xff\xff\xff"
    _check(pairs, targets, counts, output, offsets, frame_nbytes=1 << 24)


@pytest.mark.parametrize(
    "case",
    [
        "frame not a power of two",
        "frame under a tile",
        "lengths differ",
        "not uint8",
        "offsets not one per frame",
        "gap frame over 16 MiB",
    ],
)
def test_what_the_kernels_cannot_take_is_refused(case: str):
    tensor = torch.zeros(MIB, dtype=torch.uint8, device="cuda")
    call, match = {
        "frame not a power of two": (lambda: xor_frames_([tensor], [tensor.clone()], 3 * TILE_NBYTES), "power of two"),
        "frame under a tile": (lambda: xor_frames_([tensor], [tensor.clone()], TILE_NBYTES // 2), "power of two"),
        "lengths differ": (lambda: xor_frames_([tensor], [tensor[:-1].clone()], MIB), "^pair 0: "),
        "not uint8": (lambda: xor_frames_([tensor.view(torch.int16)], [tensor.view(torch.int16)], MIB), "^pair 0: "),
        "offsets not one per frame": (
            lambda: encode_gap_frames(xor_frames_([tensor], [tensor.clone()], MIB), tensor.clone(), [0, 0]),
            "one per frame",
        ),
        "gap frame over 16 MiB": (
            lambda: encode_gap_frames(xor_frames_([], [], 1 << 25), tensor.clone(), []),
            "at most 16 MiB",
        ),
    }[case]

    with pytest.raises(ValueError, match=match):
        call()
