"""Gap frames on CUDA: compressed bytes against a CPU reference, decoding through the receiver's prepared
plan and CUDA double-buffer schedule, and corrupt or truncated frames refused.

python -m pytest tests/manual/gpu_delta/test_gap.py -q
Requires CUDA; no nvCOMP library or model is loaded.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from miles.backends.sglang_utils.gpu_delta.receiver import PreparedDelta, _PreparedBatch
from miles.utils.gpu_delta.apply import prepare_status_check
from miles.utils.gpu_delta.codec import GapDecoder
from miles.utils.gpu_delta.gap import MAX_FRAME_BYTES, GapCompressor

MIB = 1 << 20
TILE, BLOCK = 1 << 16, 1 << 12
# The old pipeline's reference frame: changes 0x11, 0x22, 0x33 at 0, 300 and 70300.
GOLDEN = bytes.fromhex("002c70" "000111" "000001" "112233")
POISON = 0xA5


def _changed(nbytes, positions, seed):
    frame = np.zeros(nbytes, dtype=np.uint8)
    frame[positions] = np.random.default_rng(seed).integers(1, 256, len(positions), dtype=np.uint8)
    return frame


def _scattered(nbytes, density, seed):
    rng = np.random.default_rng(seed)
    frame = rng.integers(1, 256, nbytes, dtype=np.uint8)
    frame[rng.random(nbytes) >= density] = 0
    return frame


FRAMES = {
    "a change at 0": _changed(MIB, [0], 0),
    "around block edges": _changed(MIB, [BLOCK - 1, BLOCK, 2 * BLOCK - 1, 2 * BLOCK], 1),
    "around tile edges": _changed(MIB, [TILE - 1, TILE, 2 * TILE - 1, 2 * TILE, MIB - TILE], 2),
    "empty tiles between changes": _changed(MIB, [10, 3 * TILE + 5], 3),
    "a gap past 64 KiB in 4 MiB": _changed(4 * MIB, [5, 3 * MIB + 12], 4),
    "the last byte": _changed(MIB, [MIB - 1], 5),
    "every byte": _changed(3 * TILE + 100, np.arange(3 * TILE + 100), 6),
    "one byte": _changed(1, [0], 7),
    "an odd tail": _changed(139, [0, 1, 2, 137, 138], 8),
    "every byte of 7": _changed(7, np.arange(7), 9),
    "unchanged": np.zeros(MIB, dtype=np.uint8),
    "an unchanged tail": np.zeros(37, dtype=np.uint8),
    **{
        f"{density:.2%} scattered": _scattered(MIB, density, 10 + i)
        for i, density in enumerate((1e-4, 1e-3, 1e-2, 0.1, 0.25, 0.5))
    },
}


def _reference(frame):
    positions = np.flatnonzero(frame)
    gaps = np.diff(positions, prepend=0)
    planes = [((gaps >> (8 * plane)) & 0xFF).astype(np.uint8) for plane in range(3)]
    return np.concatenate([*planes, frame[positions]]).tobytes()


def _compress(frames):
    """Each frame at a 16-byte aligned offset of one poisoned device buffer, as the encoder's frame views."""
    device = torch.device("cuda", torch.cuda.current_device())
    starts = np.cumsum([0] + [-(-len(frame) // 16) * 16 for frame in frames])
    host = np.full(int(starts[-1]), POISON, dtype=np.uint8)
    for start, frame in zip(starts, frames, strict=False):
        host[start : start + len(frame)] = frame
    buffer = torch.from_numpy(host).to(device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        batch = GapCompressor(device).compress(
            [buffer[start : start + len(frame)] for start, frame in zip(starts, frames, strict=False)], stream
        )
    stream.synchronize()
    return batch


def test_the_old_pipelines_golden_frame():
    frame = np.zeros(MIB, dtype=np.uint8)
    frame[[0, 300, 70300]] = [0x11, 0x22, 0x33]
    batch = _compress([frame])
    assert batch.sizes.tolist() == [12] and batch.statuses.tolist() == [0]
    assert batch.outputs[0].cpu().numpy().tobytes() == GOLDEN


def test_compressed_frames_match_the_reference_at_every_edge_and_density():
    frames = list(FRAMES.values())
    batch = _compress(frames)
    assert batch.statuses.tolist() == [0] * len(frames)
    spans = []
    for name, frame, output, size in zip(FRAMES, frames, batch.outputs, batch.sizes.tolist(), strict=True):
        reference = _reference(frame)
        assert size == output.numel() == len(reference) == 4 * np.count_nonzero(frame), name
        assert output.cpu().numpy().tobytes() == reference, name
        if size:
            assert output.data_ptr() % 16 == 0, name
            spans.append((output.data_ptr(), output.data_ptr() + size))
    assert all(end <= start for (_, end), (start, _) in zip(spans, spans[1:], strict=False))


def test_the_compressor_refuses_frames_it_cannot_encode():
    device = torch.device("cuda", torch.cuda.current_device())
    compressor, stream = GapCompressor(device), torch.cuda.Stream(device=device)
    buffer = torch.zeros(MAX_FRAME_BYTES + 32, dtype=torch.uint8, device=device)
    for frames in ([buffer[1:17]], [buffer[: MAX_FRAME_BYTES + 1]], [buffer[:0]], [buffer.cpu()[:16]]):
        with pytest.raises(ValueError, match="16-byte aligned contiguous CUDA uint8"):
            compressor.compress(frames, stream)
    assert compressor.compress([], stream).outputs == []


def _arena(frames, batch):
    """The changed frames' encoded bytes in one pinned host arena: (input offset, encoded, decoded) per frame."""
    payload, rows = bytearray(), []
    for frame, output, size in zip(frames, batch.outputs, batch.sizes.tolist(), strict=True):
        if size:
            payload.extend(bytes(-len(payload) % 16))
            rows.append((len(payload), size, len(frame)))
            payload.extend(output.cpu().numpy().tobytes())
    host = torch.empty(len(payload), dtype=torch.uint8, pin_memory=True)
    host.numpy()[:] = np.frombuffer(payload, dtype=np.uint8)
    return host, rows


def _batches(rows, groups, hole=48):
    """Frame rows per decode batch, each frame 16-byte aligned in its slot after a poisoned hole."""
    batches = []
    for group in groups:
        batch, cursor = [], 0
        for index in group:
            source, encoded, decoded = rows[index]
            cursor = -(-(cursor + hole) // 16) * 16
            batch.append((source, encoded, decoded, cursor))
            cursor += decoded
        batches.append(batch)
    return batches


def _decode(host, batches):
    """The receiver's CUDA double-buffer schedule; each batch's apply snapshots its slot, statuses and sizes."""
    device = torch.device("cuda", torch.cuda.current_device())
    decoder = GapDecoder(device)
    compute, h2d = [torch.cuda.Stream(device=device) for _ in range(2)]
    table = np.asarray([row for batch in batches for row in batch], dtype=np.int64).reshape(-1, 4).T.copy()
    plan = decoder.prepare_batches(table, [len(batch) for batch in batches], host, compute)
    prepared = PreparedDelta(SimpleNamespace(device=device))
    prepared.stream, prepared.io_stream = compute, h2d
    prepared.decoded_ready = [torch.cuda.Event() for _ in range(2)]
    prepared.decoded_free = [torch.cuda.Event() for _ in range(2)]
    snapshots = []

    def snapshot():
        index = len(snapshots)
        slot = slots[index % 2]
        snapshots.append((slot.clone(), plans[index].statuses.clone(), plans[index].actual_sizes.clone()))
        slot.fill_(POISON)

    slot_bytes = max(max((row[3] + row[2] for row in batch), default=16) for batch in batches)
    with torch.cuda.stream(compute):
        slots = [torch.full((slot_bytes,), POISON, dtype=torch.uint8, device=device) for _ in range(2)]
        error = torch.zeros(1, dtype=torch.int32, device=device)
        plans = plan.bind_outputs(slots)
        prepared.batches = [
            _PreparedBatch(
                decoder=batch, apply=snapshot, zero_ranges=[], check_status=prepare_status_check(batch, error)
            )
            for batch in plans
        ]
        h2d.wait_stream(compute)
        prepared._copy_batch(prepared.batches[0], 0)
        prepared._apply_cuda_batches()
    compute.synchronize()
    h2d.synchronize()
    return [
        (slot.cpu().numpy(), statuses.tolist(), sizes.tolist()) for slot, statuses, sizes in snapshots
    ], error.item()


def _expected_slot(slot_bytes, batch, frames):
    expected = np.full(slot_bytes, POISON, dtype=np.uint8)
    for (_, _, decoded, output), frame in zip(batch, frames, strict=True):
        expected[output : output + decoded] = frame
    return expected


def test_frames_decode_exactly_through_the_receivers_cuda_schedule():
    frames = list(FRAMES.values())
    host, rows = _arena(frames, _compress(frames))
    changed = [frame for frame in frames if frame.any()]
    assert len(rows) == len(changed)
    groups = [[0, 1, 2, 3], [4], [5, 6, 7, 8, 9], [], list(range(10, len(rows))), [9, 0]]
    batches = _batches(rows, groups)
    decoded, error = _decode(host, batches)
    assert error == 0
    for (slot, statuses, sizes), batch, group in zip(decoded, batches, groups, strict=True):
        assert statuses == [0] * len(batch)
        assert sizes == [row[2] for row in batch]
        # Every frame byte is decoded and no byte outside a frame is written.
        np.testing.assert_array_equal(slot, _expected_slot(len(slot), batch, [changed[i] for i in group]))


def _cut_off_a_change(encoded):
    return encoded[:-4]


def _corrupt(plane, index, value):
    def corrupt(encoded):
        changes = len(encoded) // 4
        encoded = bytearray(encoded)
        encoded[plane * changes + index] = value
        return bytes(encoded)

    return corrupt


def _repeat_a_position(encoded):
    return _corrupt(2, 1, 0)(_corrupt(1, 1, 0)(_corrupt(0, 1, 0)(encoded)))


@pytest.mark.parametrize(
    "corruption",
    [_corrupt(2, 3, 0xFF), _repeat_a_position, _corrupt(3, 2, 0), _cut_off_a_change],
    ids=["position past the frame", "repeated position", "zero value", "a change cut off"],
)
def test_a_corrupt_frame_sets_the_error_gate_and_writes_only_inside_its_frame(corruption):
    frames = [_changed(MIB, [5, 900, 70000, 500000], 3), _changed(MIB, [17, 300000], 4)]
    batch = _compress(frames)
    encoded = [output.cpu().numpy().tobytes() for output in batch.outputs]
    encoded[0] = corruption(encoded[0])
    payload = encoded[0] + bytes(-len(encoded[0]) % 16) + encoded[1]
    host = torch.empty(len(payload), dtype=torch.uint8, pin_memory=True)
    host.numpy()[:] = np.frombuffer(payload, dtype=np.uint8)
    rows = [(0, len(encoded[0]), MIB), (len(payload) - len(encoded[1]), len(encoded[1]), MIB)]
    batches = _batches(rows, [[0, 1]])
    [(slot, statuses, _)], error = _decode(host, batches)
    assert error == 1 and statuses[0] != 0 and statuses[1] == 0
    (_, _, _, first), (_, _, _, second) = batches[0]
    outside = np.ones(len(slot), dtype=bool)
    outside[first : first + MIB] = outside[second : second + MIB] = False
    assert np.all(slot[outside] == POISON)
    np.testing.assert_array_equal(slot[second : second + MIB], frames[1])


@pytest.mark.parametrize(
    "encoded,decoded,error",
    [(10, 64, "four bytes per change"), (260, 64, "at most one change"), (0, 64, "positive lengths")],
    ids=["cut inside a change", "more changes than bytes", "empty"],
)
def test_frames_that_are_not_whole_changes_fail_before_the_pause(encoded, decoded, error):
    device = torch.device("cuda", torch.cuda.current_device())
    host = torch.zeros(512, dtype=torch.uint8, pin_memory=True)
    table = np.asarray([[0], [encoded], [decoded], [0]], dtype=np.int64)
    with pytest.raises(ValueError, match=error):
        GapDecoder(device).prepare_batches(table, [1], host, torch.cuda.Stream(device=device))
