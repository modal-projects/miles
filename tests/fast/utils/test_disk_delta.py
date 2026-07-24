import numpy as np

from miles.utils.disk_delta import sparse_xor_encode


def test_sparse_xor_encode_round_trip():
    diff = np.zeros(64, dtype=np.uint8)
    diff[[0, 7, 63]] = [1, 0x80, 0xFF]

    encoded = sparse_xor_encode(diff)
    count = int(np.frombuffer(encoded[:8], dtype="<u8")[0])
    positions = np.frombuffer(encoded, dtype="<u8", count=count, offset=8)
    values = encoded[8 + 8 * count :]
    reconstructed = np.zeros_like(diff)
    reconstructed[positions] = values

    assert count == 3
    assert np.array_equal(reconstructed, diff)
