import pytest
import torch

from miles.utils.score_centering import RolloutScoreCenteringHead


def test_score_centering_head_roundtrips_ragged_rows_and_selection():
    first = RolloutScoreCenteringHead.from_rows([[4, 2], [1]], [[-0.4, -1.2], [0.0]])
    second = RolloutScoreCenteringHead.from_rows([[8, 3, 6]], [[-0.8, -1.1, -1.7]])
    combined = RolloutScoreCenteringHead.concatenate((first, second))

    ids, logprobs, lengths = combined._select_rows(torch.tensor([2, 0]))

    assert ids.tolist() == [8, 3, 6, 4, 2]
    torch.testing.assert_close(logprobs, torch.tensor([-0.8, -1.1, -1.7, -0.4, -1.2]))
    assert lengths.tolist() == [3, 2]
    assert len(combined.prefix(2)) == 2


@pytest.mark.parametrize(
    "ids,offsets,logprobs",
    [
        ([1, 1], [0, 2], [-0.7, -0.7]),
        ([1], [0, 1], [0.1]),
        ([1, 2], [0, 2], [0.0, 0.0]),
        ([], [0, 0], []),
    ],
)
def test_score_centering_head_rejects_invalid_subdistributions(ids, offsets, logprobs):
    with pytest.raises(ValueError):
        RolloutScoreCenteringHead(ids=ids, offsets=offsets, logprobs=logprobs)


def test_empty_tuple_selection_is_valid_for_an_empty_local_cp_chunk():
    head = RolloutScoreCenteringHead.from_rows([[4, 2]], [[-0.4, -1.2]])

    ids, logprobs, lengths = head._select_rows(())

    assert ids.numel() == logprobs.numel() == lengths.numel() == 0
