from miles.tinker.tinker_compat_utils import MASK_LOGPROB


def test_mask_logprob_is_the_tinker_top_k_padding_value():
    assert MASK_LOGPROB == -99999.0
