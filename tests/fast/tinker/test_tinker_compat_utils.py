from miles.tinker.tinker_compat_utils import MASK_LOGPROB, set_topk_shape
from tinker.proto import tinker_public_pb2 as public_pb


def test_mask_logprob_is_the_tinker_top_k_padding_value():
    assert MASK_LOGPROB == -99999.0


def test_set_topk_shape_records_positions_and_k_on_the_installed_proto():
    message = public_pb.SampleResponse().topk_prompt_logprobs
    set_topk_shape(message, 7, 3)
    fields = message.DESCRIPTOR.fields_by_name
    assert getattr(message, "length" if "length" in fields else "prompt_length") == 7
    assert message.k == 3
