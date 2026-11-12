"""Beam 501B CLI dimensions; its registered Bridge provider owns model semantics."""

from model_args_utils import moe_layer_freq


def model_args(**kwargs) -> str:
    return (
        "--num-layers 52 --hidden-size 5120 --ffn-hidden-size 20480 "
        "--num-attention-heads 48 --group-query-attention --num-query-groups 8 --kv-channels 128 "
        "--disable-bias-linear --normalization RMSNorm --norm-epsilon 1e-6 --swiglu "
        "--position-embedding-type rope --rotary-base 10000 --rotary-percent 1.0 "
        "--untie-embeddings-and-output-weights --vocab-size 151936 --padded-vocab-size 151936 "
        "--make-vocab-size-divisible-by 8 --num-experts 256 --moe-ffn-hidden-size 2560 "
        "--moe-router-topk 8 --moe-router-score-function sigmoid --moe-router-dtype fp32 "
        f"--moe-layer-freq {moe_layer_freq(nlayers=52, first_k_dense_replace=3)} "
    )
