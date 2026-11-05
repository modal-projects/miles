from model_args_utils import load_sibling_model_args


def model_args() -> str:
    return load_sibling_model_args(__file__, "deepseek-v4.1", nlayers=7, compress_ratios="[0,0,2,2,1,1,1]")
