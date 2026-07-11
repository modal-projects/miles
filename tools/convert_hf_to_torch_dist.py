import gc
import os
import shutil

import torch
import torch.distributed as dist
from megatron.core.enums import ModelType
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, save_checkpoint
from megatron.training.training import get_model

import miles_plugins.mbridge  # noqa: F401
from mbridge import AutoBridge
from miles.backends.megatron_utils.arguments import set_default_megatron_args
from miles.backends.megatron_utils.initialize import init
from miles.backends.megatron_utils.model_provider import get_model_provider_func
from miles.utils.logging_utils import configure_logger_raw
from miles.utils.memory_utils import print_memory


def add_conversion_args(parser):
    """Add conversion arguments to the parser"""
    parser.add_argument("--hf-checkpoint", type=str, required=True, help="HuggingFace model path")
    parser.add_argument(
        "--megatron-to-hf-mode",
        choices=["raw", "bridge"],
        default="raw",
        help="The method to convert megatron weights to hugging face weights for SGLang.",
    )
    try:
        parser.add_argument("--padded-vocab-size", type=int, default=None)
    except Exception:
        pass
    return parser


def get_args():
    args = parse_args(add_conversion_args)
    args = set_default_megatron_args(args)

    args.debug_deterministic_collective = False
    args.enable_witness = False

    # set to pass megatron validate_args
    args.save_interval = 1
    args.micro_batch_size = 1
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.global_batch_size = int(os.environ.get("WORLD_SIZE", "1"))

    assert args.pipeline_model_parallel_size <= args.num_layers, (
        f"Pipeline model parallel size {args.pipeline_model_parallel_size} must be less than or equal to "
        f"number of layers {args.num_layers}."
    )

    def ceildiv(a, b):
        return -(a // -b)

    # Auto-derive pipeline parallelism ONLY when experts aren't being EP-sharded.
    # For a large MoE (e.g. 1T K2.6) pure PP=world_size with EP=1 leaves all experts
    # on every rank -> OOM; the caller passes --expert-model-parallel-size instead,
    # and we must keep the explicit pp (here 1) rather than override to world_size.
    if args.pipeline_model_parallel_size == 1 and world_size > 1 and args.expert_model_parallel_size <= 1:
        pp_size = world_size
        while True:
            args.pipeline_model_parallel_size = pp_size
            args.decoder_last_pipeline_num_layers = args.num_layers - ceildiv(
                args.num_layers, args.pipeline_model_parallel_size
            ) * (args.pipeline_model_parallel_size - 1)

            if args.decoder_last_pipeline_num_layers > 0:
                break

            if pp_size % 2 == 0:
                pp_size //= 2
            else:
                raise ValueError(
                    f"Cannot find a valid pipeline model parallel size for {args.num_layers} layers and {world_size} GPUs."
                )
    print(
        f"Using pipeline model parallel size: {args.pipeline_model_parallel_size}, decoder last pipeline num layers: {args.decoder_last_pipeline_num_layers}"
    )

    validate_args(args)
    return args


def main():
    if torch.version.hip:
        import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module
        import megatron.core.dist_checkpointing.strategies.torch as torch_strategy_module

        from miles.utils.rocm_checkpoint_writer import ROCmFileSystemWriterAsync

        filesystem_async_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync
        torch_strategy_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync
        print("[ROCm] Applied FileSystemWriterAsync patch for HIP compatibility")

    configure_logger_raw()

    # Initialize distributed environment
    world_size = int(os.getenv("WORLD_SIZE") or os.getenv("SLURM_NTASKS") or 1)
    local_rank = int(os.getenv("LOCAL_RANK") or os.getenv("SLURM_LOCALID") or 0)
    global_rank = int(os.getenv("RANK") or os.getenv("SLURM_PROCID") or 0)

    torch.cuda.set_device(local_rank)
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("RANK", str(global_rank))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=global_rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    args = get_args()
    init(args)
    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)

    # Load model
    hf_model_path = args.hf_checkpoint
    bridge = AutoBridge.from_pretrained(hf_model_path, trust_remote_code=True)

    bridge.load_weights(model, hf_model_path, memory_efficient=True)
    print(f"Model loaded: {hf_model_path}")

    print_memory("after loading model")
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    save_checkpoint(1, model, None, None, 0)

    dist.barrier()

    # NOTE: the upstream version here does shutil.move(iter_0000001 -> release/) on rank 0
    # and writes tracker "release". On Modal each node mounts the save Volume separately, so
    # the multi-node distcp save scatters its .distcp shards across per-node mounts — rank 0's
    # move would relocate only node 0's shards and orphan the rest. Skip the move and load
    # straight from iter_0000001 (tracker = the iteration). The launcher commits the Volume on
    # *every* node so all shards land under the one iter dir.
    #
    # This must be the *last* step and after a dist.barrier — higher-level scripts treat the
    # tracker write as the success signal.
    if dist.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, "w") as f:
            f.write("1")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
