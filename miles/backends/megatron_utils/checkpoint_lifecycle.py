"""Complete native checkpoint work before handing closed files to a caller."""

import inspect


class CheckpointLifecycle:
    """One pending checkpoint, finalized collectively on the training thread."""

    def __init__(self, args):
        self.args = args
        self.pending = None
        self.native = None
        if args.async_save:
            from megatron.training import async_utils

            self.native = async_utils
            if args.use_persistent_ckpt_worker:
                initialize = async_utils.init_persistent_async_worker
                kwargs = {"rank": args.rank} if "rank" in inspect.signature(initialize).parameters else {}
                initialize(**kwargs)

    def prepare(self):
        self.poll(blocking=True)

    def saved(self, iteration, checkpoint_dir, hf_dir):
        """Register files only after the calling thread has finished HF export."""
        if self.pending is not None:
            raise RuntimeError("A checkpoint is still pending completion")
        self.pending = (iteration, checkpoint_dir, hf_dir)

    def poll(self, *, blocking=False, terminate=False):
        if self.native is not None:
            self.native.maybe_finalize_async_save(blocking=blocking, terminate=terminate)
            if not self.native.is_empty_async_queue():
                return
        if self.pending is None:
            return
        path = getattr(self.args, "custom_checkpoint_completed_hook_path", None)
        if path:
            import torch.distributed as dist

            from miles.utils.distributed_utils import get_gloo_group
            from miles.utils.misc import load_function

            error = None
            try:
                load_function(path)(self.args, *self.pending)
            except Exception as exc:
                error = f"rank {dist.get_rank()}: {type(exc).__name__}: {exc}"
            errors = [None] * dist.get_world_size()
            dist.all_gather_object(errors, error, group=get_gloo_group())
            failures = [error for error in errors if error]
            if failures:
                raise RuntimeError("Checkpoint completion failed:\n" + "\n".join(failures))
        self.pending = None
