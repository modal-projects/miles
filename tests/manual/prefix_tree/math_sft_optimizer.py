"""FP32 LoRA master optimizer and accumulation for the standalone SFT probe."""

import copy

import torch
import torch.distributed as dist


class MasterOptimizer:
    def __init__(self, model, lr):
        self.parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        self.masters = [torch.nn.Parameter(p.detach().float().clone()) for _, p in self.parameters]
        self.buffers = [torch.zeros_like(p) for p in self.masters]
        for master, gradient in zip(self.masters, self.buffers, strict=True):
            master.grad = gradient
        self.optimizer = torch.optim.AdamW(
            self.masters, lr=lr, weight_decay=0.0, fused=self.masters[0].device.type == "cuda"
        )

    @torch.no_grad()
    def clear(self, model):
        model.zero_grad(set_to_none=True)
        for gradient in self.buffers:
            gradient.zero_()

    @torch.no_grad()
    def accumulate(self, model):
        for (_, p), gradient in zip(self.parameters, self.buffers, strict=True):
            if p.grad is not None:
                gradient.add_(p.grad.float())
        model.zero_grad(set_to_none=True)

    def synchronize(self):
        # Megatron is optional for CPU optimizer-state and accumulation tests.
        from megatron.core import parallel_state

        tp = parallel_state.get_tensor_model_parallel_group()
        dp = parallel_state.get_data_parallel_group()
        edp = parallel_state.get_expert_data_parallel_group()
        for (_, parameter), gradient in zip(self.parameters, self.buffers, strict=True):
            if getattr(parameter, "sequence_parallel", False):
                dist.all_reduce(gradient, group=tp)
            group = dp if getattr(parameter, "allreduce", True) else edp
            if dist.get_world_size(group) > 1:
                dist.all_reduce(gradient, group=group)
            gradient.div_(parallel_state.get_data_parallel_world_size())

    @torch.no_grad()
    def step(self):
        self.optimizer.step()
        for (_, parameter), master in zip(self.parameters, self.masters, strict=True):
            parameter.copy_(master)

    def snapshot(self):
        return {
            name: gradient.detach().cpu().clone()
            for (name, _), gradient in zip(self.parameters, self.buffers, strict=True)
        }

    @torch.no_grad()
    def reference_step(self, reference_gradients):
        shadows = [torch.nn.Parameter(p.detach().clone()) for p in self.masters]
        reference = torch.optim.AdamW(
            shadows,
            lr=self.optimizer.param_groups[0]["lr"],
            weight_decay=0.0,
            fused=self.masters[0].device.type == "cuda",
        )
        reference.load_state_dict(copy.deepcopy(self.optimizer.state_dict()))
        for (name, _), shadow in zip(self.parameters, shadows, strict=True):
            shadow.grad = reference_gradients[name].to(shadow.device)
        reference.step()
        return shadows
