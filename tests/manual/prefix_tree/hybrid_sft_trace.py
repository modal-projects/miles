"""Canonical occurrence captures and first-block kernel dispatch for hybrid SFT."""

import json
import re

import torch


class HybridTrace:
    def __init__(self, model, output):
        self.output = output
        self.references = {}
        self.handles = []
        for name, module in model.named_modules():
            layer = re.search(r"\.layers\.(\d+)(?:\.|$)", name)
            if layer is not None:
                index = int(layer[1])
                root = name.endswith(".layers." + str(index))
                if root or index == 0:
                    self.handles.append(module.register_forward_hook(self._hook(name)))
            elif name.endswith(".norm"):
                self.handles.append(module.register_forward_hook(self._hook(name)))
        first = model.get_base_model().model.layers[0]
        self.handles.append(first.register_forward_pre_hook(self._profile_begin))
        self.handles.append(first.register_forward_hook(self._profile_end))

    def _hook(self, name):
        def hook(module, args, output):
            if isinstance(output, tuple):
                output = output[0]
            self.capture(name, output)

        return hook

    def begin(self, name, batch, *, replace):
        self.name = name
        self.rows = torch.tensor([row for path in batch.attention.plan.sample_nodes for row in path])
        self.count = len(batch.attention.plan.tokens)
        self.replace = replace
        self.comparisons = {}

    def capture(self, name, value):
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim < 3
            or value.shape[0] != 1
            or value.shape[1] != self.count
        ):
            return
        flat = value.detach().reshape(self.count, -1).contiguous()
        full = ".layers.0." in name or name.endswith(".model.norm")
        if not full:
            words = flat.view(torch.int16 if flat.dtype in (torch.bfloat16, torch.float16) else torch.int32).long()
            index = torch.arange(words.shape[1], device=words.device, dtype=torch.int64)
            flat = torch.stack(
                (
                    (words * ((index + 1) * 6364136223846793005)).sum(-1),
                    (words * (words + 3) * ((index + 11) * 2862933555777941757)).sum(-1),
                ),
                dim=-1,
            )
        canonical = flat.cpu().index_select(0, self.rows)
        reference = self.references.get(name)
        if reference is not None and not self.replace:
            equal = torch.equal(canonical, reference)
            row = dict(bitwise_equal=equal, comparison="full_tensor" if full else "two_int64_fingerprints")
            if not equal:
                changed = (canonical != reference).reshape(len(canonical), -1).any(-1)
                row.update(changed_rows=int(changed.sum()), first_original_row=int(changed.nonzero()[0]))
                if full:
                    delta = canonical.float() - reference.float()
                    row.update(
                        max_abs=float(delta.abs().max()),
                        relative_l2=float(
                            (
                                delta.square().sum(dtype=torch.float64)
                                / reference.float().square().sum(dtype=torch.float64).clamp_min(1e-30)
                            ).sqrt()
                        ),
                    )
            self.comparisons[name] = row
        if self.replace:
            self.references[name] = canonical

    def _profile_begin(self, module, args):
        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True
        )
        self.profiler.__enter__()

    def _profile_end(self, module, args, output):
        self.profiler.__exit__(None, None, None)
        events = self.profiler.events()
        kernels = [
            dict(
                operator=event.name, input_shapes=event.input_shapes, kernels=[kernel.name for kernel in event.kernels]
            )
            for event in events
            if event.kernels
        ]
        kernels.append(
            dict(
                operator="CUDA kernels (all threads)",
                input_shapes=[],
                kernels=sorted(
                    {event.name for event in events if event.device_type == torch.autograd.DeviceType.CUDA}
                ),
            )
        )
        (self.output.parent / ("kernels-" + self.name + ".json")).write_text(json.dumps(kernels))
        del self.profiler

    def report(self):
        changed = [(name, row) for name, row in self.comparisons.items() if not row["bitwise_equal"]]
        return dict(first_difference=changed[0] if changed else None, comparisons=self.comparisons)

    def close(self):
        for handle in self.handles:
            handle.remove()
