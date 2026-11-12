"""Test-only native expert/key replay with exact prefix-eligibility checks.

No production replay manager or upstream source is changed. Indices are fixed;
expert probabilities remain differentiable through the native RouterReplay API.
"""

from dataclasses import replace

import torch

from miles.backends.training_utils.prefix_tree.plan import build_prefix_tree


def plans(data):
    kwargs = dict(sharing_keys=data.slots, computation_keys=data.computation_keys)
    return (
        build_prefix_tree(data.sequences, share_prefixes=False, **kwargs),
        build_prefix_tree(data.sequences, share_prefixes=True, **kwargs),
    )


def occurrence_maps(expanded, compact, *, device):
    first = [-1] * len(compact.tokens)
    mapping = [-1] * len(expanded.tokens)
    for old_path, new_path in zip(expanded.sample_nodes, compact.sample_nodes, strict=True):
        for old, new in zip(old_path, new_path, strict=True):
            mapping[old] = new
            if first[new] < 0:
                first[new] = old
    return torch.tensor(first, device=device), torch.tensor(mapping, device=device)


def compatible_data(data, choices, *, sp_multiple=2):
    expanded, compact = plans(data)
    device = next(iter(choices.values())).device
    first, mapping = occurrence_maps(expanded, compact, device=device)
    bad = torch.zeros(len(expanded.tokens), device=device, dtype=torch.bool)
    for value in choices.values():
        restored = value.index_select(0, first).index_select(0, mapping)
        bad |= (restored != value).reshape(len(expanded.tokens), -1).any(-1)
    conflicting_rows = int(bad.sum())
    keys, forked = [], 0
    for original, path in zip(data.computation_keys, expanded.sample_nodes, strict=True):
        conflicts = bad[list(path)].nonzero()
        replacement = list(original)
        if len(conflicts):
            # Fork no later than the first incompatible token. Keeping the cut
            # on the SP grid preserves expanded/compact row-count alignment.
            cut = int(conflicts[0]) // sp_multiple * sp_multiple
            sample = len(keys)
            replacement[cut:] = [(key, ("replay-fork", sample)) for key in replacement[cut:]]
            forked += 1
        keys.append(tuple(replacement))
    result = replace(data, computation_keys=tuple(keys))
    new_expanded, new_compact = plans(result)
    first, mapping = occurrence_maps(new_expanded, new_compact, device=device)
    for value in choices.values():
        if not torch.equal(value.index_select(0, first).index_select(0, mapping), value):
            raise ValueError("Replay choices remain incompatible after conservative prefix forks")
    if len(new_compact.tokens) % sp_multiple:
        raise ValueError("Replay-aware compact rows violate SP alignment")
    return result, {
        "incompatible_original_rows": conflicting_rows,
        "forked_samples": forked,
        "compact_rows_before": len(compact.tokens),
        "compact_rows_after": len(new_compact.tokens),
    }


class ChoiceReplay:
    def __init__(self, model, expanded):
        # Optional GPU runtime dependencies; planning tests import only Torch.
        from glm53_diagnose import _record_routes
        from megatron.core import parallel_state

        self.parallel_state = parallel_state
        self.routers = _record_routes(model)
        self.expert_values = {}
        self.key_values = {}
        self.key_logical = {}
        self.native_selectors = []
        self.mode = "record"
        self.key_targets = {}
        self.positions = torch.tensor(expanded.positions, device="cuda", dtype=torch.long)
        owners = [-1] * len(expanded.tokens)
        for sample, path in enumerate(expanded.sample_nodes):
            for row in path:
                owners[row] = sample
        self.owners = torch.tensor(owners, device="cuda")
        for name, module in model.named_modules():
            if not hasattr(module, "_kpool_select"):
                continue
            original = module._kpool_select

            def select(*args, fn=original, label=name, **kwargs):
                if self.mode == "all":
                    return self.key_targets[label]
                value = fn(*args, **kwargs)
                if self.mode == "record":
                    valid = value >= 0
                    indices = value.clamp_min(0).long()
                    logical = self.positions[indices]
                    invalid = valid & (
                        (self.owners[indices] != self.owners[:, None, None])
                        | (logical > self.positions[:, None, None])
                    )
                    if bool(invalid.any()):
                        raise ValueError("Native DSA selection violates packed causal ancestry")
                    self.key_values[label] = value.detach().clone()
                    self.key_logical[label] = torch.where(valid, logical, -1).detach()
                return value

            self.native_selectors.append((module, original))
            module._kpool_select = select

    def finish_record(self):
        import torch.distributed as dist

        group = self.parallel_state.get_tensor_model_parallel_group()
        tp = dist.get_world_size(group)
        self.sharded = {}
        for name, replay in self.routers.items():
            value = replay.get_recorded_indices().detach().clone()
            self.sharded[name] = len(value) != len(self.positions)
            if self.sharded[name]:
                if len(value) * tp != len(self.positions):
                    raise ValueError("Unexpected native router row layout")
                pieces = [torch.empty_like(value) for _ in range(tp)]
                dist.all_gather(pieces, value, group=group)
                value = torch.cat(pieces)
            self.expert_values[name] = value
        return self.expert_values | {name + ".keys": value for name, value in self.key_logical.items()}

    def prepare_compact(self, expanded, compact):
        self.first, _ = occurrence_maps(expanded, compact, device="cuda")
        width = max(map(len, compact.sample_nodes))
        table = torch.full((len(compact.sample_nodes), width), -1, device="cuda", dtype=torch.long)
        for sample, path in enumerate(compact.sample_nodes):
            table[sample, : len(path)] = torch.tensor(path, device="cuda")
        owner = self.owners.index_select(0, self.first)[:, None, None]
        self.compact_keys = {}
        for name, value in self.key_logical.items():
            logical = value.index_select(0, self.first)
            addresses = owner * width + logical.clamp_min(0)
            mapped = table.flatten()[addresses]
            if bool(((logical >= 0) & (mapped < 0)).any()):
                raise ValueError("A replayed key is not on the compact query path")
            self.compact_keys[name] = torch.where(logical >= 0, mapped, -1).to(torch.int32).contiguous()

    def enable(self, mode, *, shared):
        from megatron.core.transformer.moe.router_replay import RouterReplayAction

        self.mode = mode
        tp = self.parallel_state.get_tensor_model_parallel_world_size()
        rank = self.parallel_state.get_tensor_model_parallel_rank()
        for name, replay in self.routers.items():
            replay.clear_indices()
            if mode == "none":
                replay.clear_router_replay_action()
                continue
            value = self.expert_values[name]
            if shared:
                value = value.index_select(0, self.first)
            if self.sharded[name]:
                value = value.chunk(tp)[rank]
            replay.set_target_indices(value)
            replay.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
        self.key_targets = self.compact_keys if shared else self.key_values
