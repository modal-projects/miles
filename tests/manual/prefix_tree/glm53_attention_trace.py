"""Capture matching native/tree convolution and recurrence boundaries.

Per-level temporary rows are restored by the actual tree implementation before
comparison. These hooks observe values only; they do not substitute kernels.
"""

import importlib


def install_attention_trace(model, trace, *, attach):
    # Optional Bridge/FLA dependencies are needed only in the running GPU model.
    from miles.backends.training_utils.prefix_tree.glm53 import bridge, kda

    native = importlib.import_module("megatron.bridge.models.glm5_next.kda")
    conv = importlib.import_module("fla.modules.conv.causal_conv1d")
    current = {"name": None}
    handles, originals = [], []

    def capture(label, output):
        if current["name"] is not None:
            trace.capture(
                current["name"] + label,
                output.transpose(0, 1),
                fingerprint=int(current["name"].split(".")[2]) not in trace.layers,
            )

    def wrap(module, attribute, label, *, tuple_output=False, enabled=True):
        original = getattr(module, attribute)

        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            if enabled:
                capture(label, result[0] if tuple_output else result)
            return result

        originals.append((module, attribute, original))
        setattr(module, attribute, observed)

    wrap(conv, "causal_conv1d", ".convolution.output", tuple_output=True, enabled=not attach)
    wrap(kda, "_tree_convolution", ".convolution.output", enabled=attach)
    wrap(native, "chunk_kda", ".recurrence.output", tuple_output=True, enabled=not attach)
    wrap(bridge, "tree_kda", ".recurrence.output", enabled=attach)
    for name, module in model.named_modules():
        if not hasattr(module, "local_num_heads") or not hasattr(module, "conv1d"):
            continue

        def before(module, args, label=name):
            current["name"] = label

        def after(module, args, result):
            current["name"] = None

        handles.append(module.register_forward_pre_hook(before))
        handles.append(module.register_forward_hook(after))
    return handles, originals
