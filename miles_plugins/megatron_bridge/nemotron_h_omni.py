"""Miles integration for Nemotron-H Omni checkpoints (``NemotronH_Omni_Reasoning_V3``).

Nemotron 3.5 Super VL ships the Nemotron 3 Super 120B-A12B language model with a
C-RADIO vision encoder and projector. Its HF config nests the language model under
``llm_config`` and stores every language tensor under the ``language_model.`` prefix.

This plugin post-trains the language model only:

1. Admits the Omni architecture name to ``AutoBridge`` (which matches architectures
   by name suffix) and registers :class:`MilesNemotronHOmniBridge` for it, targeting
   the same Megatron ``MambaModel`` as Nemotron-H.
2. Builds the provider from the nested language config through
   :class:`~miles_plugins.megatron_bridge.nemotron_h.MilesNemotronHBridge`, so MoE,
   latent-projection, and MTP handling stay in one place.
3. Prefixes every HF parameter mapping with ``language_model.``. Vision encoder and
   projector tensors have no Megatron counterpart; the rollout engine serves them from
   the base checkpoint and LoRA adapters apply to the language model alone.

Importing this module is idempotent.
"""

from __future__ import annotations

import copy
import logging

logger = logging.getLogger(__name__)

OMNI_ARCHITECTURE = "NemotronH_Omni_Reasoning_V3"
LANGUAGE_MODEL_PREFIX = "language_model."

_LAYER_SYMBOLS = {"mamba": "M", "attention": "*", "moe": "E", "mlp": "-"}


def layer_pattern(block_types) -> str | None:
    """Encode an HF ``layers_block_type`` list as a Megatron hybrid layer pattern."""
    if not block_types:
        return None
    return "".join(_LAYER_SYMBOLS[block_type] for block_type in block_types)


class LanguageModelConfig:
    """The nested Nemotron-H config of an Omni checkpoint plus the pattern fields Bridge reads.

    transformers 5 describes the hybrid layout with ``layers_block_type`` and drops
    ``hybrid_override_pattern``, which the Nemotron-H bridge still maps.
    """

    def __init__(self, llm_config):
        self._llm_config = llm_config

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._llm_config, name)

    @property
    def hybrid_override_pattern(self):
        pattern = getattr(self._llm_config, "hybrid_override_pattern", None)
        return pattern or layer_pattern(self._llm_config.layers_block_type)

    @property
    def mtp_hybrid_override_pattern(self):
        pattern = getattr(self._llm_config, "mtp_hybrid_override_pattern", None)
        return pattern or layer_pattern(getattr(self._llm_config, "mtp_layers_block_type", None))


def language_model_config(hf_config):
    """Return the language-model view of an Omni config; any other config passes through."""
    llm_config = getattr(hf_config, "llm_config", None)
    if llm_config is None:
        return hf_config
    return LanguageModelConfig(llm_config)


class LanguageModelPretrained:
    """``hf_pretrained`` whose ``config`` is the language-model config; weights and artifacts are shared."""

    def __init__(self, hf_pretrained):
        self._hf_pretrained = hf_pretrained

    @property
    def config(self):
        return language_model_config(self._hf_pretrained.config)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._hf_pretrained, name)


def _prefixed(mapping, prefix: str):
    copied = copy.copy(mapping)
    if isinstance(mapping.hf_param, str):
        copied.hf_param = prefix + mapping.hf_param
    else:
        copied.hf_param = {key: prefix + value for key, value in mapping.hf_param.items()}
    return copied


def _allow_omni_architecture() -> None:
    from megatron.bridge.models.conversion import auto_bridge

    if OMNI_ARCHITECTURE in auto_bridge.SUPPORTED_HF_ARCHITECTURES:
        return
    auto_bridge.SUPPORTED_HF_ARCHITECTURES = (*auto_bridge.SUPPORTED_HF_ARCHITECTURES, OMNI_ARCHITECTURE)
    auto_bridge.SUPPORTED_HF_ARCHITECTURES_DISPLAY = " or ".join(f"'{name}'" for name in auto_bridge.SUPPORTED_HF_ARCHITECTURES)


def _build_bridge_subclass():
    from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
    from megatron.core.models.mamba import MambaModel

    from miles_plugins.megatron_bridge.nemotron_h import bridge_class

    _allow_omni_architecture()

    @MegatronModelBridge.register_bridge(source=OMNI_ARCHITECTURE, target=MambaModel)
    class MilesNemotronHOmniBridge(bridge_class()):
        """Language-model-only bridge for Nemotron-H Omni checkpoints."""

        def provider_bridge(self, hf_pretrained):
            return super().provider_bridge(LanguageModelPretrained(hf_pretrained))

        def mapping_registry(self):
            # Bridge assigns the checkpoint's top-level config before building mappings.
            self.hf_config = language_model_config(getattr(self, "hf_config", None))
            registry = super().mapping_registry()
            return MegatronMappingRegistry(*(_prefixed(mapping, LANGUAGE_MODEL_PREFIX) for mapping in registry.mappings))

    return MilesNemotronHOmniBridge


_bridge_class = None


def bridge_class():
    """Return the registered :class:`MilesNemotronHOmniBridge`, building it on first use."""
    global _bridge_class
    if _bridge_class is None:
        _bridge_class = _build_bridge_subclass()
    return _bridge_class


def install() -> None:
    """Register the Nemotron-H Omni bridge subclass."""
    try:
        bridge_class()
    except Exception as e:  # best-effort; avoid breaking unrelated models
        logger.warning("miles nemotron_h_omni bridge not registered: %s", e)


install()
