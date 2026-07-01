from abc import ABC, abstractmethod


class HfWeightIteratorBase(ABC):
    @staticmethod
    def create(args, model, *, mode: str | None = None, **kwargs):
        mode = mode or args.megatron_to_hf_mode
        if mode not in {"raw", "bridge"}:
            raise ValueError(f"Unknown HF weight iterator mode: {mode!r}")

        from .hf_weight_iterator_bridge import HfWeightIteratorBridge
        from .hf_weight_iterator_direct import HfWeightIteratorDirect

        iterators = {
            "raw": HfWeightIteratorDirect,
            "bridge": HfWeightIteratorBridge,
        }

        return iterators[mode](args, model, **kwargs)

    def __init__(self, args, model, model_name, quantization_config, **kwargs):
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config

    @abstractmethod
    def get_hf_weight_chunks(self, megatron_local_weights, weight_type="base"):
        """
        Mental model of the API:
        megatron_model.to_hf_magically().named_parameters()
        """
        raise NotImplementedError
