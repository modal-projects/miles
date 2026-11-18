import torch
import torch.nn as nn
from megatron.core.transformer.module import mark_keep_in_fp32
from torch.utils.checkpoint import checkpoint

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda
    from fla.ops.kda.gate import fused_kda_gate
except ImportError:
    FusedRMSNormGated = None
    ShortConvolution = None
    chunk_kda = None
    fused_kda_gate = None

from miles_plugins.models.cp_utils import build_gdn_cp_context
from miles_plugins.models.hf_attention import HuggingfaceAttention
from miles_plugins.models.linear_attn import INT32_ELEMENTS, causal_short_conv


# Backward keeps several FP32 buffers; cap each input/state buffer at 1 GiB.
_KDA_BUFFER_ELEMENTS = 2**28


def _get_text_config(hf_config):
    return (
        getattr(hf_config, "text_config", None) or hf_config
    )  # config-access-exempt: HF configs may wrap text_config for multimodal checkpoints


def _bounded_chunk_kda(query, key, value, g, beta, cu_seqlens):
    chunk_size = 64
    batch, tokens, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    max_chunks = (tokens + chunk_size - 1) // chunk_size + cu_seqlens.numel() - 2
    elements_per_head = batch * max(tokens * max(key_dim, value_dim), max_chunks * key_dim * value_dim)
    head_limit = max(1, min(INT32_ELEMENTS, _KDA_BUFFER_ELEMENTS) // elements_per_head)
    head_block = 1 << (head_limit.bit_length() - 1)
    if head_block < heads:
        # FLA uses int32 offsets for packed inputs and recurrent states; heads are independent.
        outputs = []
        for start in range(0, heads, head_block):
            selected = slice(start, start + head_block)
            inputs = (
                query[:, :, selected],
                key[:, :, selected],
                value[:, :, selected],
                g[:, :, selected],
                beta[:, :, selected],
                cu_seqlens,
            )
            # Keep views here: checkpointing the copies would retain every group's full input.
            outputs.append(
                checkpoint(_bounded_chunk_kda, *inputs, use_reentrant=False)
                if torch.is_grad_enabled()
                else _bounded_chunk_kda(*inputs)
            )
        return torch.cat(outputs, dim=2)
    # Copy inside the checkpoint so only the group being recomputed needs contiguous inputs.
    return chunk_kda(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )[0]


class Glm5NextKDA(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int,
        gate_lower_bound: float,
        rms_norm_eps: float,
    ):
        super().__init__()
        if ShortConvolution is None or chunk_kda is None:
            raise ImportError("GLM-5.3 KDA requires flash-linear-attention >= 0.4.2 (fla.ops.kda).")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.projection_size = num_heads * head_dim
        self.conv_kernel_size = conv_kernel_size
        self.gate_lower_bound = gate_lower_bound

        self.q_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.conv1d = ShortConvolution(
            hidden_size=3 * self.projection_size,
            kernel_size=conv_kernel_size,
            bias=False,
            activation="silu",
        )
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.f_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.f_b_proj = nn.Linear(head_dim, self.projection_size, bias=False)
        self.g_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.g_b_proj = nn.Linear(head_dim, self.projection_size, bias=False)
        self.A_log = mark_keep_in_fp32(nn.Parameter(torch.zeros(num_heads, dtype=torch.float32)))
        self.dt_bias = mark_keep_in_fp32(nn.Parameter(torch.zeros(self.projection_size, dtype=torch.float32)))
        self.o_norm = FusedRMSNormGated(head_dim, eps=rms_norm_eps, activation="sigmoid")
        self.o_proj = nn.Linear(self.projection_size, hidden_size, bias=False)

    def _project_and_convolve(self, hidden_states, cu_seqlens, cp_context):
        mixed_qkv = torch.cat(
            (self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)),
            dim=-1,
        )
        return self._convolve(mixed_qkv, self.conv1d.weight[:, 0], cu_seqlens, cp_context)

    def _project_one_and_convolve(self, hidden_states, projection, weight, cu_seqlens):
        return self._convolve(projection(hidden_states), weight, cu_seqlens)

    def _convolve(self, inputs, weight, cu_seqlens, cp_context=None):
        return causal_short_conv(
            inputs,
            weight,
            self.conv1d.activation,
            self.conv1d.backend,
            cu_seqlens=cu_seqlens,
            cp_context=cp_context,
        )

    def _forget_gate(self, hidden_states):
        forget = self.f_b_proj(self.f_a_proj(hidden_states))
        return fused_kda_gate(
            forget.unflatten(-1, (self.num_heads, self.head_dim)),
            self.A_log,
            self.dt_bias,
            lower_bound=self.gate_lower_bound,
        )

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor):
        cp_context = build_gdn_cp_context(self, cu_seqlens, hidden_states.device)
        conv_cu_seqlens = cp_context.cu_seqlens if cp_context is not None else cu_seqlens
        conv_elements = hidden_states.shape[0] * hidden_states.shape[1] * 3 * self.projection_size
        if torch.is_grad_enabled() and cp_context is None and conv_elements > INT32_ELEMENTS:
            # Independent checkpoints avoid full combined-QKV input and gradient buffers.
            qkv = []
            weights = self.conv1d.weight[:, 0].split(self.projection_size)
            for projection, weight in zip((self.q_proj, self.k_proj, self.v_proj), weights, strict=True):
                qkv.append(
                    checkpoint(
                        self._project_one_and_convolve,
                        hidden_states,
                        projection,
                        weight,
                        conv_cu_seqlens,
                        use_reentrant=False,
                    )
                )
            query, key, value = qkv
            g = checkpoint(self._forget_gate, hidden_states, use_reentrant=False)
        else:
            mixed_qkv = self._project_and_convolve(hidden_states, conv_cu_seqlens, cp_context)
            query, key, value = torch.split(mixed_qkv, [self.projection_size] * 3, dim=-1)
            g = self._forget_gate(hidden_states)
        query = query.unflatten(-1, (self.num_heads, self.head_dim))
        key = key.unflatten(-1, (self.num_heads, self.head_dim))
        value = value.unflatten(-1, (self.num_heads, self.head_dim))
        beta = torch.sigmoid(self.b_proj(hidden_states).float())

        if cp_context is not None:
            core_attn_out, _ = chunk_kda(
                query,
                key,
                value,
                g=g,
                beta=beta,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cp_context.cu_seqlens,
                cp_context=cp_context,
            )
        else:
            core_attn_out = _bounded_chunk_kda(query, key, value, g, beta, cu_seqlens)

        norm_gate = self.g_b_proj(self.g_a_proj(hidden_states))
        out_shape = core_attn_out.shape
        core_attn_out = self.o_norm(
            core_attn_out.reshape(-1, self.head_dim),
            norm_gate.reshape(-1, self.head_dim),
        )
        core_attn_out = core_attn_out.reshape(out_shape[0], out_shape[1], -1)
        return self.o_proj(core_attn_out)


class Glm5NextKDAAttention(HuggingfaceAttention):

    hybrid_cp = True

    def __init__(
        self,
        args,
        config,
        layer_number: int,
        cp_comm_type: str = "p2p",
        pg_collection=None,
        name: str | None = None,
    ):
        super().__init__(args, config, layer_number, cp_comm_type, pg_collection, name=name)
        text_config = _get_text_config(self.hf_config)
        # the native config parses linear_attn_config into the linear_* fields
        if text_config.linear_lower_bound is None:
            raise ValueError("GLM-5.3 KDA requires gate_lower_bound (safe gate) in the HF config.")
        self.kda = Glm5NextKDA(
            hidden_size=text_config.hidden_size,
            num_heads=text_config.linear_num_heads,
            head_dim=text_config.linear_head_dim,
            conv_kernel_size=text_config.linear_conv_kernel_dim,
            gate_lower_bound=text_config.linear_lower_bound,
            rms_norm_eps=text_config.rms_norm_eps,
        )

    def hf_forward(self, hidden_states, packed_seq_params):
        return self.kda(hidden_states, cu_seqlens=packed_seq_params.cu_seqlens_q)
