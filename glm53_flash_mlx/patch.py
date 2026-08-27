"""Correctness fixes for the initial mlx-vlm GLM-5.3 implementation.

The fixes were validated against transformers 5.16 by PipeNetwork's
``glm53-flash-mlx`` project (Apache-2.0, revision b6665e8).  Keeping them in a
small, explicit patch makes the runtime auditable and lets us remove individual
fixes as they land upstream.
"""

from __future__ import annotations

import threading

_LOCK = threading.Lock()
_PATCHED = False


def apply_runtime_patch() -> None:
    """Install the four GLM-5.3 numerical fixes exactly once."""
    global _PATCHED
    with _LOCK:
        if _PATCHED:
            return

        import mlx.core as mx
        import mlx.nn as nn
        from mlx_vlm.models import switch_layers
        from mlx_vlm.models.deepseek_v32 import language as dsv32
        from mlx_vlm.models.glm5_next import language as glm

        class ClampedSwiGLU(nn.Module):
            def __init__(self, limit: float):
                super().__init__()
                self.limit = float(limit)

            def __call__(self, up, gate):
                gate = mx.minimum(gate, self.limit)
                up = mx.clip(up, -self.limit, self.limit)
                return nn.silu(gate) * up

        class ClampedMLP(nn.Module):
            def __init__(self, config, hidden_size=None, intermediate_size=None):
                super().__init__()
                hidden = config.hidden_size if hidden_size is None else hidden_size
                inter = (
                    config.intermediate_size
                    if intermediate_size is None
                    else intermediate_size
                )
                self.gate_proj = nn.Linear(hidden, inter, bias=False)
                self.up_proj = nn.Linear(hidden, inter, bias=False)
                self.down_proj = nn.Linear(inter, hidden, bias=False)
                self.act = ClampedSwiGLU(config.swiglu_limit)

            def __call__(self, x):
                return self.down_proj(self.act(self.up_proj(x), self.gate_proj(x)))

        class Float32Gate(dsv32.MoEGate):
            def __call__(self, x):
                logits = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
                return dsv32.group_expert_select(
                    logits,
                    self.e_score_correction_bias,
                    self.top_k,
                    self.n_group,
                    self.topk_group,
                    self.routed_scaling_factor,
                    self.norm_topk_prob,
                )

        class ClampedMoE(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.num_experts_per_tok = config.num_experts_per_tok
                self.switch_mlp = switch_layers.SwitchGLU(
                    config.hidden_size,
                    config.moe_intermediate_size,
                    config.n_routed_experts,
                    activation=ClampedSwiGLU(config.swiglu_limit),
                )
                self.gate = Float32Gate(config)
                if config.n_shared_experts is not None:
                    self.shared_experts = ClampedMLP(
                        config,
                        intermediate_size=(
                            config.moe_intermediate_size * config.n_shared_experts
                        ),
                    )

            def __call__(self, x):
                inds, scores = self.gate(x)
                y = self.switch_mlp(x, inds)
                y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
                if self.config.n_shared_experts is not None:
                    y = y + self.shared_experts(x)
                return y

        # DecoderLayer resolves these names when it is instantiated.
        glm.DeepseekMLP = ClampedMLP
        glm.DeepseekV32MoE = ClampedMoE

        original_indexer_init = glm.Glm5NextIndexer.__init__

        def indexer_init(self, args):
            original_indexer_init(self, args)
            self.k_norm.eps = 1e-6

        glm.Glm5NextIndexer.__init__ = indexer_init

        original_sparse_init = glm.Glm5NextSparseAttention.__init__

        def sparse_init(self, config):
            original_sparse_init(self, config)
            self.q_a_layernorm.eps = config.rms_norm_eps
            self.kv_a_layernorm.eps = config.rms_norm_eps

        glm.Glm5NextSparseAttention.__init__ = sparse_init

        fp32_suffixes = (
            "_hc.base",
            "_hc.scale",
            "forget_gate.A_log",
            "forget_gate.dt_bias",
            "e_score_correction_bias",
        )
        original_sanitize = glm.LanguageModel.sanitize

        def sanitize(self, weights):
            out = original_sanitize(self, weights)
            for key, value in list(out.items()):
                if key.endswith(fp32_suffixes) and value.dtype != mx.float32:
                    out[key] = value.astype(mx.float32)
            return out

        glm.LanguageModel.sanitize = sanitize
        glm.LanguageModel.FP32_SUFFIXES = fp32_suffixes

        def cast_predicate(self):
            return lambda key: not key.endswith(fp32_suffixes)

        glm.LanguageModel.cast_predicate = property(cast_predicate)
        _PATCHED = True


def patch_status() -> dict:
    return {
        "applied": _PATCHED,
        "source": "PipeNetwork/glm53-flash-mlx@b6665e8",
        "fixes": [
            "clamped_swiglu",
            "float32_router",
            "mla_rms_epsilon",
            "indexer_layernorm_epsilon",
            "mhc_kda_float32_storage",
        ],
    }
