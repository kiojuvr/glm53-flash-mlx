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
    """Install the pinned GLM-5.3 correctness fixes exactly once."""
    global _PATCHED
    with _LOCK:
        if _PATCHED:
            return

        import mlx.core as mx
        import mlx.nn as nn
        from mlx_vlm.models import switch_layers
        from mlx_vlm.models.cache import CacheList
        from mlx_vlm.models.deepseek_v32 import language as dsv32
        from mlx_vlm.models.glm5_next import language as glm

        from .indexpool import (
            build_prefill_indexpool_mask,
            indexpool_cache_kv_len,
            prepare_decode_indexpool_gather,
            sanitize_indexpool_indices,
        )
        from .nope_cache import (
            CompactIndexPoolCache,
            make_compact_nope_dsa_cache,
        )

        original_cache_list_trim = CacheList.trim

        def atomic_cache_list_trim(self, tokens):
            # Compact NoPE children expose validation so a rejected rollback
            # cannot leave the latent and IndexPool offsets out of sync.
            validators = [
                validator
                for cache in self.caches
                if callable(validator := getattr(cache, "validate_trim", None))
            ]
            for validator in validators:
                validator(tokens)
            return original_cache_list_trim(self, tokens)

        CacheList.trim = atomic_cache_list_trim

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

        original_indexer_call = glm.Glm5NextIndexer.__call__

        def indexer_call(self, x, qr, mask, cache=None):
            if isinstance(cache, CompactIndexPoolCache):
                indices = cache.update(self, x, qr, mask)
            else:
                indices = original_indexer_call(self, x, qr, mask, cache=cache)
            if indices is None:
                return None
            kv_len = indexpool_cache_kv_len(cache, x.shape[1])
            return sanitize_indexpool_indices(indices, kv_len)

        glm.Glm5NextIndexer.__call__ = indexer_call

        original_sparse_init = glm.Glm5NextSparseAttention.__init__

        def sparse_init(self, config):
            original_sparse_init(self, config)
            self.q_a_layernorm.eps = config.rms_norm_eps
            self.kv_a_layernorm.eps = config.rms_norm_eps

        glm.Glm5NextSparseAttention.__init__ = sparse_init

        def sparse_call(self, x, mask=None, cache=None):
            B, L, _ = x.shape

            if cache is not None and isinstance(cache[1], CompactIndexPoolCache):
                # Preflight before either the latent or IndexPool cache mutates.
                cache[1].validate_update(self.indexer, batch=B, length=L)

            qr = self.q_a_layernorm(self.q_a_proj(x))
            q = self.q_b_proj(qr)
            q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(
                0, 2, 1, 3
            )

            compressed_kv = self.kv_a_proj_with_mqa(x)
            kv_latent = self.kv_a_layernorm(compressed_kv)
            kv_latent = mx.expand_dims(kv_latent, axis=1)

            if cache is not None:
                kv_latent, _ = cache[0].update_and_fetch(kv_latent, kv_latent)
            else:
                cache = [None] * 2

            topk_indices = self.indexer(x, qr, mask, cache=cache[1])
            attn_mask = mask
            if topk_indices is not None:
                Kv = kv_latent.shape[2]
                if L == 1:
                    raw = topk_indices[:, :, 0, :]
                    safe_indices, valid_sel = prepare_decode_indexpool_gather(raw, Kv)
                    idx = safe_indices[..., None]
                    kv_latent = mx.take_along_axis(
                        kv_latent,
                        mx.broadcast_to(
                            idx, idx.shape[:-1] + (kv_latent.shape[-1],)
                        ),
                        axis=2,
                    )
                    sel_mask = valid_sel[:, :, None, :]
                    if mask is not None and mask.dtype == mx.bool_:
                        mkeys = mask.reshape(B, -1, Kv)[:, 0, :]
                        gathered = mx.take_along_axis(
                            mx.broadcast_to(
                                mkeys[:, None, :],
                                (B, safe_indices.shape[1], Kv),
                            ),
                            safe_indices,
                            axis=-1,
                        )
                        sel_mask = sel_mask & gathered[:, :, None, :]
                    attn_mask = sel_mask
                else:
                    sparse_mask, _ = build_prefill_indexpool_mask(topk_indices, Kv)
                    if mask is not None and mask.dtype == mx.bool_:
                        sparse_mask = sparse_mask & mask
                    attn_mask = sparse_mask

            if cache is not None and cache[0] is not None and cache[1] is not None:
                if isinstance(cache[1], CompactIndexPoolCache):
                    dependencies = cache[1].dependency_arrays()
                    if cache[0].keys is not None and dependencies:
                        cache[0].keys = mx.depends(cache[0].keys, dependencies)
                elif cache[1].keys is not None:
                    cache[0].keys = mx.depends(
                        cache[0].keys, (cache[1].keys, cache[1].values)
                    )

            if L == 1:
                q = self.embed_q(q)
                k = v = kv_latent
            else:
                k = self.embed_q(kv_latent, transpose=False)
                v = self.unembed_out(kv_latent)

            output = glm.scaled_dot_product_attention(
                q, k, v, cache=cache, scale=self.scale, mask=attn_mask
            )
            if L == 1:
                output = self.unembed_out(output)

            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            return self.o_proj(output)

        glm.Glm5NextSparseAttention.__call__ = sparse_call

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

        original_make_cache = glm.LanguageModel.make_cache

        def make_cache(self):
            if getattr(self, "_glm53_cache_backend", "direct") != "compact-nope-dsa":
                return original_make_cache(self)
            from mlx_vlm.models.cache import ArraysCache

            capacity_tokens = int(
                getattr(self, "_glm53_compact_cache_capacity_tokens", 4352)
            )
            caches = []
            for layer in self.layers:
                if layer.is_linear:
                    caches.append(ArraysCache(size=2))
                else:
                    caches.append(
                        make_compact_nope_dsa_cache(
                            layer.self_attn.indexer,
                            capacity_tokens=capacity_tokens,
                        )
                    )
            return caches

        glm.LanguageModel.make_cache = make_cache
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
            "indexpool_sentinel_and_range",
            "attention_gather_range_recheck",
            "compact_nope_dsa_cache_dispatch",
            "compact_nope_dsa_atomic_transitions",
            "mhc_kda_float32_storage",
        ],
    }
