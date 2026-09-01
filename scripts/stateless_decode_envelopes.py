"""Probe-only stateless compilation envelopes for GLM-5.3 decode.

The recurrent/sparse attention call and every cache object stay outside the
compiled functions.  Only tensor inputs and immutable module parameters cross
the compilation boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


ARMS = {
    "A": "compiled-ffn-baseline",
    "B": "post-attention-through-ffn",
    "C": "ffn-through-next-attention-handoff",
    "D": "post-attention-through-next-attention-handoff",
}
SPARSE_LAYERS = tuple(range(3, 45))


@dataclass(frozen=True)
class EnvelopePolicy:
    arm: str
    description: str
    compiled_sparse_layers: int
    compiled_callable_count: int
    stateful_attention_inside_compiled_callable: bool = False
    mutable_cache_inside_compiled_callable: bool = False


def _prepare_attention(layer, hidden):
    residual = hidden
    collapsed, post, comb = layer.attn_hc(hidden)
    normalized = layer.input_layernorm(collapsed)
    return normalized, residual, post, comb


def _finish_attention(layer, attention_output, residual, post, comb):
    from mlx_vlm.models.glm5_next.language import hc_expand

    return hc_expand(attention_output, residual, post, comb)


def _finish_ffn(layer, hidden):
    return layer._ffn_block(hidden)


def _finish_attention_and_ffn(
    layer, attention_output, residual, post, comb
):
    hidden = _finish_attention(layer, attention_output, residual, post, comb)
    return _finish_ffn(layer, hidden)


def _finish_ffn_and_prepare_next(layer, next_layer, hidden):
    hidden = _finish_ffn(layer, hidden)
    normalized, residual, post, comb = _prepare_attention(next_layer, hidden)
    return normalized, residual, post, comb


def _finish_attention_ffn_and_prepare_next(
    layer, next_layer, attention_output, residual, post, comb
):
    hidden = _finish_attention(layer, attention_output, residual, post, comb)
    hidden = _finish_ffn(layer, hidden)
    normalized, next_residual, next_post, next_comb = _prepare_attention(
        next_layer, hidden
    )
    return normalized, next_residual, next_post, next_comb


class StatelessDecodeEnvelopeRunner:
    """Run the text model with cache mutation outside every compiled closure."""

    def __init__(self, language_model, arm: str):
        if arm not in ARMS:
            raise ValueError(f"unknown stateless envelope arm: {arm}")
        self.language_model = language_model
        self.arm = arm
        self.layers = language_model.model.layers
        self._compiled: dict[int, object] = {}

        for layer_id in SPARSE_LAYERS:
            layer = self.layers[layer_id]
            next_layer = self.layers[layer_id + 1] if layer_id + 1 < len(self.layers) else None
            if arm == "A":
                layer.compile_ffn = True
                layer._ffn_c = None
                continue
            layer.compile_ffn = False
            layer._ffn_c = None
            if arm == "B":
                self._compiled[layer_id] = mx.compile(
                    lambda attention_output, residual, post, comb, layer=layer: (
                        _finish_attention_and_ffn(
                            layer, attention_output, residual, post, comb
                        )
                    )
                )
            elif arm == "C":
                if next_layer is None:
                    self._compiled[layer_id] = mx.compile(
                        lambda hidden, layer=layer: _finish_ffn(layer, hidden)
                    )
                else:
                    self._compiled[layer_id] = mx.compile(
                        lambda hidden, layer=layer, next_layer=next_layer: (
                            _finish_ffn_and_prepare_next(layer, next_layer, hidden)
                        )
                    )
            elif arm == "D":
                if next_layer is None:
                    self._compiled[layer_id] = mx.compile(
                        lambda attention_output, residual, post, comb, layer=layer: (
                            _finish_attention_and_ffn(
                                layer, attention_output, residual, post, comb
                            )
                        )
                    )
                else:
                    self._compiled[layer_id] = mx.compile(
                        lambda attention_output,
                        residual,
                        post,
                        comb,
                        layer=layer,
                        next_layer=next_layer: (
                            _finish_attention_ffn_and_prepare_next(
                                layer,
                                next_layer,
                                attention_output,
                                residual,
                                post,
                                comb,
                            )
                        )
                    )

        # Dense layers remain on the pinned upstream eager path for every arm.
        for layer_id, layer in enumerate(self.layers[: SPARSE_LAYERS[0]]):
            layer.compile_ffn = False
            layer._ffn_c = None

    @property
    def policy(self) -> EnvelopePolicy:
        compiled = len(SPARSE_LAYERS) if self.arm == "A" else len(self._compiled)
        return EnvelopePolicy(
            arm=self.arm,
            description=ARMS[self.arm],
            compiled_sparse_layers=len(SPARSE_LAYERS),
            compiled_callable_count=compiled,
        )

    def __call__(self, inputs, *, cache=None, inputs_embeds=None):
        from mlx_vlm.models.base import create_attention_mask, create_ssm_mask
        from mlx_vlm.models.glm5_next.language import LanguageModelOutput

        core = self.language_model.model
        hidden = (
            core.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        )
        if cache is None:
            cache = [None] * len(self.layers)

        fa_cache = cache[core.fa_idx]
        fa_mask = create_attention_mask(
            hidden, fa_cache[0] if fa_cache else None, return_array=True
        )
        ssm_mask = create_ssm_mask(hidden, cache[core.ssm_idx])
        hidden = mx.broadcast_to(
            hidden[:, :, None, :],
            (hidden.shape[0], hidden.shape[1], core.hc_mult, hidden.shape[2]),
        )
        hidden = mx.contiguous(hidden)

        prepared = None
        for layer_id, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            mask = ssm_mask if layer.is_linear else fa_mask
            if layer_id not in SPARSE_LAYERS:
                hidden = layer(hidden, mask=mask, cache=layer_cache)
                continue

            if self.arm == "A":
                hidden = layer(hidden, mask=mask, cache=layer_cache)
                continue

            if prepared is None:
                normalized, residual, post, comb = _prepare_attention(layer, hidden)
            else:
                normalized, residual, post, comb = prepared
                prepared = None

            # This is the only stateful operation in the sparse-layer loop.
            attention_output = layer.self_attn(
                normalized, mask=mask, cache=layer_cache
            )
            compiled = self._compiled[layer_id]

            if self.arm == "B":
                hidden = compiled(attention_output, residual, post, comb)
            elif self.arm == "C":
                hidden = _finish_attention(
                    layer, attention_output, residual, post, comb
                )
                result = compiled(hidden)
                if layer_id + 1 < len(self.layers):
                    prepared = result
                else:
                    hidden = result
            else:
                result = compiled(attention_output, residual, post, comb)
                if layer_id + 1 < len(self.layers):
                    prepared = result
                else:
                    hidden = result

        hidden = core.norm(hidden.mean(axis=2))
        if self.language_model.args.tie_word_embeddings:
            logits = core.embed_tokens.as_linear(hidden)
        else:
            logits = self.language_model.lm_head(hidden)
        return LanguageModelOutput(logits=logits)
