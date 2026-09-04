#!/usr/bin/env python3
"""Prove that value materialization is independent of cache write ownership."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from glm53_flash_mlx.abi import (
    CACHE_IDENTITY_SCHEMA,
    MLX_VLM_REVISION,
    NOPE_DSA_CACHE_ABI_COMPACT,
    NOPE_DSA_CACHE_ABI_DIRECT,
)
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import inspect_checkpoint
from glm53_flash_mlx.nope_cache import make_compact_nope_dsa_cache
from glm53_flash_mlx.patch import apply_runtime_patch
from glm53_flash_mlx.state_materialization import (
    CACHE_WRITE_SENTINEL,
    STATE_MATERIALIZATION_CONTRACT,
    MaterializationRequest,
    StateMaterializationError,
    execute_state_materialization,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-state-materialization-cache-write-ownership-20260904.json"
)


def _atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _hash_tree(value) -> str:
    digest = hashlib.sha256()

    def visit(item) -> None:
        if isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        elif item is None or isinstance(item, (str, int, bool)):
            digest.update(repr(item).encode())
        else:
            mx.eval(item)
            raw = item.view(mx.uint16) if item.dtype == mx.bfloat16 else item
            array = np.ascontiguousarray(np.asarray(raw))
            digest.update(str(tuple(item.shape)).encode())
            digest.update(str(item.dtype).encode())
            digest.update(array.tobytes())

    visit(value)
    return digest.hexdigest()


def _copy_tree(value):
    if isinstance(value, tuple):
        return tuple(_copy_tree(item) for item in value)
    if isinstance(value, list):
        return [_copy_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _copy_tree(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, bool)):
        return value
    copied = mx.array(value)
    mx.eval(copied)
    return copied


def _snapshot(cache) -> tuple[dict, ...]:
    return tuple(
        {
            "state": _copy_tree(child.state),
            "meta_state": tuple(child.meta_state),
        }
        for child in cache.caches
    )


def _restore(cache, snapshot: tuple[dict, ...]) -> None:
    if len(cache.caches) != len(snapshot):
        raise ValueError("snapshot/cache child count mismatch")
    for child, state in zip(cache.caches, snapshot, strict=True):
        child.prefix_cache_restore(state)
    arrays = [value for child in cache.caches for value in child.dependency_arrays()]
    if arrays:
        mx.eval(*arrays)


def _cache_observation(cache) -> dict:
    return {
        "state_hash": _hash_tree(
            tuple((child.state, child.meta_state) for child in cache.caches)
        ),
        "logical_offsets": [int(child.offset) for child in cache.caches],
        "physical_capacity": [
            int(cache[0].physical_capacity_tokens),
            int(cache[1].physical_capacity_rows),
        ],
        "resident_bytes": sum(int(child.nbytes) for child in cache.caches),
    }


def _make_indexer(*, topk: int = 32, kpool: int = 4):
    apply_runtime_patch()
    from mlx_vlm.models.glm5_next.language import Glm5NextIndexer

    config = SimpleNamespace(
        hidden_size=8,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=topk,
        index_kpool=kpool,
        index_kpool_always_select_tail=True,
        q_lora_rank=4,
    )
    mx.random.seed(409)
    indexer = Glm5NextIndexer(config)
    indexer.set_dtype(mx.bfloat16)
    return indexer


def _inputs(start: int, tokens: int):
    positions = mx.arange(start, start + tokens, dtype=mx.float32)[:, None]
    x = mx.sin(positions * 0.03125 + mx.arange(8)[None] * 0.0625)[None]
    qr = mx.cos(positions * 0.046875 + mx.arange(4)[None] * 0.09375)[None]
    latent = mx.sin(
        positions * 0.015625 + mx.arange(4, dtype=mx.float32)[None] * 0.125
    ).reshape(1, 1, tokens, 4)
    return (
        x.astype(mx.bfloat16),
        qr.astype(mx.bfloat16),
        latent.astype(mx.bfloat16),
    )


def _new_cache(indexer, *, tokens: int, capacity: int = 64):
    cache = make_compact_nope_dsa_cache(indexer, capacity_tokens=capacity)
    if tokens:
        x, qr, latent = _inputs(0, tokens)
        indexer(x, qr, None, cache=cache[1])
        cache[0].update_and_fetch(latent, latent)
        mx.eval(*[value for child in cache.caches for value in child.dependency_arrays()])
    return cache


def _clone_cache(indexer, source, *, capacity: int = 64):
    clone = make_compact_nope_dsa_cache(indexer, capacity_tokens=capacity)
    _restore(clone, _snapshot(source))
    return clone


def _projection(indexer, x, length: int):
    keys = indexer.k_norm(indexer.wk(x)).reshape(1, length, indexer.head_dim)
    gates = x @ indexer.index_kpool_compress_gate.swapaxes(-1, -2)
    valid = mx.ones((1, length), dtype=mx.bool_)
    mx.eval(keys, gates, valid)
    return keys, gates, valid


def _compact_cache_fixture() -> dict:
    indexer = _make_indexer(topk=32)
    base = _new_cache(indexer, tokens=4)
    reference = _clone_cache(indexer, base)
    candidate = _clone_cache(indexer, base)
    no_owner = _clone_cache(indexer, base)
    no_owner_before = _cache_observation(no_owner)
    _, _, latent = _inputs(4, 1)

    reference[0].update_and_fetch(latent, latent)
    produced = {"a": 0, "b": 0, "c": 0, "invalid": 0}

    def producer(name):
        def run():
            produced[name] += 1
            return mx.contiguous(latent, allow_col_major=False)

        return run

    arm_a = execute_state_materialization(
        MaterializationRequest(True, 0),
        cache_capacity=1,
        producer=producer("a"),
        cache_writer=lambda _slot, value: candidate[0].update_and_fetch(value, value),
    )
    arm_b = execute_state_materialization(
        MaterializationRequest(True, CACHE_WRITE_SENTINEL),
        cache_capacity=1,
        producer=producer("b"),
        cache_writer=lambda _slot, value: no_owner[0].update_and_fetch(value, value),
    )
    arm_c = execute_state_materialization(
        MaterializationRequest(False, None),
        cache_capacity=1,
        producer=producer("c"),
        cache_writer=lambda _slot, value: no_owner[0].update_and_fetch(value, value),
    )
    invalid_before = _cache_observation(no_owner)
    invalid_rejected = False
    try:
        execute_state_materialization(
            MaterializationRequest(True, 1),
            cache_capacity=1,
            producer=producer("invalid"),
            cache_writer=lambda _slot, value: no_owner[0].update_and_fetch(value, value),
        )
    except StateMaterializationError:
        invalid_rejected = True
    return {
        "arm_a_materialized_and_written": (
            arm_a.materialized and arm_a.cache_written
        ),
        "arm_a_value_exact": _hash_tree(arm_a.value) == _hash_tree(latent),
        "arm_a_cache_exact": (
            _cache_observation(candidate) == _cache_observation(reference)
        ),
        "arm_b_materialized_without_write": (
            arm_b.materialized and not arm_b.cache_written and produced["b"] == 1
        ),
        "arm_b_value_exact": _hash_tree(arm_b.value) == _hash_tree(arm_a.value),
        "arm_b_cache_and_accounting_unchanged": (
            _cache_observation(no_owner) == no_owner_before
        ),
        "arm_c_allocation_free_noop": (
            not arm_c.materialized
            and not arm_c.cache_written
            and produced["c"] == 0
        ),
        "invalid_destination_preflight_atomic": (
            invalid_rejected
            and produced["invalid"] == 0
            and _cache_observation(no_owner) == invalid_before
        ),
        "producer_calls": produced,
    }


def _apc_restore_fixture() -> dict:
    indexer = _make_indexer(topk=32)
    source = _new_cache(indexer, tokens=7)
    snapshot = _snapshot(source)
    snapshot_hash = _hash_tree(snapshot)
    live = _new_cache(indexer, tokens=3)
    reference = _clone_cache(indexer, live)
    candidate = _clone_cache(indexer, live)
    no_owner = _clone_cache(indexer, live)
    no_owner_before = _cache_observation(no_owner)
    produced = {"a": 0, "b": 0, "c": 0, "invalid": 0}

    def producer(name):
        def run():
            produced[name] += 1
            return _copy_tree(snapshot)

        return run

    _restore(reference, snapshot)
    arm_a = execute_state_materialization(
        MaterializationRequest(True, 0),
        cache_capacity=1,
        producer=producer("a"),
        cache_writer=lambda _slot, value: _restore(candidate, value),
    )
    arm_b = execute_state_materialization(
        MaterializationRequest(True, None),
        cache_capacity=1,
        producer=producer("b"),
        cache_writer=lambda _slot, value: _restore(no_owner, value),
    )
    arm_c = execute_state_materialization(
        MaterializationRequest(False, CACHE_WRITE_SENTINEL),
        cache_capacity=1,
        producer=producer("c"),
        cache_writer=lambda _slot, value: _restore(no_owner, value),
    )
    invalid_before = _cache_observation(no_owner)
    invalid_rejected = False
    try:
        execute_state_materialization(
            MaterializationRequest(True, -2),
            cache_capacity=1,
            producer=producer("invalid"),
            cache_writer=lambda _slot, value: _restore(no_owner, value),
        )
    except StateMaterializationError:
        invalid_rejected = True
    return {
        "arm_a_materialized_and_restored": (
            arm_a.materialized and arm_a.cache_written
        ),
        "arm_a_restore_exact": (
            _cache_observation(candidate) == _cache_observation(reference)
        ),
        "arm_b_materialized_without_restore": (
            arm_b.materialized and not arm_b.cache_written and produced["b"] == 1
        ),
        "arm_b_value_exact": _hash_tree(arm_b.value) == _hash_tree(arm_a.value),
        "arm_b_live_cache_unchanged": _cache_observation(no_owner) == no_owner_before,
        "snapshot_immutable": _hash_tree(snapshot) == snapshot_hash,
        "arm_c_allocation_free_noop": (
            not arm_c.materialized
            and not arm_c.cache_written
            and produced["c"] == 0
        ),
        "invalid_destination_preflight_atomic": (
            invalid_rejected
            and produced["invalid"] == 0
            and _cache_observation(no_owner) == invalid_before
        ),
        "producer_calls": produced,
    }


def _prefill_decode_transition_fixture() -> dict:
    indexer = _make_indexer(topk=32)
    base = _new_cache(indexer, tokens=32)
    reference = _clone_cache(indexer, base)
    candidate = _clone_cache(indexer, base)
    no_owner = _clone_cache(indexer, base)
    no_owner_before = _cache_observation(no_owner)
    x, qr, latent = _inputs(32, 1)
    selected_reference = indexer(x, qr, None, cache=reference[1])
    reference[0].update_and_fetch(latent, latent)
    produced = {"a": 0, "b": 0, "c": 0, "invalid": 0}
    selected_candidate = []

    def producer(name):
        def run():
            produced[name] += 1
            keys, gates, valid = _projection(indexer, x, 1)
            return keys, gates, valid, mx.contiguous(latent, allow_col_major=False)

        return run

    def persist(_slot, value) -> None:
        keys, gates, valid, latent_value = value
        candidate[1]._append_projected(keys, gates, valid)
        selected_candidate.append(
            candidate[1]._decode_selection(indexer, x, qr, valid)
        )
        candidate[0].update_and_fetch(latent_value, latent_value)

    arm_a = execute_state_materialization(
        MaterializationRequest(True, 0),
        cache_capacity=1,
        producer=producer("a"),
        cache_writer=persist,
    )
    arm_b = execute_state_materialization(
        MaterializationRequest(True, CACHE_WRITE_SENTINEL),
        cache_capacity=1,
        producer=producer("b"),
        cache_writer=lambda _slot, _value: None,
    )
    arm_c = execute_state_materialization(
        MaterializationRequest(False, None),
        cache_capacity=1,
        producer=producer("c"),
        cache_writer=lambda _slot, _value: None,
    )
    invalid_before = _cache_observation(no_owner)
    invalid_rejected = False
    try:
        execute_state_materialization(
            MaterializationRequest(True, 1),
            cache_capacity=1,
            producer=producer("invalid"),
            cache_writer=lambda _slot, _value: None,
        )
    except StateMaterializationError:
        invalid_rejected = True
    mx.eval(selected_reference, selected_candidate[0])
    return {
        "transition_context": {"prefill_tokens": 32, "decode_tokens": 1},
        "arm_a_materialized_and_written": (
            arm_a.materialized and arm_a.cache_written
        ),
        "arm_a_projection_exact": _hash_tree(arm_a.value) == _hash_tree(arm_b.value),
        "arm_a_selected_indices_exact": (
            _hash_tree(selected_candidate[0]) == _hash_tree(selected_reference)
        ),
        "arm_a_cache_exact": (
            _cache_observation(candidate) == _cache_observation(reference)
        ),
        "arm_b_materialized_without_write": (
            arm_b.materialized and not arm_b.cache_written and produced["b"] == 1
        ),
        "arm_b_cache_and_accounting_unchanged": (
            _cache_observation(no_owner) == no_owner_before
        ),
        "arm_c_allocation_free_noop": (
            not arm_c.materialized
            and not arm_c.cache_written
            and produced["c"] == 0
        ),
        "invalid_destination_preflight_atomic": (
            invalid_rejected
            and produced["invalid"] == 0
            and _cache_observation(no_owner) == invalid_before
        ),
        "producer_calls": produced,
    }


def _load_probe_modules():
    scripts = str(REPOSITORY / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import probe_exact_sigmoid_gate_metal_barrier as oracle_probe
    import probe_packed_decode_runtime as packed_probe

    return oracle_probe, packed_probe


def _full_model(path: Path, report) -> dict:
    oracle_probe, packed_probe = _load_probe_modules()
    mx.set_wired_limit(int(440e9))
    mx.set_cache_limit(int(32e9))
    started = time.perf_counter()
    model, processor = load(
        path,
        experimental_packed_decode_moe=True,
        experimental_compact_nope_dsa_cache=True,
    )
    load_seconds = time.perf_counter() - started
    warm_started = time.perf_counter()
    warm_residency(model)
    warm_seconds = time.perf_counter() - warm_started
    oracle = oracle_probe._official_oracle(model, processor, report)
    vocab = int(model.language_model.lm_head.weight.shape[0])
    ram_apc = packed_probe._ram_apc(model, vocab)
    return {
        "executed": True,
        "load_seconds": load_seconds,
        "warm_residency_seconds": warm_seconds,
        "moe_backend": getattr(model, "_glm53_moe_backend", None),
        "cache_backend": getattr(model, "_glm53_cache_backend", None),
        "official_oracle": oracle,
        "ram_apc": ram_apc,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not mx.metal.is_available():
        raise RuntimeError("state materialization ownership probe requires MLX/Metal")

    report = inspect_checkpoint(args.model, require_server_ready=True)
    fixtures = {
        "compact_cache": _compact_cache_fixture(),
        "ram_apc_restore": _apc_restore_fixture(),
        "prefill_decode_transition": _prefill_decode_transition_fixture(),
    }
    full_model = _full_model(args.model, report)
    acceptance = {
        "all_three_paths_exercised": set(fixtures)
        == {"compact_cache", "ram_apc_restore", "prefill_decode_transition"},
        "materialize_and_cache_write_exact": all(
            fixture.get("arm_a_cache_exact", fixture.get("arm_a_restore_exact", False))
            for fixture in fixtures.values()
        ),
        "materialize_without_owner_value_exact": all(
            fixture["arm_b_value_exact"]
            if "arm_b_value_exact" in fixture
            else fixture["arm_a_projection_exact"]
            for fixture in fixtures.values()
        ),
        "materialize_without_owner_state_unchanged": all(
            fixture.get(
                "arm_b_cache_and_accounting_unchanged",
                fixture.get("arm_b_live_cache_unchanged", False),
            )
            for fixture in fixtures.values()
        ),
        "no_materialization_no_owner_is_noop": all(
            fixture["arm_c_allocation_free_noop"] for fixture in fixtures.values()
        ),
        "invalid_destination_is_preflight_atomic": all(
            fixture["invalid_destination_preflight_atomic"]
            for fixture in fixtures.values()
        ),
        "apc_snapshot_immutable": fixtures["ram_apc_restore"]["snapshot_immutable"],
        "prefill_decode_selection_exact": fixtures["prefill_decode_transition"][
            "arm_a_selected_indices_exact"
        ],
        "official_16_token_oracle_exact": full_model["official_oracle"][
            "first_16_match"
        ],
        "official_128_token_oracle_exact": full_model["official_oracle"][
            "full_128_match"
        ],
        "ram_apc_continuation_exact": (
            full_model["ram_apc"]["all_logits_hashes_match"]
            and full_model["ram_apc"]["post_state_exact"]
            and full_model["ram_apc"]["snapshot_immutable"]
            and full_model["ram_apc"]["steps"] == 16
        ),
    }
    artifact = {
        "schema": "glm53-state-materialization-cache-write-ownership-v1",
        "date": date.today().isoformat(),
        "complete": all(acceptance.values()),
        "probe_only": True,
        "official_hf_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "contract": {
            "identity": STATE_MATERIALIZATION_CONTRACT,
            "cache_write_sentinel": CACHE_WRITE_SENTINEL,
            "materialization_requirement_independent_of_cache_ownership": True,
            "invalid_destination_checked_before_producer": True,
        },
        "existing_runtime_identity": {
            "cache_identity_schema": CACHE_IDENTITY_SCHEMA,
            "direct_attention_cache_abi": NOPE_DSA_CACHE_ABI_DIRECT,
            "compact_attention_cache_abi": NOPE_DSA_CACHE_ABI_COMPACT,
        },
        "fixtures": fixtures,
        "full_model": full_model,
        "runtime_changes": {
            "abi": False,
            "admission": False,
            "apc_namespace": False,
            "backend": False,
            "cache_implementation": False,
            "server": False,
        },
        "acceptance": acceptance,
        "decision": (
            "materialization_write_ownership_contract_ready_for_semantic_snapshot"
            if all(acceptance.values())
            else "stop_materialization_write_ownership_contract"
        ),
    }
    _atomic_write(args.output, artifact)
    print(json.dumps({"output": str(args.output), "complete": artifact["complete"]}))
    return 0 if artifact["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
