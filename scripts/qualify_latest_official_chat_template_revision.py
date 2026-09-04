#!/usr/bin/env python3
"""Separate and qualify an official chat-template-only checkpoint update."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.metadata
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from transformers import AutoTokenizer

from glm53_flash_mlx.abi import MLX_VLM_REVISION
from glm53_flash_mlx.manifest import (
    APPROVED_CHAT_TEMPLATE_REVISIONS,
    BASELINE_OFFICIAL_SNAPSHOT_REVISION,
    EXPECTED_TOKENIZER_DIGEST,
    LATEST_OFFICIAL_SNAPSHOT_REVISION,
    OFFICIAL_HF_REVISION,
    OFFICIAL_TOKENIZER_REVISION,
    TOKENIZER_IDENTITY_FILES,
    _component_digest_from_file_hashes,
    chat_template_content_digest,
    component_file_hashes,
    inspect_checkpoint,
    revision_identity_from_digests,
    tokenizer_content_digest,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OLD = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_NEW = Path("/Volumes/KIOXIA-PRO-1/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-latest-official-chat-template-revision-20260905.json"
)
METADATA_FILES = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "processor_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "model.safetensors.index.json",
    ".gitattributes",
    "LICENSE",
    "README.md",
)
RUNTIME_METADATA_FILES = METADATA_FILES[:-3]
ANCILLARY_REPOSITORY_FILES = METADATA_FILES[-3:]
DIFF_FILES = (
    "chat_template.jinja",
    "tokenizer_config.json",
    "config.json",
    "generation_config.json",
    "README.md",
)
SCHEMA = "glm53-latest-official-chat-template-revision-v1"
_PRINT_LOCK = threading.Lock()


def _source_release() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode())


def _atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _unified_diff(old: Path, new: Path, name: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.read_text().splitlines(keepends=True),
            new.read_text().splitlines(keepends=True),
            fromfile=f"baseline/{name}",
            tofile=f"candidate/{name}",
        )
    )


def _download_metadata(root: Path, name: str) -> dict[str, object]:
    path = root / ".cache" / "huggingface" / "download" / f"{name}.metadata"
    if not path.is_file():
        return {"available": False}
    lines = path.read_text().splitlines()
    return {
        "available": True,
        "source_snapshot_revision": lines[0] if lines else None,
        "etag": lines[1] if len(lines) > 1 else None,
        "download_timestamp": float(lines[2]) if len(lines) > 2 else None,
    }


def _metadata(root: Path) -> dict[str, dict[str, object]]:
    rows = {}
    for name in METADATA_FILES:
        path = root / name
        rows[name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256_bytes(path.read_bytes()),
            "download": _download_metadata(root, name),
        }
    return rows


def _weight_manifest(root: Path, *, chunk_mb: int) -> dict[str, object]:
    index = json.loads((root / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    rows = []
    for offset, name in enumerate(["model.safetensors.index.json", *shards], 1):
        row = component_file_hashes(root, [name], chunk_mb=chunk_mb)[0]
        rows.append(row)
        if offset == 1 or offset % 8 == 0 or offset == len(shards) + 1:
            with _PRINT_LOCK:
                print(
                    json.dumps(
                        {
                            "phase": "weight-sha256",
                            "root": str(root),
                            "completed_files": offset,
                            "total_files": len(shards) + 1,
                        }
                    ),
                    flush=True,
                )
    digest = _component_digest_from_file_hashes(
        "glm53-checkpoint-weight-content-v1", rows
    )
    return {
        "shard_count": len(shards),
        "file_count": len(rows),
        "total_bytes": sum(row[1] for row in rows),
        "checkpoint_digest": digest,
        "files": [
            {"name": name, "bytes": size, "sha256": sha256}
            for name, size, sha256 in rows
        ],
    }


def _subsequence_positions(values: list[int], needle: list[int]) -> list[int]:
    if not needle:
        return []
    return [
        offset
        for offset in range(len(values) - len(needle) + 1)
        if values[offset : offset + len(needle)] == needle
    ]


def _boundaries(tokenizer, rendered: str, token_ids: list[int]) -> dict[str, object]:
    markers = ("<|assistant|>", "<tool_call>", "<|observation|>", "<tool_response>")
    rows = {}
    for marker in markers:
        marker_ids = tokenizer.encode(marker, add_special_tokens=False)
        rows[marker] = {
            "character_offsets": [
                offset
                for offset in range(len(rendered))
                if rendered.startswith(marker, offset)
            ],
            "token_offsets": _subsequence_positions(token_ids, marker_ids),
            "marker_token_ids": marker_ids,
        }
    assistant = rows["<|assistant|>"]
    return {
        "markers": rows,
        "assistant_generation_character_offset": (
            assistant["character_offsets"][-1]
            if rendered.endswith("<|assistant|><think>")
            else None
        ),
        "assistant_generation_token_offset": (
            assistant["token_offsets"][-1]
            if rendered.endswith("<|assistant|><think>")
            else None
        ),
    }


def _tools() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "clock",
                "description": "Get time",
                "parameters": {
                    "type": "object",
                    "properties": {"zone": {"type": "string"}},
                    "required": ["zone"],
                },
            },
        },
    ]


def _fixtures() -> dict[str, dict[str, object]]:
    oracle = json.loads((REPOSITORY / "oracles/glm53-official-greedy-16.json").read_text())
    tool_turn = [
        {"role": "user", "content": "Weather and time?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-weather",
                    "type": "function",
                    "function": {"name": "weather", "arguments": {"city": "東京"}},
                },
                {
                    "id": "call-clock",
                    "type": "function",
                    "function": {"name": "clock", "arguments": {"zone": "JST"}},
                },
            ],
        },
    ]
    return {
        "accepted_oracle_prompt": {
            "messages": [{"role": "user", "content": oracle["prompt"]}],
            "add_generation_prompt": True,
        },
        "system_user": {
            "messages": [
                {"role": "system", "content": "Be exact."},
                {"role": "user", "content": "Answer in Japanese."},
            ],
            "add_generation_prompt": True,
        },
        "multi_turn": {
            "messages": [
                {"role": "user", "content": "First"},
                {
                    "role": "assistant",
                    "reasoning_content": "private reasoning",
                    "content": "First answer",
                },
                {"role": "user", "content": "Second"},
            ],
            "add_generation_prompt": True,
        },
        "assistant": {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Completed answer"},
            ],
            "add_generation_prompt": False,
        },
        "tool_declaration": {
            "messages": [{"role": "user", "content": "Use a tool"}],
            "tools": _tools(),
            "add_generation_prompt": True,
        },
        "tool_call": {
            "messages": tool_turn,
            "tools": _tools(),
            "add_generation_prompt": False,
        },
        "tool_result": {
            "messages": tool_turn
            + [
                {
                    "role": "tool",
                    "tool_call_id": "call-clock",
                    "content": "12:00",
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-weather",
                    "content": "sunny",
                },
            ],
            "tools": _tools(),
            "add_generation_prompt": True,
        },
        "reasoning_effort": {
            "messages": [{"role": "user", "content": "Think briefly"}],
            "reasoning_effort": "low",
            "add_generation_prompt": True,
        },
        "clear_thinking": {
            "messages": [
                {"role": "user", "content": "Question one"},
                {
                    "role": "assistant",
                    "reasoning_content": "discard this",
                    "content": "Answer one",
                },
                {"role": "user", "content": "Question two"},
            ],
            "clear_thinking": True,
            "add_generation_prompt": True,
        },
        "none_content": {
            "messages": [{"role": "user", "content": None}],
            "add_generation_prompt": True,
        },
    }


def _render(tokenizer, fixture: dict[str, object]) -> dict[str, object]:
    rendered = tokenizer.apply_chat_template(fixture["messages"], tokenize=False, **{
        key: value for key, value in fixture.items() if key != "messages"
    })
    encoded = tokenizer.apply_chat_template(fixture["messages"], tokenize=True, **{
        key: value for key, value in fixture.items() if key != "messages"
    })
    # transformers may return either a plain list or a BatchEncoding depending
    # on tokenizer configuration.  The semantic token stream is input_ids in
    # both cases; keep the artifact independent of that container detail.
    token_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("chat-template qualification requires one sequence")
        token_ids = token_ids[0]
    token_ids = [int(value) for value in token_ids]
    return {
        "rendered_text": rendered,
        "rendered_sha256": _sha256_text(rendered),
        "token_ids": token_ids,
        "token_ids_sha256": _sha256_bytes(
            b"".join(int(value).to_bytes(4, "little") for value in token_ids)
        ),
        "token_count": len(token_ids),
        "boundaries": _boundaries(tokenizer, rendered, token_ids),
    }


def _template_rows(old: Path, new: Path) -> dict[str, object]:
    from mlx_vlm.tokenizer_utils import load_tokenizer

    roots = {"baseline": old, "candidate": new}
    official = {
        name: AutoTokenizer.from_pretrained(root) for name, root in roots.items()
    }
    runtime = {name: load_tokenizer(root)._tokenizer for name, root in roots.items()}
    cases = {}
    for case_name, fixture in _fixtures().items():
        case = {}
        for root_name in roots:
            reference = _render(official[root_name], fixture)
            actual = _render(runtime[root_name], fixture)
            case[root_name] = reference
            case[f"{root_name}_runtime_exact"] = actual == reference
        case["rendered_text_changed"] = (
            case["baseline"]["rendered_text"]
            != case["candidate"]["rendered_text"]
        )
        case["token_ids_changed"] = (
            case["baseline"]["token_ids"] != case["candidate"]["token_ids"]
        )
        case["rendered_unified_diff"] = "".join(
            difflib.unified_diff(
                case["baseline"]["rendered_text"].splitlines(keepends=True),
                case["candidate"]["rendered_text"].splitlines(keepends=True),
                fromfile=f"baseline/{case_name}",
                tofile=f"candidate/{case_name}",
            )
        )
        cases[case_name] = case
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--new", type=Path, default=DEFAULT_NEW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-mb", type=int, default=32)
    args = parser.parse_args()
    old = args.old.expanduser().resolve()
    new = args.new.expanduser().resolve()
    if old == new:
        raise ValueError("baseline and candidate checkpoint roots must differ")
    if args.chunk_mb < 1:
        raise ValueError("chunk-mb must be positive")

    metadata = {"baseline": _metadata(old), "candidate": _metadata(new)}
    diffs = {
        name: _unified_diff(old / name, new / name, name) for name in DIFF_FILES
    }
    previous = None
    if args.output.is_file():
        try:
            previous = json.loads(args.output.read_text())
        except (OSError, json.JSONDecodeError):
            previous = None
    previous_metadata = (previous or {}).get("metadata", {})
    reusable_weights = (
        previous is not None
        and previous.get("paths")
        == {"baseline": str(old), "candidate": str(new)}
        and set(previous.get("weights", {})) == {"baseline", "candidate"}
        and all(
            previous_metadata.get(root_name, {}).get(name) == metadata[root_name][name]
            for root_name in ("baseline", "candidate")
            for name in previous_metadata.get(root_name, {})
        )
    )
    artifact = {
        "schema": SCHEMA,
        "date": str(date.today()),
        "complete": False,
        "last_completed_phase": "metadata",
        "paths": {"baseline": str(old), "candidate": str(new)},
        "provenance": {
            "runtime_source_release": _source_release(),
            "mlx_vlm_revision": MLX_VLM_REVISION,
            "transformers_version": importlib.metadata.version("transformers"),
            "baseline_snapshot_revision": metadata["baseline"]["chat_template.jinja"][
                "download"
            ]["source_snapshot_revision"],
            "candidate_snapshot_revision": metadata["candidate"]["chat_template.jinja"][
                "download"
            ]["source_snapshot_revision"],
            "checkpoint_revision": OFFICIAL_HF_REVISION,
            "tokenizer_revision": OFFICIAL_TOKENIZER_REVISION,
        },
        "metadata": metadata,
        "metadata_unified_diffs": diffs,
        "acceptance": {},
    }
    if reusable_weights:
        artifact["weights"] = previous["weights"]
        artifact["weight_hashes_reused_from_incomplete_artifact"] = True
    _atomic_write(args.output, artifact)

    if reusable_weights:
        weights = artifact["weights"]
    else:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                name: executor.submit(_weight_manifest, root, chunk_mb=args.chunk_mb)
                for name, root in (("baseline", old), ("candidate", new))
            }
            weights = {name: future.result() for name, future in futures.items()}
        artifact["weights"] = weights
    artifact["last_completed_phase"] = "weights"
    _atomic_write(args.output, artifact)

    tokenizer_digests = {
        "baseline": tokenizer_content_digest(old),
        "candidate": tokenizer_content_digest(new),
    }
    template_digests = {
        "baseline": chat_template_content_digest(old),
        "candidate": chat_template_content_digest(new),
    }
    identities = {
        name: revision_identity_from_digests(
            checkpoint_digest=weights[name]["checkpoint_digest"],
            tokenizer_digest=tokenizer_digests[name],
            chat_template_digest=template_digests[name],
        )
        for name in ("baseline", "candidate")
    }
    artifact["revision_identity"] = {
        name: {
            **identity.descriptor(),
            "namespace_sha256": identity.namespace_sha256,
        }
        for name, identity in identities.items()
    }
    artifact["template_cases"] = _template_rows(old, new)
    artifact["checkpoint_reports"] = {
        "baseline": inspect_checkpoint(old, require_server_ready=True).to_dict(),
        "candidate": inspect_checkpoint(new, require_server_ready=True).to_dict(),
    }
    artifact["last_completed_phase"] = "rendered"
    _atomic_write(args.output, artifact)

    metadata_equal = {
        name: metadata["baseline"][name]["sha256"]
        == metadata["candidate"][name]["sha256"]
        for name in METADATA_FILES
    }
    cases = artifact["template_cases"]
    acceptance = {
        "all_62_weight_shards_byte_identical": (
            weights["baseline"]["shard_count"]
            == weights["candidate"]["shard_count"]
            == 62
            and weights["baseline"]["files"] == weights["candidate"]["files"]
        ),
        "checkpoint_weight_digest_identical": (
            weights["baseline"]["checkpoint_digest"]
            == weights["candidate"]["checkpoint_digest"]
        ),
        "runtime_metadata_except_chat_template_identical": all(
            metadata_equal[name]
            for name in RUNTIME_METADATA_FILES
            if name != "chat_template.jinja"
        ),
        "chat_template_is_only_changed_runtime_metadata": (
            not metadata_equal["chat_template.jinja"]
            and sum(
                not metadata_equal[name] for name in RUNTIME_METADATA_FILES
            )
            == 1
        ),
        "ancillary_repository_files_fully_audited": all(
            name in metadata["baseline"] and name in metadata["candidate"]
            for name in ANCILLARY_REPOSITORY_FILES
        ),
        "official_readme_is_only_changed_ancillary_file": (
            not metadata_equal["README.md"]
            and metadata_equal["LICENSE"]
            and metadata_equal[".gitattributes"]
        ),
        "baseline_and_candidate_revisions_exact": (
            artifact["provenance"]["baseline_snapshot_revision"]
            == BASELINE_OFFICIAL_SNAPSHOT_REVISION
            and artifact["provenance"]["candidate_snapshot_revision"]
            == LATEST_OFFICIAL_SNAPSHOT_REVISION
        ),
        "tokenizer_digest_identical_and_approved": (
            tokenizer_digests["baseline"]
            == tokenizer_digests["candidate"]
            == EXPECTED_TOKENIZER_DIGEST
        ),
        "template_digests_independently_approved": all(
            value in APPROVED_CHAT_TEMPLATE_REVISIONS
            for value in template_digests.values()
        ),
        "runtime_matches_official_reference_all_cases": all(
            row["baseline_runtime_exact"] and row["candidate_runtime_exact"]
            for row in cases.values()
        ),
        "accepted_oracle_prompt_render_and_tokens_unchanged": (
            not cases["accepted_oracle_prompt"]["rendered_text_changed"]
            and not cases["accepted_oracle_prompt"]["token_ids_changed"]
        ),
        "candidate_none_content_never_renders_literal_none": (
            "None" not in cases["none_content"]["candidate"]["rendered_text"]
            and "None" not in cases["tool_call"]["candidate"]["rendered_text"]
            and "None" not in cases["tool_result"]["candidate"]["rendered_text"]
        ),
        "tool_result_order_follows_declared_calls": (
            cases["tool_result"]["candidate"]["rendered_text"].index("sunny")
            < cases["tool_result"]["candidate"]["rendered_text"].index("12:00")
        ),
        "reasoning_effort_low_exact": (
            "Reasoning Effort: Low"
            in cases["reasoning_effort"]["candidate"]["rendered_text"]
        ),
        "clear_thinking_removes_prior_reasoning": (
            "discard this" not in cases["clear_thinking"]["candidate"]["rendered_text"]
            and "<think></think>"
            in cases["clear_thinking"]["candidate"]["rendered_text"]
        ),
        "assistant_and_tool_boundaries_recorded": all(
            row["candidate"]["boundaries"]["markers"]["<|assistant|>"][
                "token_offsets"
            ]
            for row in cases.values()
        )
        and bool(
            cases["tool_call"]["candidate"]["boundaries"]["markers"][
                "<tool_call>"
            ]["token_offsets"]
        ),
        "both_checkpoint_reports_server_ready": all(
            artifact["checkpoint_reports"][name]["server_ready"]
            for name in ("baseline", "candidate")
        ),
        "classification_runtime_metadata_template_only": True,
        "model_runtime_or_kernel_requalification_not_required": True,
        "server_not_started": True,
    }
    artifact["classification"] = {
        "kind": "runtime-metadata-template-only",
        "weights_changed": False,
        "tokenizer_changed": False,
        "processor_changed": False,
        "chat_template_changed": True,
        "ancillary_repository_files_changed": ["README.md"],
        "requires_fp8_kernel_requalification": False,
        "requires_template_semantics_qualification": True,
    }
    artifact["acceptance"] = acceptance
    artifact["complete"] = all(acceptance.values())
    artifact["last_completed_phase"] = "complete"
    _atomic_write(args.output, artifact)
    print(json.dumps({"complete": artifact["complete"], "acceptance": acceptance}))
    return 0 if artifact["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
