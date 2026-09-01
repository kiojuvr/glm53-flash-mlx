#!/usr/bin/env python3
"""Attribute bounded steady-decode GPU idle to dynamic command boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from capture_budget import atomic_write
from capture_steady_packed_decode_critical_path import _trace_identity


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-packed-decode-bounded-telemetry-20260901.json"
)
DEFAULT_OUTPUT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-steady-decode-dynamic-idle-attribution-20260901.json"
)
OPERATOR_MICROCAPTURE = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-packed-decode-operator-microcapture-20260901.json"
)
TRACED_TOKENS = 16
SCHEMAS = (
    "metal-gpu-intervals",
    "metal-application-command-buffer-submissions",
)
GAP_BINS_NS = (25_000, 100_000, 500_000)


@dataclass(frozen=True)
class GpuEvent:
    start_ns: int
    end_ns: int
    commit_ns: int
    command_buffer_id: int
    frame_number: int | None
    label: str
    commit_source: str


@dataclass(frozen=True)
class GpuGap:
    previous_end_ns: int
    next_commit_ns: int
    next_start_ns: int
    previous_frame: int | None
    next_frame: int | None
    previous_stage: str
    next_stage: str
    commit_source: str

    @property
    def total_ns(self) -> int:
        return self.next_start_ns - self.previous_end_ns

    @property
    def application_starvation_ns(self) -> int:
        return max(0, self.next_commit_ns - self.previous_end_ns)

    @property
    def driver_or_dependency_ns(self) -> int:
        ready_ns = max(self.previous_end_ns, self.next_commit_ns)
        return max(0, self.next_start_ns - ready_ns)


def _value(element, ids: dict[str, ET.Element]) -> tuple[str | None, str | None]:
    if reference := element.get("ref"):
        element = ids.get(reference, element)
    raw = (element.text or "").strip() or None
    return raw, element.get("fmt")


def _parse_export(path: Path, pid: int) -> list[dict]:
    root = ET.parse(path).getroot()
    node = root.find(".//node")
    if node is None:
        return []
    schema = node.find("schema")
    if schema is None:
        return []
    names = [
        column.findtext("mnemonic", default=f"column_{index}")
        for index, column in enumerate(schema.findall("col"))
    ]
    ids = {
        element.get("id"): element
        for element in node.iter()
        if element.get("id") is not None
    }
    needles = (f"({pid})", f"pid: {pid}")
    rows = []
    for row in node.findall("row"):
        values = {}
        formats = []
        for index, child in enumerate(list(row)):
            name = names[index] if index < len(names) else f"column_{index}"
            raw, formatted = _value(child, ids)
            values[name] = {"raw": raw, "fmt": formatted}
            if formatted:
                formats.append(formatted)
        if any(needle in formatted for formatted in formats for needle in needles):
            rows.append(values)
    return rows


def _integer(row: dict, name: str) -> int | None:
    raw = row.get(name, {}).get("raw")
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None


def _formatted(row: dict, name: str) -> str:
    return row.get(name, {}).get("fmt") or ""


def _export(trace: Path, destination: Path) -> dict[str, Path]:
    outputs = {}
    for schema in SCHEMAS:
        output = destination / f"{schema}.xml"
        subprocess.run(
            [
                "xcrun",
                "xctrace",
                "export",
                "--input",
                str(trace),
                "--xpath",
                f"/trace-toc/run[@number='1']/data/table[@schema='{schema}']",
                "--output",
                str(output),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        outputs[schema] = output
    return outputs


def _events(exports: dict[str, Path], pid: int) -> tuple[list[GpuEvent], dict]:
    submissions = _parse_export(
        exports["metal-application-command-buffer-submissions"], pid
    )
    commit_by_buffer = {}
    for row in submissions:
        command_buffer = _integer(row, "cmdbuffer-id")
        start = _integer(row, "start")
        duration = _integer(row, "duration")
        if command_buffer is None or start is None or duration is None:
            continue
        commit_by_buffer[command_buffer] = start + duration

    rows = _parse_export(exports["metal-gpu-intervals"], pid)
    events = []
    commit_sources = {"application_submission": 0, "start_latency": 0}
    for row in rows:
        start = _integer(row, "start")
        duration = _integer(row, "duration")
        command_buffer = _integer(row, "cmdbuffer-id")
        if start is None or duration is None or command_buffer is None:
            continue
        commit = commit_by_buffer.get(command_buffer)
        source = "application_submission"
        if commit is None:
            latency = _integer(row, "start-latency")
            commit = start - max(0, latency or 0)
            source = "start_latency"
        commit_sources[source] += 1
        events.append(
            GpuEvent(
                start_ns=start,
                end_ns=start + duration,
                commit_ns=commit,
                command_buffer_id=command_buffer,
                frame_number=_integer(row, "frame-number"),
                label=_formatted(row, "event-label"),
                commit_source=source,
            )
        )
    events.sort(key=lambda event: (event.start_ns, event.end_ns))
    return events, {
        "application_submission_rows": len(submissions),
        "application_commit_ids": len(commit_by_buffer),
        "gpu_event_rows": len(events),
        "commit_sources": commit_sources,
    }


def _unclassified(_event: GpuEvent) -> str:
    return "unclassified"


def reconstruct_gaps(
    events: list[GpuEvent], stage_of=_unclassified
) -> tuple[list[GpuGap], dict]:
    if not events:
        return [], {
            "gpu_busy_ns": 0,
            "gpu_idle_ns": 0,
            "gpu_span_ns": 0,
            "merged_interval_count": 0,
        }
    clusters = []
    cluster_start = events[0].start_ns
    cluster_end = events[0].end_ns
    cluster_last = events[0]
    for event in events[1:]:
        if event.start_ns <= cluster_end:
            if event.end_ns > cluster_end:
                cluster_end = event.end_ns
                cluster_last = event
            continue
        clusters.append((cluster_start, cluster_end, cluster_last, event))
        cluster_start = event.start_ns
        cluster_end = event.end_ns
        cluster_last = event
    clusters.append((cluster_start, cluster_end, cluster_last, None))

    gaps = []
    for _, previous_end, previous, next_event in clusters[:-1]:
        if next_event is None or next_event.start_ns <= previous_end:
            continue
        gaps.append(
            GpuGap(
                previous_end_ns=previous_end,
                next_commit_ns=next_event.commit_ns,
                next_start_ns=next_event.start_ns,
                previous_frame=previous.frame_number,
                next_frame=next_event.frame_number,
                previous_stage=stage_of(previous),
                next_stage=stage_of(next_event),
                commit_source=next_event.commit_source,
            )
        )
    merged_intervals = [(start, end) for start, end, _, _ in clusters]
    busy = sum(end - start for start, end in merged_intervals)
    span = merged_intervals[-1][1] - merged_intervals[0][0]
    idle = span - busy
    return gaps, {
        "gpu_busy_ns": busy,
        "gpu_idle_ns": idle,
        "gpu_span_ns": span,
        "merged_interval_count": len(merged_intervals),
        "reconstructed_gap_ns": sum(gap.total_ns for gap in gaps),
    }


def _bin_name(value: int) -> str:
    if value < GAP_BINS_NS[0]:
        return "lt_25us"
    if value < GAP_BINS_NS[1]:
        return "25_to_100us"
    if value < GAP_BINS_NS[2]:
        return "100_to_500us"
    return "ge_500us"


def _gap_summary(gaps: list[GpuGap]) -> dict:
    bins = {
        name: {"count": 0, "total_ns": 0}
        for name in ("lt_25us", "25_to_100us", "100_to_500us", "ge_500us")
    }
    for gap in gaps:
        row = bins[_bin_name(gap.total_ns)]
        row["count"] += 1
        row["total_ns"] += gap.total_ns
    top = sorted(gaps, key=lambda gap: gap.total_ns, reverse=True)[:64]
    return {
        "gap_count": len(gaps),
        "application_starvation_ns": sum(
            gap.application_starvation_ns for gap in gaps
        ),
        "driver_or_dependency_ns": sum(
            gap.driver_or_dependency_ns for gap in gaps
        ),
        "bins": bins,
        "top_gaps": [
            asdict(gap)
            | {
                "total_ns": gap.total_ns,
                "application_starvation_ns": gap.application_starvation_ns,
                "driver_or_dependency_ns": gap.driver_or_dependency_ns,
            }
            for gap in top
        ],
    }


def _classify_gap(gap: GpuGap) -> tuple[str, GpuGap]:
    frame_delta = None
    if gap.previous_frame is not None and gap.next_frame is not None:
        frame_delta = gap.next_frame - gap.previous_frame
    if (
        gap.total_ns >= 500_000
        and gap.application_starvation_ns >= 500_000
        and frame_delta == 2
    ):
        return "readback_to_next_token", replace(
            gap,
            previous_stage="argmax/eval/readback",
            next_stage="next-token graph submission",
        )
    capture_edge = (
        gap.next_frame is not None
        and gap.next_frame <= 2
        and (gap.previous_frame is None or gap.previous_frame <= 1)
    )
    if (
        gap.total_ns >= 100_000
        and gap.application_starvation_ns == 0
        and capture_edge
    ):
        return "capture_startup_or_queue_fill", replace(
            gap,
            previous_stage="capture attach / initial queue fill",
            next_stage="first-token GPU work",
        )
    return "within_token_unclassified", replace(
        gap,
        previous_stage="unclassified within-token dynamic event",
        next_stage="unclassified within-token dynamic event",
    )


def _boundary_summary(gaps: list[GpuGap]) -> dict:
    grouped = {}
    for gap in gaps:
        boundary, classified = _classify_gap(gap)
        grouped.setdefault(boundary, []).append(classified)
    result = {}
    for boundary, rows in grouped.items():
        total = sum(row.total_ns for row in rows)
        # Keep the committed artifact aggregate-only.  The 15 periodic token
        # boundaries are all useful evidence; for the thousands of tiny
        # within-token gaps, the distribution plus the 16 largest examples is
        # sufficient and avoids committing a multi-megabyte raw-event dump.
        evidence_limit = len(rows) if boundary == "readback_to_next_token" else 16
        evidence_rows = sorted(rows, key=lambda row: row.total_ns, reverse=True)[
            :evidence_limit
        ]
        result[boundary] = {
            "count": len(rows),
            "total_ns": total,
            "total_ms_per_token": total / TRACED_TOKENS / 1e6,
            "application_starvation_ns": sum(
                row.application_starvation_ns for row in rows
            ),
            "application_starvation_ms_per_token": sum(
                row.application_starvation_ns for row in rows
            )
            / TRACED_TOKENS
            / 1e6,
            "driver_or_dependency_ns": sum(
                row.driver_or_dependency_ns for row in rows
            ),
            "driver_or_dependency_ms_per_token": sum(
                row.driver_or_dependency_ns for row in rows
            )
            / TRACED_TOKENS
            / 1e6,
            "idle_fraction": None,
            "gap_bins": _gap_summary(rows)["bins"],
            "dynamic_evidence_count": len(evidence_rows),
            "dynamic_evidence": [
                {
                    "previous_end_ns": row.previous_end_ns,
                    "next_commit_ns": row.next_commit_ns,
                    "next_start_ns": row.next_start_ns,
                    "previous_frame": row.previous_frame,
                    "next_frame": row.next_frame,
                    "total_ns": row.total_ns,
                    "application_starvation_ns": row.application_starvation_ns,
                    "driver_or_dependency_ns": row.driver_or_dependency_ns,
                }
                for row in evidence_rows
            ],
        }
    idle = sum(row["total_ns"] for row in result.values())
    for row in result.values():
        row["idle_fraction"] = row["total_ns"] / idle if idle else 0.0
    return result


def _requested_boundary_table(boundaries: dict) -> dict:
    unavailable = {
        "measured": False,
        "reason": (
            "the bounded System Trace has dynamic command-buffer/frame IDs but "
            "no shader-dispatch intervals; this boundary remains inside the "
            "explicit within-token unclassified budget"
        ),
    }
    return {
        "layer_to_kda": dict(unavailable),
        "layer_to_dsa": dict(unavailable),
        "attention_to_ffn": dict(unavailable),
        "ffn_to_next_layer": dict(unavailable),
        "final_norm_to_lm_head": dict(unavailable),
        "lm_head_to_argmax": dict(unavailable),
        "readback_to_next_token": {
            "measured": True,
            **boundaries["readback_to_next_token"],
        },
        "materialization": {
            "measured": False,
            "reason": "the bounded trace covers steps 1-16; step 256 is outside it",
        },
        "within_token_unclassified": {
            "measured": True,
            **boundaries.get("within_token_unclassified", {}),
        },
        "capture_startup_or_queue_fill": {
            "measured": True,
            **boundaries.get("capture_startup_or_queue_fill", {}),
        },
    }


def _arm(trace: Path, pid: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="glm53-idle-attribution-") as temporary:
        exports = _export(trace, Path(temporary))
        events, coverage = _events(exports, pid)
    gaps, reconstruction = reconstruct_gaps(events)
    boundaries = _boundary_summary(gaps)
    return {
        "trace": _trace_identity(trace),
        "pid": pid,
        "coverage": coverage,
        "reconstruction": reconstruction,
        "unclassified_gap_summary": _gap_summary(gaps),
        "dynamic_boundaries": boundaries,
        "requested_boundary_table": _requested_boundary_table(boundaries),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finalize(artifact: dict, source: dict) -> dict:
    gates = {}
    for arm in ("A", "B"):
        row = artifact["arms"][arm]
        reconstructed = row["reconstruction"]["gpu_idle_ns"]
        recorded = round(source["arms"][arm]["telemetry"]["gpu_idle_gap_ms"] * 1e6)
        relative_error = abs(reconstructed - recorded) / recorded if recorded else 0.0
        unclassified = row["dynamic_boundaries"]["within_token_unclassified"][
            "total_ns"
        ]
        unclassified_fraction = unclassified / reconstructed if reconstructed else 0.0
        row["recorded_gpu_idle_ns"] = recorded
        row["idle_reconstruction_relative_error"] = relative_error
        row["within_token_unclassified_idle_fraction"] = unclassified_fraction
        gates[f"{arm}_idle_reconstruction_within_5pct"] = relative_error <= 0.05
        gates[f"{arm}_unclassified_idle_le_10pct"] = unclassified_fraction <= 0.10
        gates[f"{arm}_token_boundary_count_15"] = (
            row["dynamic_boundaries"]["readback_to_next_token"]["count"]
            == TRACED_TOKENS - 1
        )

    a = artifact["arms"]["A"]["dynamic_boundaries"]["readback_to_next_token"]
    b = artifact["arms"]["B"]["dynamic_boundaries"]["readback_to_next_token"]
    total_delta = (
        artifact["arms"]["A"]["reconstruction"]["gpu_idle_ns"]
        - artifact["arms"]["B"]["reconstruction"]["gpu_idle_ns"]
    )
    boundary_delta = a["total_ns"] - b["total_ns"]
    artifact["compiled_vs_eager"] = {
        "total_idle_reduction_ns": total_delta,
        "total_idle_reduction_ms_per_token": total_delta
        / TRACED_TOKENS
        / 1e6,
        "readback_boundary_reduction_ns": boundary_delta,
        "readback_boundary_reduction_ms_per_token": boundary_delta
        / TRACED_TOKENS
        / 1e6,
        "readback_boundary_attribution_fraction": boundary_delta / total_delta
        if total_delta
        else None,
        "application_starvation_reduction_ns": (
            a["application_starvation_ns"] - b["application_starvation_ns"]
        ),
        "driver_or_dependency_reduction_ns": (
            a["driver_or_dependency_ns"] - b["driver_or_dependency_ns"]
        ),
        "interpretation": (
            "compiled FFN changes when the next token graph reaches Metal; "
            "the arithmetic GPU busy path is unchanged"
        ),
    }
    gates["compiled_idle_delta_attributed_ge_80pct"] = (
        boundary_delta / total_delta >= 0.80 if total_delta > 0 else False
    )
    gates["static_labels_used_for_stage_assignment"] = False
    artifact["acceptance"] = gates
    artifact["complete"] = all(
        value for key, value in gates.items() if key != "static_labels_used_for_stage_assignment"
    ) and not gates["static_labels_used_for_stage_assignment"]
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--explore", action="store_true")
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    artifact = {
        "schema": "glm53-steady-decode-dynamic-idle-attribution-v1",
        "complete": False,
        "source_artifact": str(args.source),
        "traced_tokens": TRACED_TOKENS,
        "arms": {},
        "operator_microcapture_artifact": str(OPERATOR_MICROCAPTURE),
        "operator_microcapture_artifact_sha256": _sha256(OPERATOR_MICROCAPTURE),
        "stage_assignment_contract": {
            "dynamic_command_buffer_and_frame_ids_required": True,
            "static_kernel_labels_used": False,
            "microcapture_role": (
                "operator-stage vocabulary and kernel inventory only; no stage "
                "is assigned without a dynamic System Trace boundary"
            ),
        },
    }
    for arm in ("A", "B"):
        row = source["arms"][arm]
        artifact["arms"][arm] = _arm(Path(row["trace"]["path"]), int(row["pid"]))
    artifact = _finalize(artifact, source)
    if args.explore:
        print(json.dumps(artifact, indent=2))
        return 0
    atomic_write(args.output, artifact)
    print(json.dumps({"complete": artifact["complete"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
