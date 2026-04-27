from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Iterable

from src.common.report import ensure_directory, to_repo_relative, write_json


ARTIFACT_METRICS_FIELDS = [
    "branch",
    "model_name",
    "experiment_name",
    "ratio",
    "steps",
    "pruning_mode",
    "finetune_epochs",
    "batch_size",
    "samples",
    "accuracy",
    "error_rate",
    "error_rate_percent",
    "avg_loss",
    "parameter_count",
    "operation_count",
    "operation_count_gmacs",
    "om_size_bytes",
    "om_size_mib",
    "num_instances",
    "active_instances",
    "buffer_depth",
    "end_to_end_avg_latency_ms",
    "end_to_end_p50_latency_ms",
    "end_to_end_p95_latency_ms",
    "end_to_end_p99_latency_ms",
    "end_to_end_throughput_samples_per_sec",
    "pure_infer_avg_latency_ms",
    "pure_infer_throughput_samples_per_sec",
    "h2d_memcpy_total_ms",
    "execute_wait_total_ms",
    "d2h_memcpy_total_ms",
    "output_decode_total_ms",
    "artifact_path",
    "accuracy_summary_path",
    "efficiency_summary_path",
    "confusion_matrix_csv",
    "confusion_matrix_png",
    "per_class_metrics_csv",
]
MISSING_INPUT_FIELDS = [
    "branch",
    "model_name",
    "experiment_name",
    "missing_accuracy",
    "missing_efficiency",
    "accuracy_summary_path",
    "efficiency_summary_path",
]
PARETO_FIELDS = ARTIFACT_METRICS_FIELDS + ["pareto_front"]
TOPK_FIELDS = ARTIFACT_METRICS_FIELDS + ["rank", "accuracy_floor"]
BRANCH_PAIR_FIELDS = [
    "model_name",
    "experiment_name",
    "ratio",
    "steps",
    "pruning_accuracy",
    "amct_accuracy",
    "accuracy_delta_amct_minus_pruning",
    "pruning_error_rate",
    "amct_error_rate",
    "error_rate_delta_amct_minus_pruning",
    "pruning_throughput",
    "amct_throughput",
    "throughput_ratio_amct_over_pruning",
    "pruning_latency",
    "amct_latency",
    "latency_ratio_amct_over_pruning",
]


def write_table(file_path: str | Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> Path:
    resolved = Path(file_path)
    ensure_directory(resolved.parent)
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return resolved


def write_index(
    file_path: str | Path,
    *,
    filters: dict[str, object],
    records_count: int,
    missing_count: int,
    table_paths: dict[str, Path],
    plot_paths: dict[str, Path],
    top_candidates: list[dict[str, object]],
    branch_pairs: list[dict[str, object]],
) -> Path:
    payload = {
        "filters": filters,
        "records_count": records_count,
        "missing_count": missing_count,
        "tables": {name: to_repo_relative(path) for name, path in sorted(table_paths.items())},
        "plots": {name: to_repo_relative(path) for name, path in sorted(plot_paths.items())},
        "top_candidates": _compact_candidates(top_candidates[:10]),
        "branch_pair_summary": _branch_pair_summary(branch_pairs),
    }
    return write_json(file_path, payload)


def write_markdown_summary(
    file_path: str | Path,
    *,
    records_count: int,
    missing_count: int,
    top_candidates: list[dict[str, object]],
    branch_pairs: list[dict[str, object]],
) -> Path:
    resolved = Path(file_path)
    ensure_directory(resolved.parent)
    best = top_candidates[0] if top_candidates else None
    pair_summary = _branch_pair_summary(branch_pairs)

    lines = [
        "# ResNet_Acl Visualization Summary",
        "",
        f"- Joined experiment records: `{records_count}`",
        f"- Missing input records: `{missing_count}`",
    ]
    if best is not None:
        lines.extend(
            [
                "- Best throughput candidate under the accuracy floor: "
                f"`{best['branch']} / {best['model_name']} / {best['experiment_name']}` "
                f"(accuracy={_fmt(best.get('accuracy'))}, "
                f"error_rate={_fmt(best.get('error_rate'))}, "
                f"throughput={_fmt(best.get('end_to_end_throughput_samples_per_sec'))} samples/s, "
                f"latency={_fmt(best.get('end_to_end_avg_latency_ms'))} ms)",
            ]
        )
    if pair_summary:
        lines.extend(
            [
                "- Paired branch median throughput ratio "
                f"`amct/pruning={_fmt(pair_summary.get('median_throughput_ratio_amct_over_pruning'))}`.",
                "- Paired branch median latency ratio "
                f"`amct/pruning={_fmt(pair_summary.get('median_latency_ratio_amct_over_pruning'))}`.",
            ]
        )
    lines.extend(
        [
            "",
            "Interpretation: use the Pareto plots as the main evidence for the deployment trade-off. "
            "Treat `num_instances` and `buffer_depth` as fixed filters for this run.",
            "",
        ]
    )
    resolved.write_text("\n".join(lines), encoding="utf-8")
    return resolved


def _compact_candidates(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = [
        "rank",
        "branch",
        "model_name",
        "experiment_name",
        "accuracy",
        "error_rate",
        "end_to_end_throughput_samples_per_sec",
        "end_to_end_avg_latency_ms",
        "operation_count_gmacs",
    ]
    return [{key: row.get(key, "") for key in keys} for row in rows]


def _branch_pair_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {}
    throughput_ratios = _numeric_values(row.get("throughput_ratio_amct_over_pruning") for row in rows)
    latency_ratios = _numeric_values(row.get("latency_ratio_amct_over_pruning") for row in rows)
    accuracy_deltas = _numeric_values(row.get("accuracy_delta_amct_minus_pruning") for row in rows)
    error_rate_deltas = _numeric_values(row.get("error_rate_delta_amct_minus_pruning") for row in rows)
    return {
        "paired_records": len(rows),
        "median_throughput_ratio_amct_over_pruning": _median_or_empty(throughput_ratios),
        "median_latency_ratio_amct_over_pruning": _median_or_empty(latency_ratios),
        "median_accuracy_delta_amct_minus_pruning": _median_or_empty(accuracy_deltas),
        "median_error_rate_delta_amct_minus_pruning": _median_or_empty(error_rate_deltas),
    }


def _numeric_values(values: Iterable[object]) -> list[float]:
    result: list[float] = []
    for value in values:
        if value in (None, ""):
            continue
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            continue
    return result


def _median_or_empty(values: list[float]) -> float | str:
    return statistics.median(values) if values else ""


def _fmt(value: object) -> str:
    if value in (None, ""):
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)
