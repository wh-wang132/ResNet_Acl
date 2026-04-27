from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from src.common.report import ensure_directory, to_repo_relative, write_json


PAPER_TOP_FIELDS = [
    "rank",
    "branch",
    "branch_display",
    "model_name",
    "experiment_name",
    "ratio",
    "steps",
    "accuracy",
    "error_rate",
    "end_to_end_throughput_samples_per_sec",
    "end_to_end_avg_latency_ms",
    "operation_count_gmacs",
    "parameter_count",
]
PAPER_PARETO_FIELDS = [
    "pareto_front",
    "branch",
    "branch_display",
    "model_name",
    "experiment_name",
    "ratio",
    "steps",
    "accuracy",
    "error_rate",
    "end_to_end_throughput_samples_per_sec",
    "end_to_end_avg_latency_ms",
    "operation_count_gmacs",
    "parameter_count",
]
PAPER_BRANCH_PAIR_SUMMARY_FIELDS = [
    "group",
    "model_name",
    "pairs",
    "accuracy_delta_pp_median",
    "accuracy_delta_pp_min",
    "accuracy_delta_pp_max",
    "throughput_ratio_median",
    "throughput_ratio_min",
    "throughput_ratio_max",
    "latency_ratio_median",
    "latency_ratio_min",
    "latency_ratio_max",
    "fp16_throughput_median",
    "int8_throughput_median",
    "fp16_latency_median",
    "int8_latency_median",
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
    paper_top_candidates: list[dict[str, object]],
    paper_pair_summary: list[dict[str, object]],
) -> Path:
    payload = {
        "filters": filters,
        "records_count": records_count,
        "missing_count": missing_count,
        "paper_tables": {name: to_repo_relative(path) for name, path in sorted(table_paths.items())},
        "paper_figures": {name: to_repo_relative(path) for name, path in sorted(plot_paths.items())},
        "paper_top_candidates": _compact_paper_candidates(paper_top_candidates[:10]),
        "paper_branch_pair_summary": paper_pair_summary,
    }
    return write_json(file_path, payload)


def write_paper_markdown_summary(
    file_path: str | Path,
    *,
    records_count: int,
    missing_count: int,
    pareto_rows: list[dict[str, object]],
    paper_top_candidates: list[dict[str, object]],
    paper_pair_summary: list[dict[str, object]],
    plot_paths: dict[str, Path],
) -> Path:
    resolved = Path(file_path)
    ensure_directory(resolved.parent)
    best = paper_top_candidates[0] if paper_top_candidates else None
    overall = next((row for row in paper_pair_summary if row.get("group") == "overall"), {})
    pareto_counts: dict[str, int] = {}
    for row in pareto_rows:
        key = str(row.get("pareto_front", "unknown"))
        pareto_counts[key] = pareto_counts.get(key, 0) + 1

    lines = [
        "# ResNet_Acl Paper Figures Summary",
        "",
        f"- Joined experiment records: `{records_count}`",
        f"- Missing input records: `{missing_count}`",
        f"- Pareto candidates: `{sum(pareto_counts.values())}` "
        f"({', '.join(f'{key}={value}' for key, value in sorted(pareto_counts.items())) or 'none'})",
        f"- Generated paper figures: `{len(plot_paths)}`",
    ]
    if best is not None:
        lines.append(
            "- Best paper candidate: "
            f"`{best['branch_display']} / {best['model_name']} / {best['experiment_name']}` "
            f"(accuracy={_fmt(best.get('accuracy'))}, "
            f"error_rate={_fmt(best.get('error_rate'))}, "
            f"throughput={_fmt(best.get('end_to_end_throughput_samples_per_sec'))} samples/s, "
            f"latency={_fmt(best.get('end_to_end_avg_latency_ms'))} ms)"
        )
    if overall:
        lines.extend(
            [
                "- INT8 AMCT vs FP16 pruning paired median throughput ratio: "
                f"`{_fmt(overall.get('throughput_ratio_median'))}`.",
                "- INT8 AMCT vs FP16 pruning paired median latency ratio: "
                f"`{_fmt(overall.get('latency_ratio_median'))}`.",
                "- INT8 AMCT vs FP16 pruning paired median accuracy delta: "
                f"`{_fmt(overall.get('accuracy_delta_pp_median'))}` percentage points.",
            ]
        )
    lines.extend(
        [
            "",
            "Interpretation: the current paper figures intentionally present FP16 pruning as the dominant "
            "accuracy-efficiency Pareto branch under this measurement setup. INT8 AMCT keeps similar "
            "accuracy in many pairs, but its measured throughput is lower and latency is higher overall.",
            "",
        ]
    )
    resolved.write_text("\n".join(lines), encoding="utf-8")
    return resolved


def _compact_paper_candidates(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = [
        "rank",
        "branch_display",
        "model_name",
        "experiment_name",
        "accuracy",
        "error_rate",
        "end_to_end_throughput_samples_per_sec",
        "end_to_end_avg_latency_ms",
        "operation_count_gmacs",
    ]
    return [{key: row.get(key, "") for key in keys} for row in rows]


def _fmt(value: object) -> str:
    if value in (None, ""):
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)
