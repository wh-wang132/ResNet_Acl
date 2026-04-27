from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common.branch_specs import BRANCH_SPECS
from src.common.report import ensure_directory, resolve_repo_path, to_repo_relative

from .plots import (
    plot_branch_pair_slope,
    plot_error_rate_latency_pareto,
    plot_error_rate_throughput_pareto,
    plot_macs_latency_scatter,
    plot_params_throughput_scatter,
    plot_per_class_metrics_heatmap,
    plot_topk_candidates,
)
from .report import (
    ARTIFACT_METRICS_FIELDS,
    BRANCH_PAIR_FIELDS,
    MISSING_INPUT_FIELDS,
    PARETO_FIELDS,
    TOPK_FIELDS,
    write_index,
    write_markdown_summary,
    write_table,
)
from .sources import VisualizationError, load_visualization_records
from .tables import (
    build_artifact_metrics,
    build_branch_pairs,
    build_missing_inputs,
    build_pareto_candidates,
    build_topk_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ResNet_Acl 离线评测结果可视化")
    parser.add_argument("--accuracy_root", type=Path, default=Path("output/accuracy"))
    parser.add_argument("--efficiency_root", type=Path, default=Path("output/efficiency"))
    parser.add_argument("--efficiency_pattern", default="summary__instances*.json")
    parser.add_argument("--output_root", type=Path, default=Path("output/visualization"))
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--branch", choices=sorted(BRANCH_SPECS), default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--num_instances", type=int, default=1)
    parser.add_argument("--buffer_depth", type=int, default=1)
    parser.add_argument(
        "--plot_set",
        choices=["overview", "accuracy", "efficiency", "pareto", "branch_compare", "class_metrics", "all"],
        default="all",
    )
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--accuracy_floor", type=float, default=0.99)
    parser.add_argument("--format", choices=["png", "svg"], default="png")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    try:
        args = parse_args()
        run_visualization(args)
    except (VisualizationError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc


def run_visualization(args: argparse.Namespace) -> Path:
    if args.top_k <= 0:
        raise VisualizationError("--top_k 必须为正整数")
    if args.dpi <= 0:
        raise VisualizationError("--dpi 必须为正整数")
    if args.num_instances <= 0:
        raise VisualizationError("--num_instances 必须为正整数")
    if args.buffer_depth <= 0:
        raise VisualizationError("--buffer_depth 必须为正整数")

    records, missing_inputs = load_visualization_records(
        accuracy_root=args.accuracy_root,
        efficiency_root=args.efficiency_root,
        efficiency_pattern=args.efficiency_pattern,
        branch=args.branch,
        model_name=args.model_name,
        experiment_name=args.experiment_name,
        num_instances=args.num_instances,
        buffer_depth=args.buffer_depth,
    )
    if args.strict and missing_inputs:
        raise VisualizationError(f"发现 {len(missing_inputs)} 个缺失输入，--strict 模式中止")
    if not records:
        raise VisualizationError("没有找到可合并的 accuracy + efficiency 记录")

    run_name = args.run_name or args.branch or "all"
    output_root = resolve_repo_path(args.output_root)
    run_dir = ensure_directory(output_root / run_name)
    tables_dir = ensure_directory(run_dir / "tables")
    plots_dir = ensure_directory(run_dir / "plots")

    artifact_rows = build_artifact_metrics(records)
    missing_rows = build_missing_inputs(missing_inputs)
    pareto_rows = build_pareto_candidates(records)
    topk_rows = build_topk_candidates(records, top_k=args.top_k, accuracy_floor=args.accuracy_floor)
    branch_pair_rows = build_branch_pairs(records)

    table_paths = {
        "artifact_metrics": write_table(tables_dir / "artifact_metrics.csv", artifact_rows, ARTIFACT_METRICS_FIELDS),
        "missing_inputs": write_table(tables_dir / "missing_inputs.csv", missing_rows, MISSING_INPUT_FIELDS),
        "pareto_candidates": write_table(tables_dir / "pareto_candidates.csv", pareto_rows, PARETO_FIELDS),
        "topk_candidates": write_table(tables_dir / "topk_candidates.csv", topk_rows, TOPK_FIELDS),
        "branch_pairs": write_table(tables_dir / "branch_pairs.csv", branch_pair_rows, BRANCH_PAIR_FIELDS),
    }
    plot_paths = _write_plots(
        args,
        records=records,
        artifact_rows=artifact_rows,
        topk_rows=topk_rows,
        branch_pair_rows=branch_pair_rows,
        plots_dir=plots_dir,
    )

    filters = {
        "accuracy_root": to_repo_relative(resolve_repo_path(args.accuracy_root)),
        "efficiency_root": to_repo_relative(resolve_repo_path(args.efficiency_root)),
        "efficiency_pattern": args.efficiency_pattern,
        "branch": args.branch,
        "model_name": args.model_name,
        "experiment_name": args.experiment_name,
        "num_instances": args.num_instances,
        "buffer_depth": args.buffer_depth,
        "accuracy_floor": args.accuracy_floor,
    }
    index_path = write_index(
        run_dir / "index.json",
        filters=filters,
        records_count=len(records),
        missing_count=len(missing_inputs),
        table_paths=table_paths,
        plot_paths=plot_paths,
        top_candidates=topk_rows,
        branch_pairs=branch_pair_rows,
    )
    write_markdown_summary(
        run_dir / "summary.md",
        records_count=len(records),
        missing_count=len(missing_inputs),
        top_candidates=topk_rows,
        branch_pairs=branch_pair_rows,
    )
    print(to_repo_relative(index_path))
    return index_path


def _write_plots(
    args: argparse.Namespace,
    *,
    records,
    artifact_rows: list[dict[str, object]],
    topk_rows: list[dict[str, object]],
    branch_pair_rows: list[dict[str, object]],
    plots_dir: Path,
) -> dict[str, Path]:
    selected = _selected_plot_names(args.plot_set)
    plot_paths: dict[str, Path] = {}

    if "error_rate_throughput_pareto" in selected:
        plot_paths["error_rate_throughput_pareto"] = plot_error_rate_throughput_pareto(
            artifact_rows,
            plots_dir / f"error_rate_throughput_pareto.{args.format}",
            dpi=args.dpi,
        )
    if "error_rate_latency_pareto" in selected:
        plot_paths["error_rate_latency_pareto"] = plot_error_rate_latency_pareto(
            artifact_rows,
            plots_dir / f"error_rate_latency_pareto.{args.format}",
            dpi=args.dpi,
        )
    if "macs_latency_scatter" in selected:
        plot_paths["macs_latency_scatter"] = plot_macs_latency_scatter(
            artifact_rows,
            plots_dir / f"macs_latency_scatter.{args.format}",
            dpi=args.dpi,
        )
    if "params_throughput_scatter" in selected:
        plot_paths["params_throughput_scatter"] = plot_params_throughput_scatter(
            artifact_rows,
            plots_dir / f"params_throughput_scatter.{args.format}",
            dpi=args.dpi,
        )
    if "branch_pair_slope" in selected and branch_pair_rows:
        plot_paths["branch_pair_slope_error_throughput_latency"] = plot_branch_pair_slope(
            branch_pair_rows,
            plots_dir / f"branch_pair_slope_error_throughput_latency.{args.format}",
            dpi=args.dpi,
        )
    if "topk_candidates" in selected and topk_rows:
        plot_paths["topk_candidates"] = plot_topk_candidates(
            topk_rows,
            plots_dir / f"topk_candidates.{args.format}",
            dpi=args.dpi,
        )
    if "per_class_metrics" in selected and topk_rows:
        best = _find_record_for_row(records, topk_rows[0])
        if best is not None:
            path = plot_per_class_metrics_heatmap(
                best,
                plots_dir / f"per_class_metrics_top_candidate.{args.format}",
                dpi=args.dpi,
            )
            if path is not None:
                plot_paths["per_class_metrics_top_candidate"] = path
    return plot_paths


def _selected_plot_names(plot_set: str) -> set[str]:
    if plot_set == "all":
        return {
            "error_rate_throughput_pareto",
            "error_rate_latency_pareto",
            "macs_latency_scatter",
            "params_throughput_scatter",
            "branch_pair_slope",
            "topk_candidates",
            "per_class_metrics",
        }
    if plot_set == "overview":
        return {
            "error_rate_throughput_pareto",
            "error_rate_latency_pareto",
            "macs_latency_scatter",
            "topk_candidates",
        }
    if plot_set == "accuracy":
        return {"error_rate_throughput_pareto", "error_rate_latency_pareto", "topk_candidates", "per_class_metrics"}
    if plot_set == "efficiency":
        return {"macs_latency_scatter", "params_throughput_scatter", "topk_candidates"}
    if plot_set == "pareto":
        return {"error_rate_throughput_pareto", "error_rate_latency_pareto", "topk_candidates"}
    if plot_set == "branch_compare":
        return {"branch_pair_slope"}
    if plot_set == "class_metrics":
        return {"per_class_metrics"}
    raise VisualizationError(f"不支持的 plot_set: {plot_set}")


def _find_record_for_row(records, row: dict[str, object]):
    for record in records:
        if (
            record.key.branch == row.get("branch")
            and record.key.model_name == row.get("model_name")
            and record.key.experiment_name == row.get("experiment_name")
        ):
            return record
    return None


if __name__ == "__main__":
    main()
