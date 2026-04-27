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
    plot_paper_branch_pair_summary,
    plot_paper_complexity_tradeoff,
    plot_paper_error_rate_latency_pareto,
    plot_paper_error_rate_throughput_pareto,
    plot_paper_per_class_pair_delta,
    plot_per_class_metrics_heatmap,
    plot_topk_candidates,
)
from .report import (
    ARTIFACT_METRICS_FIELDS,
    BRANCH_PAIR_FIELDS,
    MISSING_INPUT_FIELDS,
    PAPER_BRANCH_PAIR_SUMMARY_FIELDS,
    PAPER_TOP_FIELDS,
    PARETO_FIELDS,
    TOPK_FIELDS,
    write_index,
    write_markdown_summary,
    write_paper_markdown_summary,
    write_table,
)
from .sources import VisualizationError, load_visualization_records
from .tables import (
    build_artifact_metrics,
    build_branch_pairs,
    build_missing_inputs,
    build_paper_branch_pair_summary,
    build_paper_top_candidates,
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
        choices=["paper", "overview", "accuracy", "efficiency", "pareto", "branch_compare", "class_metrics", "all"],
        default="paper",
    )
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--accuracy_floor", type=float, default=0.99)
    parser.add_argument("--format", choices=["png", "svg"], default="svg")
    parser.add_argument("--dpi", type=int, default=300)
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

    run_name = args.run_name or ("paper" if args.plot_set == "paper" else args.branch or "all")
    output_root = resolve_repo_path(args.output_root)
    run_dir = ensure_directory(output_root / run_name)
    tables_dir = ensure_directory(run_dir / "tables")
    plots_dir = ensure_directory(run_dir / "plots")

    artifact_rows = build_artifact_metrics(records)
    missing_rows = build_missing_inputs(missing_inputs)
    pareto_rows = build_pareto_candidates(records)
    topk_rows = build_topk_candidates(records, top_k=args.top_k, accuracy_floor=args.accuracy_floor)
    branch_pair_rows = build_branch_pairs(records)
    paper_top_rows = build_paper_top_candidates(topk_rows)
    paper_pair_summary_rows = build_paper_branch_pair_summary(branch_pair_rows)

    table_paths = {
        "artifact_metrics": write_table(tables_dir / "artifact_metrics.csv", artifact_rows, ARTIFACT_METRICS_FIELDS),
        "missing_inputs": write_table(tables_dir / "missing_inputs.csv", missing_rows, MISSING_INPUT_FIELDS),
        "pareto_candidates": write_table(tables_dir / "pareto_candidates.csv", pareto_rows, PARETO_FIELDS),
        "topk_candidates": write_table(tables_dir / "topk_candidates.csv", topk_rows, TOPK_FIELDS),
        "branch_pairs": write_table(tables_dir / "branch_pairs.csv", branch_pair_rows, BRANCH_PAIR_FIELDS),
    }
    if args.plot_set == "paper":
        table_paths.update(
            {
                "paper_top5_candidates": write_table(
                    tables_dir / "paper_top5_candidates.csv",
                    paper_top_rows,
                    PAPER_TOP_FIELDS,
                ),
                "paper_branch_pair_summary": write_table(
                    tables_dir / "paper_branch_pair_summary.csv",
                    paper_pair_summary_rows,
                    PAPER_BRANCH_PAIR_SUMMARY_FIELDS,
                ),
            }
        )
        plot_paths = _write_paper_plots(
            args,
            records=records,
            artifact_rows=artifact_rows,
            pareto_rows=pareto_rows,
            topk_rows=topk_rows,
            branch_pair_rows=branch_pair_rows,
            plots_dir=plots_dir,
        )
    else:
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
        "plot_set": args.plot_set,
        "format": args.format,
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
    if args.plot_set == "paper":
        write_paper_markdown_summary(
            run_dir / "paper_summary.md",
            records_count=len(records),
            missing_count=len(missing_inputs),
            pareto_rows=pareto_rows,
            paper_top_candidates=paper_top_rows,
            paper_pair_summary=paper_pair_summary_rows,
            plot_paths=plot_paths,
        )
    else:
        write_markdown_summary(
            run_dir / "summary.md",
            records_count=len(records),
            missing_count=len(missing_inputs),
            top_candidates=topk_rows,
            branch_pairs=branch_pair_rows,
        )
    print(to_repo_relative(index_path))
    return index_path


def _write_paper_plots(
    args: argparse.Namespace,
    *,
    records,
    artifact_rows: list[dict[str, object]],
    pareto_rows: list[dict[str, object]],
    topk_rows: list[dict[str, object]],
    branch_pair_rows: list[dict[str, object]],
    plots_dir: Path,
) -> dict[str, Path]:
    plot_paths: dict[str, Path] = {}
    plot_paths["fig1_pareto_error_throughput"] = plot_paper_error_rate_throughput_pareto(
        artifact_rows,
        pareto_rows,
        topk_rows,
        plots_dir / f"fig1_pareto_error_throughput.{args.format}",
        dpi=args.dpi,
    )
    plot_paths["fig2_pareto_error_latency"] = plot_paper_error_rate_latency_pareto(
        artifact_rows,
        pareto_rows,
        topk_rows,
        plots_dir / f"fig2_pareto_error_latency.{args.format}",
        dpi=args.dpi,
    )
    if branch_pair_rows:
        plot_paths["fig3_fp16_int8_pair_summary"] = plot_paper_branch_pair_summary(
            branch_pair_rows,
            plots_dir / f"fig3_fp16_int8_pair_summary.{args.format}",
            dpi=args.dpi,
        )
    plot_paths["fig4_complexity_tradeoff"] = plot_paper_complexity_tradeoff(
        artifact_rows,
        plots_dir / f"fig4_complexity_tradeoff.{args.format}",
        dpi=args.dpi,
    )

    pair = _find_best_fp16_int8_pair(records, topk_rows[0] if topk_rows else None)
    if pair is not None:
        fp16_record, int8_record = pair
        path = plot_paper_per_class_pair_delta(
            fp16_record,
            int8_record,
            plots_dir / f"fig5_per_class_recall_delta.{args.format}",
            dpi=args.dpi,
        )
        if path is not None:
            plot_paths["fig5_per_class_recall_delta"] = path
    return plot_paths


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
    if plot_set == "paper":
        return set()
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


def _find_best_fp16_int8_pair(records, top_row: dict[str, object] | None):
    if top_row is None:
        return None
    model_name = top_row.get("model_name")
    experiment_name = top_row.get("experiment_name")
    fp16_record = None
    int8_record = None
    for record in records:
        if record.key.model_name != model_name or record.key.experiment_name != experiment_name:
            continue
        if record.key.branch == "pruning_fp16":
            fp16_record = record
        elif record.key.branch == "amct_deploy":
            int8_record = record
    if fp16_record is None or int8_record is None:
        return None
    return fp16_record, int8_record


if __name__ == "__main__":
    main()
