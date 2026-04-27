from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

from src.common.report import ensure_directory

from .schema import VisualizationRecord


BRANCH_COLORS = {
    "pruning_fp16": "#1f77b4",
    "amct_deploy": "#ff7f0e",
}
BRANCH_DISPLAY_NAMES = {
    "pruning_fp16": "FP16 pruning",
    "amct_deploy": "INT8 AMCT",
}
BRANCH_MARKERS = {
    "pruning_fp16": "o",
    "amct_deploy": "s",
}


LOG_SAFE_ERROR_RATE_FLOOR = 1e-12
LOG_SAFE_COMPLEXITY_FLOOR = 1e-12
LOG_SAFE_LATENCY_FLOOR = 1e-12


def plot_paper_error_rate_throughput_pareto(
    rows: list[dict[str, object]],
    pareto_rows: list[dict[str, object]],
    topk_rows: list[dict[str, object]],
    output_path: Path,
    *,
    dpi: int,
) -> Path:
    return _plot_paper_pareto(
        rows,
        pareto_rows,
        topk_rows,
        output_path,
        front_name="error_rate_throughput",
        x_key="end_to_end_throughput_samples_per_sec",
        y_key="error_rate",
        xlabel="End-to-end throughput (samples/s)",
        ylabel="Error rate (1 - accuracy, log scale)",
        title="Accuracy-efficiency Pareto: error rate vs throughput",
        dpi=dpi,
    )


def plot_paper_error_rate_latency_pareto(
    rows: list[dict[str, object]],
    pareto_rows: list[dict[str, object]],
    topk_rows: list[dict[str, object]],
    output_path: Path,
    *,
    dpi: int,
) -> Path:
    return _plot_paper_pareto(
        rows,
        pareto_rows,
        topk_rows,
        output_path,
        front_name="error_rate_latency",
        x_key="end_to_end_avg_latency_ms",
        y_key="error_rate",
        xlabel="End-to-end average latency (ms, log scale)",
        ylabel="Error rate (1 - accuracy, log scale)",
        title="Accuracy-efficiency Pareto: error rate vs latency",
        x_log=True,
        dpi=dpi,
    )


def plot_paper_branch_pair_summary(rows: list[dict[str, object]], output_path: Path, *, dpi: int) -> Path:
    plt = _load_pyplot()
    _apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    metrics = [
        ("accuracy_delta_amct_minus_pruning", "Accuracy delta\n(INT8 - FP16, pp)", 0.0, 100.0),
        ("throughput_ratio_amct_over_pruning", "Throughput ratio\n(INT8 / FP16)", 1.0, 1.0),
        ("latency_ratio_amct_over_pruning", "Latency ratio\n(INT8 / FP16)", 1.0, 1.0),
    ]

    for ax, (key, ylabel, baseline, scale) in zip(axes, metrics, strict=True):
        values = [(_to_float(row.get(key)) or 0.0) * scale for row in rows if _to_float(row.get(key)) is not None]
        if not values:
            continue
        positions = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.asarray([0.0])
        colors = [
            "#2ca02c" if _metric_is_better(value / scale, key) else "#d62728"
            for value in values
        ]
        ax.scatter(positions, values, s=24, c=colors, alpha=0.58, edgecolors="white", linewidths=0.4)
        ax.boxplot(
            values,
            positions=[0],
            widths=0.28,
            vert=True,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#d9d9d9", "alpha": 0.55, "linewidth": 0.8},
            medianprops={"color": "black", "linewidth": 1.2},
        )
        median_value = float(np.median(np.asarray(values, dtype=np.float64)))
        ax.axhline(baseline * scale, color="black", linewidth=0.8, linestyle="--", alpha=0.75)
        ax.text(
            0.18,
            median_value,
            f"median={median_value:.3f}",
            va="center",
            fontsize=9,
        )
        ax.set_xlim(-0.3, 0.55)
        ax.set_xticks([])
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", linestyle="--", alpha=0.28)
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="white", markerfacecolor="#2ca02c", markersize=7, label="INT8 better/equal"),
        plt.Line2D([0], [0], marker="o", color="white", markerfacecolor="#d62728", markersize=7, label="INT8 worse"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(f"INT8 AMCT vs FP16 pruning paired comparison ({len(rows)} pairs)")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return _save(fig, output_path, dpi=dpi)


def plot_paper_complexity_tradeoff(rows: list[dict[str, object]], output_path: Path, *, dpi: int) -> Path:
    plt = _load_pyplot()
    _apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    _paper_branch_scatter(
        axes[0],
        rows,
        x_key="operation_count_gmacs",
        y_key="end_to_end_avg_latency_ms",
        xlabel="Operation count (GMACs, log scale)",
        ylabel="End-to-end average latency (ms, log scale)",
        x_log=True,
        y_log=True,
    )
    axes[0].set_title("Computation vs latency")
    _paper_branch_scatter(
        axes[1],
        rows,
        x_key="parameter_count",
        y_key="end_to_end_throughput_samples_per_sec",
        xlabel="Parameter count (log scale)",
        ylabel="End-to-end throughput (samples/s)",
        x_log=True,
    )
    axes[1].set_title("Model size vs throughput")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return _save(fig, output_path, dpi=dpi)


def plot_paper_per_class_pair_delta(
    fp16_record: VisualizationRecord,
    int8_record: VisualizationRecord,
    output_path: Path,
    *,
    dpi: int,
) -> Path | None:
    fp16_rows = _load_per_class_rows(fp16_record)
    int8_rows = _load_per_class_rows(int8_record)
    if not fp16_rows or not int8_rows:
        return None

    class_names = [name for name in fp16_rows if name in int8_rows]
    if not class_names:
        return None
    deltas = [int8_rows[name] - fp16_rows[name] for name in class_names]
    colors = ["#2ca02c" if value >= 0 else "#d62728" for value in deltas]

    plt = _load_pyplot()
    _apply_paper_style()
    fig_height = max(5.5, 0.28 * len(class_names) + 1.4)
    fig, ax = plt.subplots(figsize=(7.2, fig_height))
    y_positions = np.arange(len(class_names))
    ax.barh(y_positions, deltas, color=colors, alpha=0.82)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(y_positions, class_names)
    ax.set_xlabel("Recall delta (INT8 AMCT - FP16 pruning)")
    ax.set_title(f"Per-class recall delta: {fp16_record.key.model_name}, {fp16_record.key.experiment_name}")
    ax.grid(True, axis="x", linestyle="--", alpha=0.25)
    fig.tight_layout()
    return _save(fig, output_path, dpi=dpi)


def _plot_paper_pareto(
    rows: list[dict[str, object]],
    pareto_rows: list[dict[str, object]],
    topk_rows: list[dict[str, object]],
    output_path: Path,
    *,
    front_name: str,
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    dpi: int,
    x_log: bool = False,
) -> Path:
    plt = _load_pyplot()
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    _paper_branch_scatter(
        ax,
        rows,
        x_key=x_key,
        y_key=y_key,
        xlabel=xlabel,
        ylabel=ylabel,
        y_log=True,
        x_log=x_log,
        alpha=0.24,
        size=28,
    )

    front_rows = [row for row in pareto_rows if row.get("pareto_front") == front_name]
    front_points = _valid_plot_points(front_rows, x_key=x_key, y_key=y_key)
    front_points.sort(key=lambda item: item[1])
    if front_points:
        ax.plot(
            [point[1] for point in front_points],
            [point[2] for point in front_points],
            color="black",
            linewidth=1.15,
            alpha=0.76,
            label="Pareto front",
            zorder=4,
        )
        for row, x_value, y_value in front_points:
            ax.scatter(
                [x_value],
                [y_value],
                s=78,
                color=BRANCH_COLORS.get(str(row.get("branch")), "#7f7f7f"),
                marker=BRANCH_MARKERS.get(str(row.get("branch")), "o"),
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
            )

    top_keys = {_row_identity(row) for row in topk_rows}
    for rank, row in enumerate(topk_rows, start=1):
        x_value = _plot_value(row.get(x_key), x_key)
        y_value = _plot_value(row.get(y_key), y_key)
        if x_value is None or y_value is None or _row_identity(row) not in top_keys:
            continue
        ax.annotate(
            f"T{rank}",
            xy=(x_value, y_value),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            weight="bold",
            zorder=6,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.28)
    _add_paper_legend(ax)
    fig.tight_layout()
    return _save(fig, output_path, dpi=dpi)


def _paper_branch_scatter(
    ax,
    rows: list[dict[str, object]],
    *,
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    x_log: bool = False,
    y_log: bool = False,
    alpha: float = 0.62,
    size: float = 34.0,
) -> None:
    for branch in ("pruning_fp16", "amct_deploy"):
        points = _valid_plot_points([row for row in rows if row.get("branch") == branch], x_key=x_key, y_key=y_key)
        if not points:
            continue
        ax.scatter(
            [point[1] for point in points],
            [point[2] for point in points],
            s=size,
            color=BRANCH_COLORS.get(branch, "#7f7f7f"),
            marker=BRANCH_MARKERS.get(branch, "o"),
            alpha=alpha,
            edgecolors="none",
            label=BRANCH_DISPLAY_NAMES.get(branch, branch),
        )
    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.28)


def _valid_plot_points(
    rows: list[dict[str, object]],
    *,
    x_key: str,
    y_key: str,
) -> list[tuple[dict[str, object], float, float]]:
    points: list[tuple[dict[str, object], float, float]] = []
    for row in rows:
        x_value = _plot_value(row.get(x_key), x_key)
        y_value = _plot_value(row.get(y_key), y_key)
        if x_value is None or y_value is None:
            continue
        points.append((row, x_value, y_value))
    return points


def _row_identity(row: dict[str, object]) -> tuple[str, str, str]:
    return (
        str(row.get("branch", "")),
        str(row.get("model_name", "")),
        str(row.get("experiment_name", "")),
    )


def _metric_is_better(value: float, key: str) -> bool:
    if key == "accuracy_delta_amct_minus_pruning":
        return value >= 0.0
    if key == "throughput_ratio_amct_over_pruning":
        return value >= 1.0
    if key == "latency_ratio_amct_over_pruning":
        return value <= 1.0
    return True


def _load_per_class_rows(record: VisualizationRecord) -> dict[str, float]:
    path = record.accuracy.per_class_metrics_csv
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            str(row["class_name"]): float(row.get("recall", 0.0))
            for row in csv.DictReader(handle)
            if row.get("class_name")
        }


def _apply_paper_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
        }
    )


def _add_paper_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels, strict=True):
        unique.setdefault(label, handle)
    ax.legend(unique.values(), unique.keys(), loc="best", frameon=True, framealpha=0.9)


def _load_pyplot():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    ensure_directory(Path(os.environ["MPLCONFIGDIR"]))

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = ["Times New Roman"]
    matplotlib.rcParams["font.sans-serif"] = ["Times New Roman"]
    matplotlib.rcParams["font.serif"] = ["Times New Roman"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    import matplotlib.pyplot as plt

    return plt


def _save(fig, output_path: Path, *, dpi: int) -> Path:
    ensure_directory(output_path.parent)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.clf()
    return output_path


def _to_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _plot_value(value: object, key: str) -> float | None:
    metric = _to_float(value)
    if metric is None:
        return None
    if key.endswith("error_rate") or key == "error_rate":
        return max(metric, LOG_SAFE_ERROR_RATE_FLOOR)
    if key in {"operation_count_gmacs", "operation_count", "parameter_count"}:
        return max(metric, LOG_SAFE_COMPLEXITY_FLOOR)
    if key in {
        "end_to_end_avg_latency_ms",
        "end_to_end_p50_latency_ms",
        "end_to_end_p95_latency_ms",
        "end_to_end_p99_latency_ms",
        "pure_infer_avg_latency_ms",
    }:
        return max(metric, LOG_SAFE_LATENCY_FLOOR)
    return metric
