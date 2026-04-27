from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from src.common.report import ensure_directory

from .schema import VisualizationRecord


BRANCH_COLORS = {
    "pruning_fp16": "#1f77b4",
    "amct_deploy": "#ff7f0e",
}
MODEL_MARKERS = {
    "resnet6_2d": "o",
    "resnet10_2d": "s",
    "resnet14_2d": "^",
    "resnet18_2d": "D",
    "resnet34_2d": "P",
}
MODEL_MARKER_SIZES = {
    "resnet6_2d": 50.0,
    "resnet10_2d": 50.0,
    "resnet14_2d": 50.0,
    "resnet18_2d": 25.0,
    "resnet34_2d": 25.0,
}


LOG_SAFE_ERROR_RATE_FLOOR = 1e-12
LOG_SAFE_COMPLEXITY_FLOOR = 1e-12
LOG_SAFE_LATENCY_FLOOR = 1e-12


def plot_error_rate_throughput_pareto(rows: list[dict[str, object]], output_path: Path, *, dpi: int) -> Path:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(9, 6))
    _scatter_by_branch_model(
        ax,
        rows,
        x_key="end_to_end_throughput_samples_per_sec",
        y_key="error_rate",
        size_key="operation_count_gmacs",
    )
    ax.set_xlabel("End-to-end throughput (samples/s)")
    ax.set_ylabel("Error rate (1 - accuracy, log scale)")
    ax.set_yscale("log")
    ax.set_title("Error Rate vs Throughput Pareto View")
    ax.grid(True, linestyle="--", alpha=0.3)
    _add_legend(ax)
    return _save(fig, output_path, dpi=dpi)


def plot_error_rate_latency_pareto(rows: list[dict[str, object]], output_path: Path, *, dpi: int) -> Path:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(9, 6))
    _scatter_by_branch_model(
        ax,
        rows,
        x_key="end_to_end_avg_latency_ms",
        y_key="error_rate",
        size_key="operation_count_gmacs",
    )
    ax.set_xlabel("End-to-end avg latency (ms, log scale)")
    ax.set_ylabel("Error rate (1 - accuracy, log scale)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Error Rate vs Latency Pareto View")
    ax.grid(True, linestyle="--", alpha=0.3)
    _add_legend(ax)
    return _save(fig, output_path, dpi=dpi)


def plot_macs_latency_scatter(rows: list[dict[str, object]], output_path: Path, *, dpi: int) -> Path:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(9, 6))
    _scatter_by_branch_model(
        ax,
        rows,
        x_key="operation_count_gmacs",
        y_key="end_to_end_avg_latency_ms",
        size_key="accuracy",
    )
    ax.set_xlabel("Operation count (GMACs, log scale)")
    ax.set_ylabel("End-to-end avg latency (ms, log scale)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Model Complexity vs Latency")
    ax.grid(True, linestyle="--", alpha=0.3)
    _add_legend(ax)
    return _save(fig, output_path, dpi=dpi)


def plot_params_throughput_scatter(rows: list[dict[str, object]], output_path: Path, *, dpi: int) -> Path:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(9, 6))
    _scatter_by_branch_model(
        ax,
        rows,
        x_key="parameter_count",
        y_key="end_to_end_throughput_samples_per_sec",
        size_key="accuracy",
    )
    ax.set_xlabel("Parameter count (log scale)")
    ax.set_ylabel("End-to-end throughput (samples/s)")
    ax.set_xscale("log")
    ax.set_title("Parameter Count vs Throughput")
    ax.grid(True, linestyle="--", alpha=0.3)
    _add_legend(ax)
    return _save(fig, output_path, dpi=dpi)


def plot_branch_pair_slope(rows: list[dict[str, object]], output_path: Path, *, dpi: int) -> Path:
    plt = _load_pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metric_pairs = [
        ("pruning_error_rate", "amct_error_rate", "Error rate", True),
        ("pruning_throughput", "amct_throughput", "Throughput (samples/s)", False),
        ("pruning_latency", "amct_latency", "Latency (ms)", True),
    ]
    for ax, (left_key, right_key, title, lower_is_better) in zip(axes, metric_pairs, strict=True):
        plotted = 0
        for row in rows:
            left = _plot_value(row.get(left_key), left_key)
            right = _plot_value(row.get(right_key), right_key)
            if left is None or right is None:
                continue
            color = "#2ca02c" if (right <= left if lower_is_better else right >= left) else "#d62728"
            ax.plot([0, 1], [left, right], color=color, alpha=0.35, linewidth=1.2)
            plotted += 1
        ax.set_xticks([0, 1], ["pruning_fp16", "amct_deploy"])
        ax.set_title(f"{title} ({plotted} pairs)")
        if left_key.endswith("error_rate") or left_key.endswith("latency"):
            ax.set_yscale("log")
            ax.set_ylabel("log scale")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.suptitle("Paired Branch Comparison")
    fig.tight_layout()
    return _save(fig, output_path, dpi=dpi)


def plot_topk_candidates(rows: list[dict[str, object]], output_path: Path, *, dpi: int) -> Path:
    plt = _load_pyplot()
    selected = list(reversed(rows))
    labels = [
        f"{row['branch']} | {row['model_name']} | r={row['ratio']} s={row['steps']}"
        for row in selected
    ]
    values = [_to_float(row.get("end_to_end_throughput_samples_per_sec")) or 0.0 for row in selected]
    colors = [BRANCH_COLORS.get(str(row["branch"]), "#7f7f7f") for row in selected]

    fig_height = max(5, 0.35 * len(selected) + 1.5)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.barh(range(len(selected)), values, color=colors)
    ax.set_yticks(range(len(selected)), labels)
    ax.set_xlabel("End-to-end throughput (samples/s)")
    ax.set_title("Top Candidates Under Accuracy Constraint")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    for index, row in enumerate(selected):
        error_rate = _to_float(row.get("error_rate"))
        if error_rate is not None:
            ax.text(values[index], index, f"  err={error_rate:.2e}", va="center", fontsize=8)
    fig.tight_layout()
    return _save(fig, output_path, dpi=dpi)


def plot_per_class_metrics_heatmap(record: VisualizationRecord, output_path: Path, *, dpi: int) -> Path | None:
    per_class_path = record.accuracy.per_class_metrics_csv
    if per_class_path is None or not per_class_path.exists():
        return None

    with per_class_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None

    metric_names = ["precision", "recall", "specificity"]
    class_names = [str(row["class_name"]) for row in rows]
    matrix = np.asarray(
        [[float(row.get(metric_name, 0.0)) for metric_name in metric_names] for row in rows],
        dtype=np.float64,
    )

    plt = _load_pyplot()
    fig_height = max(6, 0.35 * len(class_names))
    fig, ax = plt.subplots(figsize=(7, fig_height))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(metric_names)), metric_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_title(f"Per-class Metrics: {record.key.branch} / {record.key.model_name}")
    for y_index in range(matrix.shape[0]):
        for x_index in range(matrix.shape[1]):
            ax.text(x_index, y_index, f"{matrix[y_index, x_index]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _save(fig, output_path, dpi=dpi)


def _scatter_by_branch_model(
    ax,
    rows: list[dict[str, object]],
    *,
    x_key: str,
    y_key: str,
    size_key: str,
) -> None:
    for branch in sorted({str(row["branch"]) for row in rows}):
        for model_name in sorted({str(row["model_name"]) for row in rows if str(row["branch"]) == branch}):
            selected = [row for row in rows if row["branch"] == branch and row["model_name"] == model_name]
            x_values = [_plot_value(row.get(x_key), x_key) for row in selected]
            y_values = [_plot_value(row.get(y_key), y_key) for row in selected]
            size_values = [_marker_size(model_name=model_name) for _ in selected]
            points = [
                (row, x_value, y_value, size_value)
                for row, x_value, y_value, size_value in zip(selected, x_values, y_values, size_values, strict=True)
                if x_value is not None and y_value is not None
            ]
            if not points:
                continue
            ax.scatter(
                [point[1] for point in points],
                [point[2] for point in points],
                s=[point[3] for point in points],
                alpha=0.72,
                color=BRANCH_COLORS.get(branch, "#7f7f7f"),
                marker=MODEL_MARKERS.get(model_name, "o"),
                edgecolors="black",
                linewidths=0.3,
                label=f"{branch}/{model_name}",
            )
            _annotate_points(ax, points)


def _marker_size(*, model_name: str) -> float:
    return MODEL_MARKER_SIZES.get(model_name, 50.0)


def _add_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels, strict=True):
        unique.setdefault(label, handle)
    ax.legend(
        unique.values(),
        unique.keys(),
        loc="best",
        fontsize=5,
        markerscale=0.55,
        labelspacing=0.25,
        handletextpad=0.25,
        borderpad=0.25,
        framealpha=0.75,
        ncols=2,
        frameon=True,
    )


def _annotate_points(ax, points: list[tuple[dict[str, object], float, float, float]]) -> None:
    for row, x_value, y_value, _ in points:
        label = _format_pruning_label(row)
        if not label:
            continue
        ax.annotate(
            label,
            xy=(x_value, y_value),
            xytext=(2, 2),
            textcoords="offset points",
            fontsize=5,
            alpha=0.75,
        )


def _format_pruning_label(row: dict[str, object]) -> str:
    ratio = _to_float(row.get("ratio"))
    steps = _to_float(row.get("steps"))
    if ratio is None or steps is None:
        return ""
    return f"{ratio:.2f}/{int(steps)}"


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
        "pruning_latency",
        "amct_latency",
    }:
        return max(metric, LOG_SAFE_LATENCY_FLOOR)
    return metric
