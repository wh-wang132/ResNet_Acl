from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.common.report import to_repo_relative

from .schema import MissingInput, VisualizationRecord

BRANCH_DISPLAY_NAMES = {
    "pruning_fp16": "FP16 pruning",
    "amct_deploy": "INT8 AMCT",
}


def build_artifact_metrics(records: Iterable[VisualizationRecord]) -> list[dict[str, object]]:
    return [_record_to_row(record) for record in sorted(records, key=_record_sort_key)]


def build_missing_inputs(missing_inputs: Iterable[MissingInput]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in sorted(missing_inputs, key=lambda value: value.key):
        rows.append(
            {
                "branch": item.key.branch,
                "model_name": item.key.model_name,
                "experiment_name": item.key.experiment_name,
                "missing_accuracy": item.missing_accuracy,
                "missing_efficiency": item.missing_efficiency,
                "accuracy_summary_path": _path_or_empty(item.accuracy_summary_path),
                "efficiency_summary_path": _path_or_empty(item.efficiency_summary_path),
            }
        )
    return rows


def build_pareto_candidates(records: Iterable[VisualizationRecord]) -> list[dict[str, object]]:
    record_list = list(records)
    rows: list[dict[str, object]] = []
    for front_name, front_records in (
        ("error_rate_throughput", _pareto_accuracy_throughput(record_list)),
        ("error_rate_latency", _pareto_accuracy_latency(record_list)),
    ):
        for record in sorted(front_records, key=_record_sort_key):
            row = _record_to_row(record)
            row["pareto_front"] = front_name
            rows.append(row)
    return rows


def build_topk_candidates(
    records: Iterable[VisualizationRecord],
    *,
    top_k: int,
    accuracy_floor: float,
) -> list[dict[str, object]]:
    record_list = list(records)
    filtered = [
        record
        for record in record_list
        if _float_metric(record, "accuracy") is not None and _float_metric(record, "accuracy") >= accuracy_floor
    ]
    if not filtered:
        filtered = record_list
    ranked = sorted(
        filtered,
        key=lambda record: (
            _float_metric(record, "end_to_end_throughput_samples_per_sec") or 0.0,
            _float_metric(record, "accuracy") or 0.0,
        ),
        reverse=True,
    )
    rows = []
    for rank, record in enumerate(ranked[:top_k], start=1):
        row = _record_to_row(record)
        row["rank"] = rank
        row["accuracy_floor"] = accuracy_floor
        rows.append(row)
    return rows


def build_paper_top_candidates(topk_rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in topk_rows:
        rows.append(
            {
                "rank": row.get("rank", ""),
                "branch": row.get("branch", ""),
                "branch_display": branch_display_name(row.get("branch")),
                "model_name": row.get("model_name", ""),
                "experiment_name": row.get("experiment_name", ""),
                "ratio": row.get("ratio", ""),
                "steps": row.get("steps", ""),
                "accuracy": row.get("accuracy", ""),
                "error_rate": row.get("error_rate", ""),
                "end_to_end_throughput_samples_per_sec": row.get("end_to_end_throughput_samples_per_sec", ""),
                "end_to_end_avg_latency_ms": row.get("end_to_end_avg_latency_ms", ""),
                "operation_count_gmacs": row.get("operation_count_gmacs", ""),
                "parameter_count": row.get("parameter_count", ""),
            }
        )
    return rows


def build_branch_pairs(records: Iterable[VisualizationRecord]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, VisualizationRecord]] = {}
    for record in records:
        grouped.setdefault((record.key.model_name, record.key.experiment_name), {})[record.key.branch] = record

    rows: list[dict[str, object]] = []
    for (model_name, experiment_name), by_branch in sorted(grouped.items()):
        pruning = by_branch.get("pruning_fp16")
        amct = by_branch.get("amct_deploy")
        if pruning is None or amct is None:
            continue
        pruning_row = _record_to_row(pruning)
        amct_row = _record_to_row(amct)
        pruning_throughput = _to_float(pruning_row["end_to_end_throughput_samples_per_sec"])
        amct_throughput = _to_float(amct_row["end_to_end_throughput_samples_per_sec"])
        pruning_latency = _to_float(pruning_row["end_to_end_avg_latency_ms"])
        amct_latency = _to_float(amct_row["end_to_end_avg_latency_ms"])
        pruning_accuracy = _to_float(pruning_row["accuracy"])
        amct_accuracy = _to_float(amct_row["accuracy"])
        pruning_error_rate = _to_float(pruning_row["error_rate"])
        amct_error_rate = _to_float(amct_row["error_rate"])
        rows.append(
            {
                "model_name": model_name,
                "experiment_name": experiment_name,
                "ratio": pruning_row["ratio"],
                "steps": pruning_row["steps"],
                "pruning_accuracy": pruning_accuracy,
                "amct_accuracy": amct_accuracy,
                "accuracy_delta_amct_minus_pruning": _delta(amct_accuracy, pruning_accuracy),
                "pruning_error_rate": pruning_error_rate,
                "amct_error_rate": amct_error_rate,
                "error_rate_delta_amct_minus_pruning": _delta(amct_error_rate, pruning_error_rate),
                "pruning_throughput": pruning_throughput,
                "amct_throughput": amct_throughput,
                "throughput_ratio_amct_over_pruning": _ratio(amct_throughput, pruning_throughput),
                "pruning_latency": pruning_latency,
                "amct_latency": amct_latency,
                "latency_ratio_amct_over_pruning": _ratio(amct_latency, pruning_latency),
            }
        )
    return rows


def build_paper_branch_pair_summary(branch_pair_rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    rows = list(branch_pair_rows)
    grouped: dict[str, list[dict[str, object]]] = {"overall": rows}
    for row in rows:
        grouped.setdefault(str(row.get("model_name", "")), []).append(row)

    summary_rows: list[dict[str, object]] = []
    for group_name, group_rows in grouped.items():
        if not group_rows:
            continue
        summary_rows.append(_build_pair_summary_row(group_name, group_rows))
    return summary_rows


def branch_display_name(branch: object) -> str:
    return BRANCH_DISPLAY_NAMES.get(str(branch), str(branch))


def _record_to_row(record: VisualizationRecord) -> dict[str, object]:
    accuracy = record.accuracy.payload
    efficiency = record.efficiency.payload
    accuracy_value = _to_float(accuracy.get("accuracy"))
    error_rate = 1.0 - accuracy_value if accuracy_value is not None else None
    operation_count = _to_int(efficiency.get("operation_count"))
    parameter_count = _to_int(efficiency.get("parameter_count"))
    return {
        "branch": record.key.branch,
        "model_name": record.key.model_name,
        "experiment_name": record.key.experiment_name,
        "ratio": _none_to_empty(record.experiment.ratio),
        "steps": _none_to_empty(record.experiment.steps),
        "pruning_mode": _none_to_empty(record.experiment.pruning_mode),
        "finetune_epochs": _none_to_empty(record.experiment.finetune_epochs),
        "batch_size": _none_to_empty(record.experiment.batch_size),
        "samples": _none_to_empty(accuracy.get("samples")),
        "accuracy": _none_to_empty(accuracy.get("accuracy")),
        "error_rate": _none_to_empty(error_rate),
        "error_rate_percent": _none_to_empty(error_rate * 100.0 if error_rate is not None else None),
        "avg_loss": _none_to_empty(accuracy.get("avg_loss")),
        "parameter_count": _none_to_empty(parameter_count),
        "operation_count": _none_to_empty(operation_count),
        "operation_count_gmacs": _none_to_empty(operation_count / 1_000_000_000 if operation_count is not None else None),
        "om_size_bytes": _none_to_empty(record.om_size_bytes),
        "om_size_mib": _none_to_empty(record.om_size_bytes / 1024 / 1024 if record.om_size_bytes is not None else None),
        "num_instances": _none_to_empty(efficiency.get("num_instances")),
        "active_instances": _none_to_empty(efficiency.get("active_instances")),
        "buffer_depth": _none_to_empty(efficiency.get("buffer_depth")),
        "end_to_end_avg_latency_ms": _none_to_empty(efficiency.get("end_to_end_avg_latency_ms")),
        "end_to_end_p50_latency_ms": _none_to_empty(efficiency.get("end_to_end_p50_latency_ms")),
        "end_to_end_p95_latency_ms": _none_to_empty(efficiency.get("end_to_end_p95_latency_ms")),
        "end_to_end_p99_latency_ms": _none_to_empty(efficiency.get("end_to_end_p99_latency_ms")),
        "end_to_end_throughput_samples_per_sec": _none_to_empty(
            efficiency.get("end_to_end_throughput_samples_per_sec")
        ),
        "pure_infer_avg_latency_ms": _none_to_empty(efficiency.get("pure_infer_avg_latency_ms")),
        "pure_infer_throughput_samples_per_sec": _none_to_empty(
            efficiency.get("pure_infer_throughput_samples_per_sec")
        ),
        "h2d_memcpy_total_ms": _none_to_empty(efficiency.get("h2d_memcpy_total_ms")),
        "execute_wait_total_ms": _none_to_empty(efficiency.get("execute_wait_total_ms")),
        "d2h_memcpy_total_ms": _none_to_empty(efficiency.get("d2h_memcpy_total_ms")),
        "output_decode_total_ms": _none_to_empty(efficiency.get("output_decode_total_ms")),
        "artifact_path": _none_to_empty(efficiency.get("artifact_path")),
        "accuracy_summary_path": to_repo_relative(record.accuracy.summary_path),
        "efficiency_summary_path": to_repo_relative(record.efficiency.summary_path),
        "confusion_matrix_csv": _path_or_empty(record.accuracy.confusion_matrix_csv),
        "confusion_matrix_png": _path_or_empty(record.accuracy.confusion_matrix_png),
        "per_class_metrics_csv": _path_or_empty(record.accuracy.per_class_metrics_csv),
    }


def _pareto_accuracy_throughput(records: list[VisualizationRecord]) -> list[VisualizationRecord]:
    candidates = [
        record
        for record in records
        if _float_metric(record, "accuracy") is not None
        and _float_metric(record, "end_to_end_throughput_samples_per_sec") is not None
    ]
    return [
        record
        for record in candidates
        if not any(_dominates_accuracy_throughput(other, record) for other in candidates if other is not record)
    ]


def _pareto_accuracy_latency(records: list[VisualizationRecord]) -> list[VisualizationRecord]:
    candidates = [
        record
        for record in records
        if _float_metric(record, "accuracy") is not None and _float_metric(record, "end_to_end_avg_latency_ms") is not None
    ]
    return [
        record
        for record in candidates
        if not any(_dominates_accuracy_latency(other, record) for other in candidates if other is not record)
    ]


def _dominates_accuracy_throughput(left: VisualizationRecord, right: VisualizationRecord) -> bool:
    left_accuracy = _float_metric(left, "accuracy") or 0.0
    right_accuracy = _float_metric(right, "accuracy") or 0.0
    left_throughput = _float_metric(left, "end_to_end_throughput_samples_per_sec") or 0.0
    right_throughput = _float_metric(right, "end_to_end_throughput_samples_per_sec") or 0.0
    return (
        left_accuracy >= right_accuracy
        and left_throughput >= right_throughput
        and (left_accuracy > right_accuracy or left_throughput > right_throughput)
    )


def _dominates_accuracy_latency(left: VisualizationRecord, right: VisualizationRecord) -> bool:
    left_accuracy = _float_metric(left, "accuracy") or 0.0
    right_accuracy = _float_metric(right, "accuracy") or 0.0
    left_latency = _float_metric(left, "end_to_end_avg_latency_ms")
    right_latency = _float_metric(right, "end_to_end_avg_latency_ms")
    if left_latency is None or right_latency is None:
        return False
    return (
        left_accuracy >= right_accuracy
        and left_latency <= right_latency
        and (left_accuracy > right_accuracy or left_latency < right_latency)
    )


def _record_sort_key(record: VisualizationRecord) -> tuple[object, ...]:
    return (
        record.key.branch,
        _natural_model_number(record.key.model_name),
        record.key.model_name,
        record.experiment.ratio if record.experiment.ratio is not None else -1.0,
        record.experiment.steps if record.experiment.steps is not None else -1,
        record.key.experiment_name,
    )


def _natural_model_number(model_name: str) -> int:
    digits = "".join(char for char in model_name if char.isdigit())
    return int(digits) if digits else 0


def _float_metric(record: VisualizationRecord, key: str) -> float | None:
    if key in {"accuracy", "avg_loss"}:
        return _to_float(record.accuracy.payload.get(key))
    return _to_float(record.efficiency.payload.get(key))


def _path_or_empty(path: Path | None) -> str:
    return "" if path is None else to_repo_relative(path)


def _none_to_empty(value: object) -> object:
    return "" if value is None else value


def _to_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _delta(left: float | None, right: float | None) -> float | str:
    if left is None or right is None:
        return ""
    return left - right


def _ratio(left: float | None, right: float | None) -> float | str:
    if left is None or right in (None, 0.0):
        return ""
    return left / right


def _build_pair_summary_row(group_name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    accuracy_delta_pp = [
        value * 100.0
        for value in _numeric_values(row.get("accuracy_delta_amct_minus_pruning") for row in rows)
    ]
    throughput_ratio = _numeric_values(row.get("throughput_ratio_amct_over_pruning") for row in rows)
    latency_ratio = _numeric_values(row.get("latency_ratio_amct_over_pruning") for row in rows)
    fp16_throughput = _numeric_values(row.get("pruning_throughput") for row in rows)
    int8_throughput = _numeric_values(row.get("amct_throughput") for row in rows)
    fp16_latency = _numeric_values(row.get("pruning_latency") for row in rows)
    int8_latency = _numeric_values(row.get("amct_latency") for row in rows)
    return {
        "group": "overall" if group_name == "overall" else "model",
        "model_name": "" if group_name == "overall" else group_name,
        "pairs": len(rows),
        "accuracy_delta_pp_median": _median_or_empty(accuracy_delta_pp),
        "accuracy_delta_pp_min": _min_or_empty(accuracy_delta_pp),
        "accuracy_delta_pp_max": _max_or_empty(accuracy_delta_pp),
        "throughput_ratio_median": _median_or_empty(throughput_ratio),
        "throughput_ratio_min": _min_or_empty(throughput_ratio),
        "throughput_ratio_max": _max_or_empty(throughput_ratio),
        "latency_ratio_median": _median_or_empty(latency_ratio),
        "latency_ratio_min": _min_or_empty(latency_ratio),
        "latency_ratio_max": _max_or_empty(latency_ratio),
        "fp16_throughput_median": _median_or_empty(fp16_throughput),
        "int8_throughput_median": _median_or_empty(int8_throughput),
        "fp16_latency_median": _median_or_empty(fp16_latency),
        "int8_latency_median": _median_or_empty(int8_latency),
    }


def _numeric_values(values: Iterable[object]) -> list[float]:
    result: list[float] = []
    for value in values:
        converted = _to_float(value)
        if converted is not None:
            result.append(converted)
    return result


def _median_or_empty(values: list[float]) -> float | str:
    if not values:
        return ""
    values = sorted(values)
    midpoint = len(values) // 2
    if len(values) % 2:
        return values[midpoint]
    return (values[midpoint - 1] + values[midpoint]) / 2.0


def _min_or_empty(values: list[float]) -> float | str:
    return min(values) if values else ""


def _max_or_empty(values: list[float]) -> float | str:
    return max(values) if values else ""
