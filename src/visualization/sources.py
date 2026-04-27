from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.common.report import resolve_repo_path

from .schema import (
    AccuracyResult,
    ArtifactKey,
    EfficiencyResult,
    ExperimentSpec,
    MissingInput,
    VisualizationRecord,
)


EXPERIMENT_PATTERN = re.compile(
    r"^from_ratio(?P<ratio>[0-9.]+)_steps(?P<steps>\d+)_(?P<mode>[^_]+)_ft(?P<ft>\d+)_bs(?P<bs>\d+)$"
)


class VisualizationError(RuntimeError):
    """可视化数据加载或生成失败。"""


def parse_experiment_name(experiment_name: str) -> ExperimentSpec:
    match = EXPERIMENT_PATTERN.match(experiment_name)
    if match is None:
        return ExperimentSpec(
            ratio=None,
            steps=None,
            pruning_mode=None,
            finetune_epochs=None,
            batch_size=None,
        )
    return ExperimentSpec(
        ratio=float(match.group("ratio")),
        steps=int(match.group("steps")),
        pruning_mode=match.group("mode"),
        finetune_epochs=int(match.group("ft")),
        batch_size=int(match.group("bs")),
    )


def load_visualization_records(
    *,
    accuracy_root: str | Path,
    efficiency_root: str | Path,
    efficiency_pattern: str,
    branch: str | None = None,
    model_name: str | None = None,
    experiment_name: str | None = None,
    num_instances: int | None = 1,
    buffer_depth: int | None = 1,
) -> tuple[list[VisualizationRecord], list[MissingInput]]:
    accuracy_results = scan_accuracy_results(
        accuracy_root,
        branch=branch,
        model_name=model_name,
        experiment_name=experiment_name,
    )
    efficiency_results = scan_efficiency_results(
        efficiency_root,
        efficiency_pattern=efficiency_pattern,
        branch=branch,
        model_name=model_name,
        experiment_name=experiment_name,
        num_instances=num_instances,
        buffer_depth=buffer_depth,
    )

    keys = sorted(set(accuracy_results).union(efficiency_results))
    records: list[VisualizationRecord] = []
    missing_inputs: list[MissingInput] = []
    for key in keys:
        accuracy = accuracy_results.get(key)
        efficiency = efficiency_results.get(key)
        if accuracy is None or efficiency is None:
            missing_inputs.append(
                MissingInput(
                    key=key,
                    missing_accuracy=accuracy is None,
                    missing_efficiency=efficiency is None,
                    accuracy_summary_path=accuracy.summary_path if accuracy else None,
                    efficiency_summary_path=efficiency.summary_path if efficiency else None,
                )
            )
            continue

        records.append(
            VisualizationRecord(
                key=key,
                experiment=parse_experiment_name(key.experiment_name),
                accuracy=accuracy,
                efficiency=efficiency,
                om_size_bytes=_resolve_om_size(efficiency.payload.get("artifact_path")),
            )
        )

    return records, missing_inputs


def scan_accuracy_results(
    accuracy_root: str | Path,
    *,
    branch: str | None = None,
    model_name: str | None = None,
    experiment_name: str | None = None,
) -> dict[ArtifactKey, AccuracyResult]:
    root = resolve_repo_path(accuracy_root)
    if not root.exists():
        raise VisualizationError(f"找不到 accuracy 输出目录: {root}")

    results: dict[ArtifactKey, AccuracyResult] = {}
    for summary_path in sorted(root.rglob("summary.json")):
        payload = _load_json(summary_path)
        key = _key_from_payload_or_path(payload, summary_path, root)
        if not _matches_filters(key, branch=branch, model_name=model_name, experiment_name=experiment_name):
            continue

        results[key] = AccuracyResult(
            key=key,
            summary_path=summary_path,
            payload=payload,
            confusion_matrix_csv=_optional_summary_path(payload, "confusion_matrix_csv"),
            confusion_matrix_png=_optional_summary_path(payload, "confusion_matrix_png"),
            per_class_metrics_csv=_optional_summary_path(payload, "per_class_metrics_csv"),
        )
    return results


def scan_efficiency_results(
    efficiency_root: str | Path,
    *,
    efficiency_pattern: str,
    branch: str | None = None,
    model_name: str | None = None,
    experiment_name: str | None = None,
    num_instances: int | None = 1,
    buffer_depth: int | None = 1,
) -> dict[ArtifactKey, EfficiencyResult]:
    root = resolve_repo_path(efficiency_root)
    if not root.exists():
        raise VisualizationError(f"找不到 efficiency 输出目录: {root}")

    results: dict[ArtifactKey, EfficiencyResult] = {}
    for summary_path in sorted(root.rglob(efficiency_pattern)):
        payload = _load_json(summary_path)
        key = _key_from_payload_or_path(payload, summary_path, root)
        if not _matches_filters(key, branch=branch, model_name=model_name, experiment_name=experiment_name):
            continue
        if num_instances is not None and int(payload.get("num_instances", -1)) != int(num_instances):
            continue
        if buffer_depth is not None and int(payload.get("buffer_depth", -1)) != int(buffer_depth):
            continue

        results[key] = EfficiencyResult(
            key=key,
            summary_path=summary_path,
            payload=payload,
        )
    return results


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise VisualizationError(f"summary 不是 JSON object: {path}")
    return payload


def _key_from_payload_or_path(payload: dict[str, Any], summary_path: Path, root: Path) -> ArtifactKey:
    branch = payload.get("branch")
    model_name = payload.get("model_name")
    experiment_name = payload.get("experiment_name")
    if isinstance(branch, str) and isinstance(model_name, str) and isinstance(experiment_name, str):
        return ArtifactKey(branch=branch, model_name=model_name, experiment_name=experiment_name)

    try:
        relative_parts = summary_path.relative_to(root).parts
    except ValueError as exc:
        raise VisualizationError(f"无法从路径解析 summary key: {summary_path}") from exc
    if len(relative_parts) < 4:
        raise VisualizationError(f"summary 路径层级不足: {summary_path}")
    return ArtifactKey(
        branch=relative_parts[0],
        model_name=relative_parts[1],
        experiment_name=relative_parts[2],
    )


def _matches_filters(
    key: ArtifactKey,
    *,
    branch: str | None,
    model_name: str | None,
    experiment_name: str | None,
) -> bool:
    return (
        (branch is None or key.branch == branch)
        and (model_name is None or key.model_name == model_name)
        and (experiment_name is None or key.experiment_name == experiment_name)
    )


def _optional_summary_path(payload: dict[str, Any], key: str) -> Path | None:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        return None
    return resolve_repo_path(value)


def _resolve_om_size(artifact_path: object) -> int | None:
    if not isinstance(artifact_path, str) or not artifact_path:
        return None
    path = resolve_repo_path(artifact_path)
    if not path.exists() or not path.is_file():
        return None
    return int(path.stat().st_size)
