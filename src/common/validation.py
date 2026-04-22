from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .artifact_scanner import ArtifactRecord, ArtifactScanError, resolve_artifacts
from .branch_specs import BRANCH_SPECS, REPO_ROOT
from .data import preload_npy_samples
from .manifest import (
    build_all_records,
    build_test_records,
    load_manifest,
    natural_sort_key,
    resolve_repo_path,
    scan_data_records,
)
from .model_complexity import ModelComplexityError, collect_model_complexity

DEFAULT_SAMPLE_LIMIT = 32


class ValidationError(RuntimeError):
    """仓库预检失败。"""


def to_repo_relative(path_like: str | Path) -> str:
    path = Path(path_like).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_cli_path(path_like: str | Path) -> Path:
    return resolve_repo_path(path_like).resolve()


def _validate_sample_limit(sample_limit: int) -> int:
    if sample_limit <= 0:
        raise ValidationError("--sample_limit 必须为正整数")
    return int(sample_limit)


def _validate_branch_and_artifact(branch: str | None, artifact_path: str | Path | None) -> Path | None:
    if artifact_path is None:
        return None
    if branch is None:
        raise ValidationError("指定 --artifact_path 时必须同时指定 --branch")
    return resolve_cli_path(artifact_path)


def _select_test_records(records, sample_limit: int):
    if not records:
        raise ValidationError("test 集为空，无法执行静态样本校验")
    return records[:sample_limit]


def _summarize_data_scan(manifest: dict[str, object], data_dir: Path) -> dict[str, object]:
    class_names = [str(name) for name in manifest["class_names"]]
    scanned_records = scan_data_records(data_dir)
    manifest_records = build_all_records(manifest, data_dir)

    actual_class_names = sorted(
        [item.name for item in data_dir.iterdir() if item.is_dir()],
        key=natural_sort_key,
    )
    scanned_paths = {record.relative_path for record in scanned_records}
    manifest_paths = {record.relative_path for record in manifest_records}
    per_class_counts = Counter(record.label_name for record in scanned_records)
    missing_referenced = sorted(manifest_paths.difference(scanned_paths), key=natural_sort_key)
    untracked_files = sorted(scanned_paths.difference(manifest_paths), key=natural_sort_key)

    counts_by_manifest_order = {label_name: int(per_class_counts.get(label_name, 0)) for label_name in class_names}
    non_zero_counts = [count for count in counts_by_manifest_order.values() if count > 0]

    return {
        "manifest_declared_data_dir": str(manifest.get("data_dir", "")),
        "actual_class_names": actual_class_names,
        "class_names_match_manifest": actual_class_names == class_names,
        "total_files": len(scanned_records),
        "files_per_class": counts_by_manifest_order,
        "min_files_per_class": int(min(non_zero_counts)) if non_zero_counts else 0,
        "max_files_per_class": int(max(non_zero_counts)) if non_zero_counts else 0,
        "missing_referenced_files_count": len(missing_referenced),
        "missing_referenced_examples": missing_referenced[:5],
        "untracked_files_count": len(untracked_files),
        "untracked_examples": untracked_files[:5],
    }


def _run_complexity_precheck(artifact: ArtifactRecord, *, skip_complexity_precheck: bool) -> None:
    if skip_complexity_precheck:
        return
    try:
        collect_model_complexity(artifact)
    except (ArtifactScanError, ModelComplexityError) as exc:
        raise ValidationError(
            f"complexity 预检失败: artifact={to_repo_relative(artifact.model_path)}: {exc}"
        ) from exc


def _build_artifact_entry(
    artifact: ArtifactRecord,
    *,
    checked_samples: int,
    skip_complexity_precheck: bool,
) -> dict[str, object]:
    _run_complexity_precheck(artifact, skip_complexity_precheck=skip_complexity_precheck)
    return {
        "model_name": artifact.model_name,
        "experiment_name": artifact.experiment_name,
        "artifact_path": to_repo_relative(artifact.model_path),
        "summary_path": to_repo_relative(artifact.summary_path),
        "input_name": artifact.input_spec.name,
        "input_shape": list(artifact.input_spec.shape),
        "input_dtype": str(np.dtype(artifact.input_spec.dtype)),
        "output_name": artifact.output_specs[0].name,
        "output_shape": list(artifact.output_specs[0].shape),
        "output_dtype": str(np.dtype(artifact.output_specs[0].dtype)),
        "sample_checks": checked_samples,
    }


def _build_branch_report(branch: str, selected_records, *, skip_complexity_precheck: bool) -> dict[str, object]:
    artifacts = resolve_artifacts(branch)
    preload_cache: dict[tuple[tuple[int, ...], str], int] = {}
    artifact_entries: list[dict[str, object]] = []

    for artifact in artifacts:
        cache_key = (tuple(artifact.input_spec.shape), str(np.dtype(artifact.input_spec.dtype)))
        checked_samples = preload_cache.get(cache_key)
        if checked_samples is None:
            preloaded = preload_npy_samples(
                selected_records,
                artifact.input_spec,
                preload_dtype=np.dtype(artifact.input_spec.dtype),
            )
            checked_samples = len(preloaded)
            preload_cache[cache_key] = checked_samples

        artifact_entries.append(
            _build_artifact_entry(
                artifact,
                checked_samples=checked_samples,
                skip_complexity_precheck=skip_complexity_precheck,
            )
        )

    unique_input_shapes = sorted({tuple(entry["input_shape"]) for entry in artifact_entries})
    unique_output_shapes = sorted({tuple(entry["output_shape"]) for entry in artifact_entries})
    input_dtypes = sorted({str(entry["input_dtype"]) for entry in artifact_entries})
    output_dtypes = sorted({str(entry["output_dtype"]) for entry in artifact_entries})

    return {
        "branch": branch,
        "artifact_count": len(artifact_entries),
        "model_names": sorted({entry["model_name"] for entry in artifact_entries}, key=natural_sort_key),
        "input_shapes": [list(shape) for shape in unique_input_shapes],
        "output_shapes": [list(shape) for shape in unique_output_shapes],
        "input_dtypes": input_dtypes,
        "output_dtypes": output_dtypes,
        "artifacts": artifact_entries,
    }


def _build_single_artifact_report(
    branch: str,
    artifact_path: Path,
    selected_records,
    *,
    skip_complexity_precheck: bool,
) -> dict[str, object]:
    artifact = resolve_artifacts(branch, artifact_path=artifact_path)[0]
    checked_samples = len(
        preload_npy_samples(
            selected_records,
            artifact.input_spec,
            preload_dtype=np.dtype(artifact.input_spec.dtype),
        )
    )
    return {
        "branch": branch,
        "artifact_count": 1,
        "model_names": [artifact.model_name],
        "input_shapes": [list(artifact.input_spec.shape)],
        "output_shapes": [list(artifact.output_specs[0].shape)],
        "input_dtypes": [str(np.dtype(artifact.input_spec.dtype))],
        "output_dtypes": [str(np.dtype(artifact.output_specs[0].dtype))],
        "artifacts": [
            _build_artifact_entry(
                artifact,
                checked_samples=checked_samples,
                skip_complexity_precheck=skip_complexity_precheck,
            )
        ],
    }


def build_validation_report(
    *,
    branch: str | None,
    artifact_path: str | Path | None,
    data_dir: str | Path,
    split_manifest: str | Path,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    skip_complexity_precheck: bool = False,
) -> dict[str, object]:
    sample_limit = _validate_sample_limit(sample_limit)
    resolved_data_dir = resolve_cli_path(data_dir)
    resolved_manifest = resolve_cli_path(split_manifest)
    resolved_artifact_path = _validate_branch_and_artifact(branch, artifact_path)

    if not resolved_data_dir.exists():
        raise ValidationError(f"找不到数据目录: {resolved_data_dir}")
    if not resolved_data_dir.is_dir():
        raise ValidationError(f"数据目录不是目录: {resolved_data_dir}")

    manifest = load_manifest(resolved_manifest)
    split_counts = {
        "train": len(manifest["train_files"]),
        "val": len(manifest["val_files"]),
        "test": len(manifest["test_files"]),
    }
    test_records = build_test_records(manifest, resolved_data_dir)
    selected_records = _select_test_records(test_records, sample_limit)

    if branch is None:
        branch_reports = [
            _build_branch_report(
                branch_name,
                selected_records,
                skip_complexity_precheck=skip_complexity_precheck,
            )
            for branch_name in sorted(BRANCH_SPECS)
        ]
    else:
        if branch not in BRANCH_SPECS:
            raise ValidationError(f"不支持的推理分支: {branch}")
        branch_reports = [
            _build_single_artifact_report(
                branch,
                resolved_artifact_path,
                selected_records,
                skip_complexity_precheck=skip_complexity_precheck,
            )
            if resolved_artifact_path is not None
            else _build_branch_report(
                branch,
                selected_records,
                skip_complexity_precheck=skip_complexity_precheck,
            )
        ]

    return {
        "repo_root": str(REPO_ROOT),
        "manifest_path": to_repo_relative(resolved_manifest),
        "data_dir": to_repo_relative(resolved_data_dir),
        "class_count": len(manifest["class_names"]),
        "split_counts": split_counts,
        "sample_check_count": len(selected_records),
        "data_scan": _summarize_data_scan(manifest, resolved_data_dir),
        "branches": branch_reports,
    }
