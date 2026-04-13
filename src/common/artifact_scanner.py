from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .branch_specs import BranchSpec, get_branch_spec


ACL_ELEM_TYPE_TO_DTYPE: dict[int, np.dtype] = {
    1: np.dtype(np.float32),
    10: np.dtype(np.float16),
}


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    elem_type: int


@dataclass(frozen=True)
class ArtifactRecord:
    branch: str
    model_name: str
    experiment_name: str
    artifact_dir: Path
    model_path: Path
    summary_path: Path
    input_spec: TensorSpec
    output_specs: tuple[TensorSpec, ...]
    summary: dict[str, Any]


class ArtifactScanError(RuntimeError):
    """模型扫描或摘要校验失败。"""


def _resolve_static_shape(shape_like: list[Any], *, batch_value: int = 1) -> tuple[int, ...]:
    resolved: list[int] = []
    for dim in shape_like:
        if dim == "batch":
            resolved.append(batch_value)
            continue
        if not isinstance(dim, int):
            raise ArtifactScanError(f"发现不支持的维度定义: {shape_like}")
        resolved.append(dim)
    return tuple(resolved)


def _resolve_dtype(elem_type: int) -> np.dtype:
    try:
        return ACL_ELEM_TYPE_TO_DTYPE[int(elem_type)]
    except KeyError as exc:
        raise ArtifactScanError(f"不支持的 ACL elem_type: {elem_type}") from exc


def _load_summary(summary_path: Path) -> dict[str, Any]:
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_artifact(model_path: Path, spec: BranchSpec) -> ArtifactRecord:
    artifact_dir = model_path.parent
    summary_path = artifact_dir / spec.summary_filename
    if not summary_path.exists():
        raise ArtifactScanError(f"找不到摘要文件: {summary_path}")

    summary = _load_summary(summary_path)
    if summary.get("stage") != "atc":
        raise ArtifactScanError(f"摘要 stage 非 atc: {summary_path}")
    if summary.get("branch") != spec.branch:
        raise ArtifactScanError(
            f"摘要 branch 与参数不一致: expected={spec.branch}, actual={summary.get('branch')}"
        )

    source_interface = summary.get("source_interface")
    if not isinstance(source_interface, dict):
        raise ArtifactScanError(f"摘要缺少 source_interface: {summary_path}")

    input_name = str(source_interface.get("input_name", "input"))
    input_elem_type = int(source_interface["input_elem_type"])
    input_dtype = _resolve_dtype(input_elem_type)
    source_input_shape = _resolve_static_shape(source_interface["input_shape"])
    resolved_input_shape = tuple(int(dim) for dim in summary["resolved_input_shape"])
    if source_input_shape != resolved_input_shape:
        raise ArtifactScanError(
            f"source_interface.input_shape 与 resolved_input_shape 不一致: {summary_path}"
        )

    output_name = str(source_interface.get("output_name", "output"))
    output_elem_type = int(source_interface["output_elem_type"])
    output_dtype = _resolve_dtype(output_elem_type)
    output_shape = _resolve_static_shape(source_interface["output_shape"])

    input_spec = TensorSpec(
        name=input_name,
        shape=resolved_input_shape,
        dtype=input_dtype,
        elem_type=input_elem_type,
    )
    output_specs = (
        TensorSpec(
            name=output_name,
            shape=output_shape,
            dtype=output_dtype,
            elem_type=output_elem_type,
        ),
    )

    return ArtifactRecord(
        branch=spec.branch,
        model_name=artifact_dir.parent.name,
        experiment_name=artifact_dir.name,
        artifact_dir=artifact_dir,
        model_path=model_path,
        summary_path=summary_path,
        input_spec=input_spec,
        output_specs=output_specs,
        summary=summary,
    )


def scan_branch_artifacts(branch: str, scan_root: str | Path | None = None) -> list[ArtifactRecord]:
    spec = get_branch_spec(branch)
    root = spec.scan_root if scan_root is None else Path(scan_root)
    if not root.exists():
        raise ArtifactScanError(f"找不到扫描目录: {root}")

    model_paths = sorted(root.rglob(spec.model_filename))
    if not model_paths:
        raise ArtifactScanError(f"在 {root} 下未找到 {spec.model_filename}")
    return [_validate_artifact(model_path, spec) for model_path in model_paths]


def resolve_artifacts(
    branch: str,
    artifact_path: str | Path | None = None,
    scan_root: str | Path | None = None,
) -> list[ArtifactRecord]:
    spec = get_branch_spec(branch)
    if artifact_path is None:
        return scan_branch_artifacts(branch, scan_root=scan_root)

    candidate = Path(artifact_path)
    model_path = candidate / spec.model_filename if candidate.is_dir() else candidate
    if model_path.name != spec.model_filename:
        raise ArtifactScanError(
            f"当前分支 {branch} 只接受 {spec.model_filename}，实际为 {model_path.name}"
        )
    if not model_path.exists():
        raise ArtifactScanError(f"找不到模型文件: {model_path}")
    return [_validate_artifact(model_path, spec)]
