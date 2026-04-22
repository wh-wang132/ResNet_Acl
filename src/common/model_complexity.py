from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifact_scanner import ArtifactRecord


CONV_OP_TYPES = frozenset({"Conv", "Conv2D", "Convolution"})
MATMUL_OP_TYPES = frozenset({"Gemm", "MatMul", "MatMulV2"})
IGNORED_OP_TYPES = frozenset(
    {
        "Add",
        "Adds",
        "AntiQuant",
        "AscendAntiQuant",
        "AscendDequant",
        "AscendQuant",
        "BiasAdd",
        "Cast",
        "Const",
        "Data",
        "Dequant",
        "Flatten",
        "GlobalAveragePool",
        "MaxPool",
        "MaxPoolV3",
        "Mul",
        "NetOutput",
        "Quant",
        "ReduceMeanD",
        "Relu",
        "TransData",
    }
)


class ModelComplexityError(RuntimeError):
    """模型复杂度提取或估算失败。"""


@dataclass(frozen=True)
class ModelComplexity:
    parameter_count: int
    operation_count: int
    operation_count_unit: str
    operation_count_scope: str
    operation_count_included_op_types: tuple[str, ...]
    operation_count_excluded_op_types: tuple[str, ...]


def collect_model_complexity(artifact: ArtifactRecord) -> ModelComplexity:
    operation_count, included_op_types, excluded_op_types = estimate_macs_from_om(artifact.model_path)
    return ModelComplexity(
        parameter_count=artifact.parameter_count,
        operation_count=operation_count,
        operation_count_unit="MACs",
        operation_count_scope="per_forward_pass",
        operation_count_included_op_types=included_op_types,
        operation_count_excluded_op_types=excluded_op_types,
    )


def estimate_macs_from_om(model_path: str | Path) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    payload = _load_om_json(model_path)
    total_macs = 0
    included_op_types: set[str] = set()
    excluded_op_types: set[str] = set()
    unknown_op_types: set[str] = set()

    for graph in payload.get("graph", []):
        if not isinstance(graph, dict):
            continue
        for op in graph.get("op", []):
            if not isinstance(op, dict):
                continue
            op_type = str(op.get("type", ""))
            if not op_type:
                continue
            if op_type in CONV_OP_TYPES:
                total_macs += _estimate_conv_macs(op)
                included_op_types.add(op_type)
                continue
            if op_type in MATMUL_OP_TYPES:
                total_macs += _estimate_matmul_macs(op)
                included_op_types.add(op_type)
                continue
            if op_type in IGNORED_OP_TYPES:
                excluded_op_types.add(op_type)
                continue
            unknown_op_types.add(op_type)

    if unknown_op_types:
        raise ModelComplexityError(
            f"发现未分类 OM op type: {sorted(unknown_op_types)}，请先定义其是否参与 MACs 统计"
        )
    return total_macs, tuple(sorted(included_op_types)), tuple(sorted(excluded_op_types))


def _load_om_json(model_path: str | Path) -> dict[str, Any]:
    atc = shutil.which("atc")
    if atc is None:
        raise ModelComplexityError("找不到 atc 命令，请先加载 Ascend ATC 环境后再执行效率评测")

    model_path = Path(model_path).resolve()
    with tempfile.TemporaryDirectory(prefix="om_json_", dir="/tmp") as tmp_dir:
        json_path = Path(tmp_dir) / "model.json"
        process = subprocess.run(
            [atc, "--mode=1", f"--om={model_path}", f"--json={json_path}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            output = _tail_output(process.stdout)
            raise ModelComplexityError(
                f"atc --mode=1 解析 OM 失败: {model_path}\n{output}"
            )
        if not json_path.exists():
            raise ModelComplexityError(
                f"atc --mode=1 未生成 JSON 文件: {json_path}"
            )
        with json_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


def _tail_output(output: str, *, max_lines: int = 40) -> str:
    lines = [line for line in output.strip().splitlines() if line.strip()]
    if not lines:
        return "<empty output>"
    return "\n".join(lines[-max_lines:])


def _estimate_conv_macs(op: dict[str, Any]) -> int:
    input_desc = op.get("input_desc", [])
    output_desc = op.get("output_desc", [])
    input_shape = _require_origin_shape(op, input_desc, 0)
    weight_shape = _require_origin_shape(op, input_desc, 1)
    output_shape = _require_origin_shape(op, output_desc, 0)

    if len(input_shape) != 4 or len(weight_shape) != 4 or len(output_shape) != 4:
        raise ModelComplexityError(
            f"Conv op shape 维度异常: op={op.get('name')} input={input_shape} weight={weight_shape} output={output_shape}"
        )

    out_n, out_c, out_h, out_w = output_shape
    weight_out_c, weight_in_c, kernel_h, kernel_w = weight_shape
    if out_c != weight_out_c:
        raise ModelComplexityError(
            f"Conv 输出通道与权重不一致: op={op.get('name')} output_c={out_c} weight_c={weight_out_c}"
        )
    return int(out_n * out_c * out_h * out_w * weight_in_c * kernel_h * kernel_w)


def _estimate_matmul_macs(op: dict[str, Any]) -> int:
    input_desc = op.get("input_desc", [])
    output_desc = op.get("output_desc", [])
    input_shape = _require_origin_shape(op, input_desc, 0)
    weight_shape = _require_origin_shape(op, input_desc, 1)
    output_shape = _require_origin_shape(op, output_desc, 0)

    if len(input_shape) < 2 or len(weight_shape) < 2 or len(output_shape) < 2:
        raise ModelComplexityError(
            f"MatMul op shape 维度异常: op={op.get('name')} input={input_shape} weight={weight_shape} output={output_shape}"
        )

    reduction_dim = input_shape[-1]
    output_dim = output_shape[-1]
    if reduction_dim not in weight_shape[-2:] or output_dim not in weight_shape[-2:]:
        raise ModelComplexityError(
            f"MatMul 权重 shape 与输入/输出不一致: op={op.get('name')} input={input_shape} weight={weight_shape} output={output_shape}"
        )
    return int(math.prod(output_shape) * reduction_dim)


def _require_origin_shape(op: dict[str, Any], descs: list[dict[str, Any]], index: int) -> tuple[int, ...]:
    try:
        desc = descs[index]
    except IndexError as exc:
        raise ModelComplexityError(
            f"OM op 缺少 desc: op={op.get('name')} index={index}"
        ) from exc

    origin_shape = _extract_origin_shape(desc)
    if origin_shape is None:
        raise ModelComplexityError(
            f"OM op 缺少 origin_shape: op={op.get('name')} index={index}"
        )
    return origin_shape


def _extract_origin_shape(desc: dict[str, Any]) -> tuple[int, ...] | None:
    for attr in desc.get("attr", []):
        if attr.get("key") != "origin_shape":
            continue
        value = attr.get("value")
        if not isinstance(value, dict):
            continue
        shape_like = value.get("list", {}).get("i")
        if not isinstance(shape_like, list):
            continue
        return tuple(int(dim) for dim in shape_like)

    shape_like = desc.get("shape", {}).get("dim")
    if isinstance(shape_like, list):
        return tuple(int(dim) for dim in shape_like)
    return None
