from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .artifact_scanner import TensorSpec
from .manifest import SampleRecord

MAX_SHAPE_VALIDATION_SAMPLES = 32


class DataError(RuntimeError):
    """样本读取或预处理失败。"""


@dataclass(frozen=True)
class PreloadedSample:
    record: SampleRecord
    input_fp16: np.ndarray


class RuntimeInputAdapter:
    def __init__(self, input_spec: TensorSpec):
        self._shape = tuple(input_spec.shape)
        self._target_dtype = np.dtype(input_spec.dtype)
        self._fp32_buffer = np.empty(self._shape, dtype=np.float32) if self._target_dtype == np.float32 else None

    def adapt(self, input_fp16: np.ndarray) -> np.ndarray:
        if tuple(input_fp16.shape) != self._shape:
            raise DataError(f"预加载输入 shape 不匹配: expected={self._shape}, actual={tuple(input_fp16.shape)}")
        if np.dtype(input_fp16.dtype) != np.float16:
            raise DataError(f"预加载输入 dtype 必须为 float16，实际为 {input_fp16.dtype}")
        if not input_fp16.flags.c_contiguous:
            raise DataError("预加载输入数组必须是 C contiguous")

        if self._target_dtype == np.float16:
            return input_fp16
        if self._target_dtype == np.float32:
            assert self._fp32_buffer is not None
            np.copyto(self._fp32_buffer, input_fp16)
            return self._fp32_buffer
        raise DataError(f"不支持的目标输入 dtype: {self._target_dtype}")


def _load_npy_array(file_path: str | Path) -> np.ndarray:
    path = Path(file_path)
    if not path.exists():
        raise DataError(f"找不到样本文件: {path}")

    try:
        data = np.load(path)
    except Exception as exc:
        raise DataError(f"读取样本失败: {path}") from exc

    if not isinstance(data, np.ndarray):
        raise DataError(f"样本不是 numpy.ndarray: {path}")
    return data


def _resolve_preloaded_shape(
    sample_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    *,
    file_path: str | Path | None = None,
) -> tuple[int, ...]:
    if sample_shape == target_shape:
        return target_shape

    rank_diff = len(target_shape) - len(sample_shape)
    if rank_diff >= 0 and sample_shape == target_shape[rank_diff:] and all(dim == 1 for dim in target_shape[:rank_diff]):
        return target_shape

    location = f"path={file_path}, " if file_path is not None else ""
    raise DataError(
        "样本形状无法通过预加载阶段补前导 1 维匹配摘要输入: "
        f"{location}raw_shape={sample_shape}, target_shape={target_shape}"
    )


def prepare_sample_array(
    data: np.ndarray,
    target_shape: tuple[int, ...],
    dtype: np.dtype,
    *,
    file_path: str | Path | None = None,
) -> np.ndarray:
    normalized_shape = _resolve_preloaded_shape(tuple(data.shape), tuple(target_shape), file_path=file_path)
    casted = data.astype(np.dtype(dtype), copy=False)
    normalized = casted.reshape(normalized_shape)
    return np.ascontiguousarray(normalized)


def load_npy_sample(file_path: str | Path, target_shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    data = _load_npy_array(file_path)
    return prepare_sample_array(data, target_shape, dtype, file_path=file_path)


def preload_npy_samples(
    records: Iterable[SampleRecord],
    input_spec: TensorSpec,
    *,
    preload_dtype: np.dtype = np.float16,
) -> list[PreloadedSample]:
    target_dtype = np.dtype(preload_dtype)
    target_shape = tuple(input_spec.shape)
    return [
        PreloadedSample(
            record=record,
            input_fp16=load_npy_sample(record.file_path, target_shape, target_dtype),
        )
        for record in records
    ]


def _select_validation_sample_indices(total_samples: int, max_samples: int) -> list[int]:
    if total_samples <= 0:
        return []
    if total_samples <= max_samples:
        return list(range(total_samples))
    if max_samples <= 1:
        return [0]
    return sorted({round(index * (total_samples - 1) / (max_samples - 1)) for index in range(max_samples)})


def validate_preloaded_sample_shapes(
    preloaded_samples: list[PreloadedSample],
    summary_input_spec: TensorSpec,
    model_input_spec: TensorSpec,
    *,
    max_samples: int = MAX_SHAPE_VALIDATION_SAMPLES,
) -> None:
    if not preloaded_samples:
        raise DataError("预加载样本为空，无法执行推理前 shape 抽样校验")

    summary_shape = tuple(summary_input_spec.shape)
    model_shape = tuple(model_input_spec.shape)
    sampled_indices = _select_validation_sample_indices(len(preloaded_samples), max_samples)

    for sample_index in sampled_indices:
        preloaded = preloaded_samples[sample_index]
        sample_shape = tuple(preloaded.input_fp16.shape)
        if sample_shape != summary_shape or sample_shape != model_shape:
            raise DataError(
                "推理前 shape 三方校验失败: "
                f"path={preloaded.record.file_path}, sample_shape={sample_shape}, "
                f"summary_shape={summary_shape}, om_shape={model_shape}"
            )
