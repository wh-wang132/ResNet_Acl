from __future__ import annotations

from pathlib import Path

import numpy as np


EXPECTED_SAMPLE_SHAPE = (543, 512)
EXPECTED_BATCHED_SHAPE = (1, 1, 543, 512)


class DataError(RuntimeError):
    """样本读取或预处理失败。"""


def load_npy_sample(file_path: str | Path, dtype: np.dtype) -> np.ndarray:
    path = Path(file_path)
    if not path.exists():
        raise DataError(f"找不到样本文件: {path}")

    try:
        data = np.load(path)
    except Exception as exc:
        raise DataError(f"读取样本失败: {path}") from exc

    if not isinstance(data, np.ndarray):
        raise DataError(f"样本不是 numpy.ndarray: {path}")
    if tuple(data.shape) != EXPECTED_SAMPLE_SHAPE:
        raise DataError(
            f"样本形状不合法: path={path}, expected={EXPECTED_SAMPLE_SHAPE}, actual={tuple(data.shape)}"
        )

    casted = data.astype(dtype, copy=False)
    batched = np.expand_dims(np.expand_dims(casted, axis=0), axis=0)
    return np.ascontiguousarray(batched)
