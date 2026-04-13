from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Times New Roman"]
matplotlib.rcParams["font.sans-serif"] = ["Times New Roman"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from .branch_specs import REPO_ROOT


def ensure_directory(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def create_run_directory(output_root: str | Path, branch: str) -> Path:
    root = resolve_repo_path(output_root)
    return ensure_directory(root / branch)


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def to_repo_relative(path_like: str | Path) -> str:
    path = Path(path_like).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_json(file_path: str | Path, payload: dict | list) -> Path:
    resolved = Path(file_path)
    ensure_directory(resolved.parent)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return resolved


def write_csv(file_path: str | Path, rows: Iterable[dict[str, object]]) -> Path:
    resolved = Path(file_path)
    rows = list(rows)
    ensure_directory(resolved.parent)
    if not rows:
        resolved.write_text("", encoding="utf-8")
        return resolved

    fieldnames = list(rows[0].keys())
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return resolved


def write_confusion_matrix_csv(
    file_path: str | Path,
    matrix: np.ndarray,
    class_names: list[str],
) -> Path:
    rows = []
    for row_idx, predicted_label in enumerate(class_names):
        row = {"predicted\\true": predicted_label}
        for col_idx, true_label in enumerate(class_names):
            row[true_label] = int(matrix[row_idx, col_idx])
        rows.append(row)
    return write_csv(file_path, rows)


def plot_confusion_matrix(
    file_path: str | Path,
    matrix: np.ndarray,
    class_names: list[str],
) -> Path:
    resolved = Path(file_path)
    ensure_directory(resolved.parent)
    plt.figure(figsize=(20, 16))
    plt.imshow(matrix, cmap=cm.get_cmap("Blues"))
    plt.xticks(range(len(class_names)), class_names, rotation=45, fontsize=28)
    plt.yticks(range(len(class_names)), class_names, fontsize=28)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=24)
    plt.xlabel("True Labels", fontsize=32)
    plt.ylabel("Predicted Labels", fontsize=32)
    threshold = matrix.max() / 2 if matrix.size else 0
    for x in range(matrix.shape[1]):
        for y in range(matrix.shape[0]):
            value = int(matrix[y, x])
            plt.text(
                x,
                y,
                str(value),
                va="center",
                ha="center",
                color="white" if value > threshold else "black",
                fontsize=24,
            )
    plt.tight_layout()
    plt.savefig(resolved, dpi=100, bbox_inches="tight")
    plt.close()
    return resolved
