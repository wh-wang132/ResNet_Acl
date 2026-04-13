from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .branch_specs import REPO_ROOT


@dataclass(frozen=True)
class SampleRecord:
    file_path: Path
    relative_path: str
    label_idx: int
    label_name: str


class ManifestError(RuntimeError):
    """split manifest 读取或校验失败。"""


SplitEntries = list[dict[str, object]]


def natural_sort_key(text: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_manifest(manifest_path: str | Path) -> dict[str, object]:
    resolved = resolve_repo_path(manifest_path)
    if not resolved.exists():
        raise ManifestError(f"找不到 split manifest: {resolved}")

    with resolved.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    required_keys = {
        "class_names",
        "class_to_idx",
        "train_files",
        "val_files",
        "test_files",
    }
    missing = sorted(required_keys.difference(manifest))
    if missing:
        raise ManifestError(f"split manifest 缺少字段: {missing}")
    return manifest


def _entry_to_record(entry: dict[str, object], data_dir: Path) -> SampleRecord:
    relative_path = str(entry["path"])
    return SampleRecord(
        file_path=data_dir / relative_path,
        relative_path=relative_path,
        label_idx=int(entry["label_idx"]),
        label_name=str(entry["label_name"]),
    )


def _build_records(entries: SplitEntries, data_dir: Path) -> list[SampleRecord]:
    return [_entry_to_record(entry, data_dir) for entry in entries]


def build_test_records(manifest: dict[str, object], data_dir: str | Path) -> list[SampleRecord]:
    resolved_data_dir = resolve_repo_path(data_dir)
    return _build_records(manifest["test_files"], resolved_data_dir)


def build_all_records(manifest: dict[str, object], data_dir: str | Path) -> list[SampleRecord]:
    resolved_data_dir = resolve_repo_path(data_dir)
    entries: SplitEntries = []
    for split_name in ("train_files", "val_files", "test_files"):
        entries.extend(manifest[split_name])
    return _build_records(entries, resolved_data_dir)


def scan_data_records(data_dir: str | Path, class_names: Iterable[str] | None = None) -> list[SampleRecord]:
    resolved_data_dir = resolve_repo_path(data_dir)
    if not resolved_data_dir.exists():
        raise ManifestError(f"找不到数据目录: {resolved_data_dir}")

    if class_names is None:
        classes = [item.name for item in resolved_data_dir.iterdir() if item.is_dir()]
        classes.sort(key=natural_sort_key)
    else:
        classes = list(class_names)

    class_to_idx = {label_name: idx for idx, label_name in enumerate(classes)}
    records: list[SampleRecord] = []
    for label_name in classes:
        class_dir = resolved_data_dir / label_name
        if not class_dir.is_dir():
            continue
        for file_path in sorted(class_dir.glob("*.npy"), key=lambda item: natural_sort_key(item.name)):
            records.append(
                SampleRecord(
                    file_path=file_path,
                    relative_path=str(file_path.relative_to(resolved_data_dir)),
                    label_idx=class_to_idx[label_name],
                    label_name=label_name,
                )
            )
    return records
