from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, order=True)
class ArtifactKey:
    branch: str
    model_name: str
    experiment_name: str


@dataclass(frozen=True)
class ExperimentSpec:
    ratio: float | None
    steps: int | None
    pruning_mode: str | None
    finetune_epochs: int | None
    batch_size: int | None


@dataclass(frozen=True)
class AccuracyResult:
    key: ArtifactKey
    summary_path: Path
    payload: dict[str, Any]
    confusion_matrix_csv: Path | None
    confusion_matrix_png: Path | None
    per_class_metrics_csv: Path | None


@dataclass(frozen=True)
class EfficiencyResult:
    key: ArtifactKey
    summary_path: Path
    payload: dict[str, Any]


@dataclass(frozen=True)
class VisualizationRecord:
    key: ArtifactKey
    experiment: ExperimentSpec
    accuracy: AccuracyResult
    efficiency: EfficiencyResult
    om_size_bytes: int | None


@dataclass(frozen=True)
class MissingInput:
    key: ArtifactKey
    missing_accuracy: bool
    missing_efficiency: bool
    accuracy_summary_path: Path | None
    efficiency_summary_path: Path | None
