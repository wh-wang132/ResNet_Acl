from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .artifact_scanner import ArtifactRecord, TensorSpec, resolve_artifact
from .data import (
    PreloadedSample,
    RuntimeInputAdapter,
    preload_npy_samples,
    validate_preloaded_sample_shapes,
)
from .manifest import build_test_records, load_manifest


@dataclass(frozen=True)
class PreparedArtifactTestRun:
    artifact: ArtifactRecord
    class_names: list[str]
    preloaded_samples: list[PreloadedSample]
    preload_dtype: np.dtype


@dataclass(frozen=True)
class SerialRunnerBootstrap:
    input_adapter: RuntimeInputAdapter
    model_input_spec: TensorSpec
    model_output_spec: TensorSpec


def prepare_artifact_test_run(
    *,
    branch: str,
    artifact_path: str | Path,
    split_manifest: str | Path,
    data_dir: str | Path,
    limit: int | None,
    empty_records_message: str,
    empty_records_error_factory: Callable[[str], Exception],
) -> PreparedArtifactTestRun:
    artifact = resolve_artifact(branch, artifact_path)
    manifest = load_manifest(split_manifest)
    class_names = [str(name) for name in manifest["class_names"]]
    records = build_test_records(manifest, data_dir)
    if limit is not None:
        records = records[:limit]
    if not records:
        raise empty_records_error_factory(empty_records_message)

    preload_dtype = np.dtype(artifact.input_spec.dtype)
    preloaded_samples = preload_npy_samples(records, artifact.input_spec, preload_dtype=preload_dtype)
    return PreparedArtifactTestRun(
        artifact=artifact,
        class_names=class_names,
        preloaded_samples=preloaded_samples,
        preload_dtype=preload_dtype,
    )


def bootstrap_serial_runner(
    artifact: ArtifactRecord,
    preloaded_samples: list[PreloadedSample],
    runner: Any,
) -> SerialRunnerBootstrap:
    validate_preloaded_sample_shapes(preloaded_samples, artifact.input_spec, runner.input_spec)
    model_input_spec = runner.input_spec
    model_output_spec = runner.output_specs[0]
    return SerialRunnerBootstrap(
        input_adapter=RuntimeInputAdapter(model_input_spec),
        model_input_spec=model_input_spec,
        model_output_spec=model_output_spec,
    )
