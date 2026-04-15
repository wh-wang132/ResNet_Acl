from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np

from src.common.acl_runner import ACLModelRunner
from src.common.args import parse_efficiency_args
from src.common.artifact_scanner import ArtifactRecord, resolve_artifact
from src.common.data import RuntimeInputAdapter, preload_npy_samples, validate_preloaded_sample_shapes
from src.common.manifest import build_all_records, load_manifest, scan_data_records
from src.common.metrics import LatencyMeter, argmax_predictions
from src.common.report import create_run_directory, to_repo_relative, write_json


class EfficiencyRunError(RuntimeError):
    """效率评测执行失败。"""


def main() -> None:
    args = parse_efficiency_args()
    artifact = resolve_artifact(args.branch, args.artifact_path)
    manifest = load_manifest(args.split_manifest)
    records = select_records(args, manifest)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise EfficiencyRunError("评测数据为空，无法执行效率评测")

    preloaded_samples = preload_npy_samples(records, artifact.input_spec)

    run_dir = create_run_directory(args.output_root, args.branch)
    run_efficiency_for_artifact(artifact, preloaded_samples, run_dir, args)
    print(to_repo_relative(run_dir / artifact.model_name / artifact.experiment_name / "summary.json"))


def select_records(args, manifest: dict[str, object]):
    if args.dataset_scope == "manifest_all":
        return build_all_records(manifest, args.data_dir)
    if args.dataset_scope == "data_all":
        class_names = [str(name) for name in manifest["class_names"]]
        return scan_data_records(args.data_dir, class_names=class_names)
    raise EfficiencyRunError(f"不支持的 dataset_scope: {args.dataset_scope}")


def run_efficiency_for_artifact(
    artifact: ArtifactRecord,
    preloaded_samples,
    run_dir: Path,
    args,
) -> dict[str, object]:
    artifact_dir = run_dir / artifact.model_name / artifact.experiment_name
    meter = LatencyMeter()

    with ACLModelRunner(artifact, device_id=args.device_id) as runner:
        validate_preloaded_sample_shapes(preloaded_samples, artifact.input_spec, runner.input_spec)
        input_adapter = RuntimeInputAdapter(runner.input_spec)
        model_input_spec = runner.input_spec
        model_output_spec = runner.output_specs[0]
        if args.warmup_steps > 0:
            warmup_cycle = itertools.cycle(preloaded_samples)
            for _ in range(args.warmup_steps):
                preloaded = next(warmup_cycle)
                sample = input_adapter.adapt(preloaded.input_fp16)
                runner.infer(sample)

        for _ in range(args.repeat):
            for preloaded in preloaded_samples:
                start_e2e = time.perf_counter()
                sample = input_adapter.adapt(preloaded.input_fp16)
                outputs, infer_ms = runner.infer_with_timing(sample)
                argmax_predictions(outputs[0], axis=1)
                end_to_end_ms = (time.perf_counter() - start_e2e) * 1000.0
                meter.record(infer_ms, end_to_end_ms)

    total_samples = len(preloaded_samples) * args.repeat

    summary = {
        "branch": artifact.branch,
        "model_name": artifact.model_name,
        "experiment_name": artifact.experiment_name,
        "artifact_path": to_repo_relative(artifact.model_path),
        "summary_path": to_repo_relative(artifact.summary_path),
        "dataset_scope": args.dataset_scope,
        "warmup_steps": args.warmup_steps,
        "repeat": args.repeat,
        "preload_dtype": str(np.dtype(np.float16)),
        "input_dtype": str(model_input_spec.dtype),
        "output_dtype": str(model_output_spec.dtype),
        "time_mode": args.time_mode,
    }
    summary.update(filter_summary_by_time_mode(meter.build_summary(samples=total_samples), args.time_mode))
    write_json(artifact_dir / "summary.json", summary)
    return summary


def filter_summary_by_time_mode(summary: dict[str, object], time_mode: str) -> dict[str, object]:
    if time_mode == "both":
        return summary
    if time_mode == "pure":
        return {key: value for key, value in summary.items() if not key.startswith("end_to_end_")}
    if time_mode == "end_to_end":
        return {
            key: value
            for key, value in summary.items()
            if not key.startswith("pure_infer_") and key not in {"avg_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms"}
        }
    raise EfficiencyRunError(f"不支持的 time_mode: {time_mode}")


if __name__ == "__main__":
    main()
