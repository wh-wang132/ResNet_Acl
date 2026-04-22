from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np

from src.common.args import parse_efficiency_args
from src.common.artifact_scanner import ArtifactRecord, ArtifactScanError, resolve_artifact
from src.common.data import DataError, RuntimeInputAdapter, preload_npy_samples, validate_preloaded_sample_shapes
from src.common.manifest import ManifestError, build_test_records, load_manifest
from src.common.metrics import LatencyMeter, argmax_predictions
from src.common.model_complexity import ModelComplexityError, collect_model_complexity
from src.common.report import create_run_directory, to_repo_relative, write_json


class EfficiencyRunError(RuntimeError):
    """效率评测执行失败。"""


def build_summary_filename(num_instances: int, buffer_depth: int) -> str:
    return f"summary__instances{int(num_instances)}_buffer{int(buffer_depth)}.json"


def main() -> None:
    try:
        args = parse_efficiency_args()
        artifact = resolve_artifact(args.branch, args.artifact_path)
        manifest = load_manifest(args.split_manifest)
        records = build_test_records(manifest, args.data_dir)
        if args.limit is not None:
            records = records[: args.limit]
        if not records:
            raise EfficiencyRunError("评测数据为空，无法执行效率评测")

        preload_dtype = np.dtype(artifact.input_spec.dtype)
        preloaded_samples = preload_npy_samples(records, artifact.input_spec, preload_dtype=preload_dtype)

        run_dir = create_run_directory(args.output_root, args.branch)
        run_efficiency_for_artifact(artifact, preloaded_samples, run_dir, args)
        print(
            to_repo_relative(
                run_dir
                / artifact.model_name
                / artifact.experiment_name
                / build_summary_filename(args.num_instances, args.buffer_depth)
            )
        )
    except (EfficiencyRunError, ArtifactScanError, ManifestError, DataError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


def run_efficiency_for_artifact(
    artifact: ArtifactRecord,
    preloaded_samples,
    run_dir: Path,
    args,
) -> dict[str, object]:
    try:
        from src.common.acl_runner import ACLConcurrentOrderedExecutor, ACLModelRunner
    except ImportError as exc:
        raise EfficiencyRunError(
            "导入 ACL 运行时失败，请先确认 Ascend CANN/ACL 环境已正确安装并可被当前 Python 环境加载"
        ) from exc

    artifact_dir = run_dir / artifact.model_name / artifact.experiment_name
    summary_file_path = artifact_dir / build_summary_filename(args.num_instances, args.buffer_depth)
    try:
        complexity = collect_model_complexity(artifact)
    except (ArtifactScanError, ModelComplexityError) as exc:
        raise EfficiencyRunError(str(exc)) from exc
    meter = LatencyMeter()
    h2d_stage_total_ms = 0.0
    h2d_memcpy_total_ms = 0.0
    execute_wait_total_ms = 0.0
    d2h_memcpy_total_ms = 0.0
    output_decode_total_ms = 0.0
    dispatch_block_total_ms = 0.0
    collect_wait_total_ms = 0.0
    total_samples = len(preloaded_samples) * args.repeat
    active_instances = 1
    preload_dtype = np.dtype(artifact.input_spec.dtype)
    timing_start_stage = "post_preload_validation_warmup"
    pure_infer_wall_total_ms: float | None = None
    end_to_end_wall_total_ms: float | None = None
    timing_started_ns: int | None = None
    timing_finished_ns: int | None = None

    if args.num_instances == 1:
        with ACLModelRunner(artifact, device_id=args.device_id) as runner:
            validate_preloaded_sample_shapes(preloaded_samples, artifact.input_spec, runner.input_spec)
            input_adapter = RuntimeInputAdapter(runner.input_spec)
            model_input_spec = runner.input_spec
            model_output_spec = runner.output_specs[0]
            if args.warmup_steps > 0:
                warmup_cycle = itertools.cycle(preloaded_samples)
                for _ in range(args.warmup_steps):
                    preloaded = next(warmup_cycle)
                    sample = input_adapter.adapt(preloaded.input_tensor)
                    runner.infer(sample)

            timing_started_ns = time.perf_counter_ns()
            for _ in range(args.repeat):
                for preloaded in preloaded_samples:
                    start_e2e = time.perf_counter()
                    sample = input_adapter.adapt(preloaded.input_tensor)
                    outputs, infer_ms, timing = runner.infer_with_breakdown(sample)
                    argmax_predictions(outputs[0], axis=1)
                    end_to_end_ms = (time.perf_counter() - start_e2e) * 1000.0
                    meter.record(infer_ms, end_to_end_ms)
                    h2d_stage_total_ms += timing.h2d_stage_ms
                    h2d_memcpy_total_ms += timing.h2d_memcpy_ms
                    execute_wait_total_ms += timing.execute_wait_ms
                    d2h_memcpy_total_ms += timing.d2h_memcpy_ms
                    output_decode_total_ms += timing.output_decode_ms
            timing_finished_ns = time.perf_counter_ns()
    else:
        executor = ACLConcurrentOrderedExecutor(
            artifact,
            device_id=args.device_id,
            num_instances=args.num_instances,
            buffer_depth=args.buffer_depth,
        )
        for result in executor.iter_ordered(
            preloaded_samples,
            repeat=args.repeat,
            warmup_steps=args.warmup_steps,
        ):
            argmax_predictions(result.outputs[0], axis=1)
            end_to_end_ms = (time.perf_counter_ns() - result.dispatch_started_ns) / 1_000_000.0
            meter.record(result.pure_infer_ms, end_to_end_ms)
            h2d_stage_total_ms += result.timing.h2d_stage_ms
            h2d_memcpy_total_ms += result.timing.h2d_memcpy_ms
            execute_wait_total_ms += result.timing.execute_wait_ms
            d2h_memcpy_total_ms += result.timing.d2h_memcpy_ms
            output_decode_total_ms += result.timing.output_decode_ms
        active_instances = executor.active_instances
        model_input_spec = executor.input_spec
        model_output_spec = executor.output_specs[0]
        dispatch_block_total_ms = executor.dispatch_block_total_ms
        collect_wait_total_ms = executor.collect_wait_total_ms
        if executor.first_execute_started_ns is not None and executor.last_execute_finished_ns is not None:
            pure_infer_wall_total_ms = (
                executor.last_execute_finished_ns - executor.first_execute_started_ns
            ) / 1_000_000.0
        if executor.run_started_ns is not None:
            timing_started_ns = executor.run_started_ns
            timing_finished_ns = time.perf_counter_ns()
            end_to_end_wall_total_ms = (timing_finished_ns - timing_started_ns) / 1_000_000.0

    if args.num_instances == 1 and timing_started_ns is not None and timing_finished_ns is not None:
        end_to_end_wall_total_ms = (timing_finished_ns - timing_started_ns) / 1_000_000.0

    summary = {
        "branch": artifact.branch,
        "model_name": artifact.model_name,
        "experiment_name": artifact.experiment_name,
        "artifact_path": to_repo_relative(artifact.model_path),
        "summary_path": to_repo_relative(artifact.summary_path),
        "parameter_count": complexity.parameter_count,
        "operation_count": complexity.operation_count,
        "operation_count_unit": complexity.operation_count_unit,
        "operation_count_scope": complexity.operation_count_scope,
        "operation_count_included_op_types": list(complexity.operation_count_included_op_types),
        "operation_count_excluded_op_types": list(complexity.operation_count_excluded_op_types),
        "dataset_scope": "manifest_test",
        "timing_start_stage": timing_start_stage,
        "timing_excludes": ["preload", "validation", "warmup"],
        "pure_infer_scope": "model_execute_only",
        "end_to_end_scope": "dispatch_to_postprocess",
        "warmup_steps": args.warmup_steps,
        "warmup_scope": "global" if args.num_instances == 1 else "per_instance",
        "repeat": args.repeat,
        "preload_dtype": str(preload_dtype),
        "input_dtype": str(model_input_spec.dtype),
        "output_dtype": str(model_output_spec.dtype),
        "execution_mode": "serial" if args.num_instances == 1 else "multi_instance_concurrent",
        "num_instances": args.num_instances,
        "active_instances": active_instances,
        "buffer_depth": args.buffer_depth,
        "dispatch_policy": "serial" if args.num_instances == 1 else "modulo_round_robin",
        "ordering_policy": "serial" if args.num_instances == 1 else "strict_modulo_roundtrip",
        "time_mode": args.time_mode,
        "dispatch_block_total_ms": dispatch_block_total_ms,
        "collect_wait_total_ms": collect_wait_total_ms,
        "h2d_stage_total_ms": h2d_stage_total_ms,
        "h2d_memcpy_total_ms": h2d_memcpy_total_ms,
        "execute_wait_total_ms": execute_wait_total_ms,
        "d2h_memcpy_total_ms": d2h_memcpy_total_ms,
        "output_decode_total_ms": output_decode_total_ms,
    }
    summary.update(
        filter_summary_by_time_mode(
            meter.build_summary(
                samples=total_samples,
                pure_infer_wall_total_ms=pure_infer_wall_total_ms,
                end_to_end_wall_total_ms=end_to_end_wall_total_ms,
            ),
            args.time_mode,
        )
    )
    write_json(summary_file_path, summary)
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
