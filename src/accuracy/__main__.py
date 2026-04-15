from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.acl_runner import ACLConcurrentOrderedExecutor, ACLModelRunner
from src.common.args import parse_accuracy_args
from src.common.artifact_scanner import ArtifactRecord, resolve_artifact
from src.common.data import RuntimeInputAdapter, preload_npy_samples, validate_preloaded_sample_shapes
from src.common.manifest import build_test_records, load_manifest
from src.common.metrics import (
    ConfusionMatrixAccumulator,
    argmax_predictions,
    cross_entropy_from_logits,
    softmax_np,
)
from src.common.report import (
    create_run_directory,
    plot_confusion_matrix,
    to_repo_relative,
    write_confusion_matrix_csv,
    write_csv,
    write_json,
)


class AccuracyRunError(RuntimeError):
    """精度评测执行失败。"""


def main() -> None:
    args = parse_accuracy_args()
    artifact = resolve_artifact(args.branch, args.artifact_path)
    manifest = load_manifest(args.split_manifest)
    class_names = [str(name) for name in manifest["class_names"]]
    records = build_test_records(manifest, args.data_dir)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise AccuracyRunError("test 集为空，无法执行精度评测")

    preloaded_samples = preload_npy_samples(records, artifact.input_spec)

    run_dir = create_run_directory(args.output_root, args.branch)
    run_accuracy_for_artifact(artifact, preloaded_samples, class_names, run_dir, args.device_id, args)
    print(to_repo_relative(run_dir / artifact.model_name / artifact.experiment_name / "summary.json"))


def run_accuracy_for_artifact(
    artifact: ArtifactRecord,
    preloaded_samples,
    class_names: list[str],
    run_dir: Path,
    device_id: int,
    args,
) -> dict[str, object]:
    artifact_dir = run_dir / artifact.model_name / artifact.experiment_name
    confusion = ConfusionMatrixAccumulator(num_classes=len(class_names))
    total_loss = 0.0
    total_samples = 0
    active_instances = 1

    if args.num_instances == 1:
        with ACLModelRunner(artifact, device_id=device_id) as runner:
            validate_preloaded_sample_shapes(preloaded_samples, artifact.input_spec, runner.input_spec)
            input_adapter = RuntimeInputAdapter(runner.input_spec)
            model_input_spec = runner.input_spec
            model_output_spec = runner.output_specs[0]
            for preloaded in preloaded_samples:
                sample = input_adapter.adapt(preloaded.input_fp16)
                outputs = runner.infer(sample)
                logits = outputs[0]
                probabilities = softmax_np(logits, axis=1)
                predictions = argmax_predictions(probabilities, axis=1)
                labels = np.asarray([preloaded.record.label_idx], dtype=np.int64)
                losses = cross_entropy_from_logits(logits.astype(np.float64, copy=False), labels)
                confusion.update(predictions, labels)
                total_loss += float(losses.sum())
                total_samples += int(labels.shape[0])
    else:
        executor = ACLConcurrentOrderedExecutor(
            artifact,
            device_id=device_id,
            num_instances=args.num_instances,
            buffer_depth=args.buffer_depth,
        )
        for result in executor.iter_ordered(preloaded_samples):
            logits = result.outputs[0]
            probabilities = softmax_np(logits, axis=1)
            predictions = argmax_predictions(probabilities, axis=1)
            labels = np.asarray([result.preloaded.record.label_idx], dtype=np.int64)
            losses = cross_entropy_from_logits(logits.astype(np.float64, copy=False), labels)
            confusion.update(predictions, labels)
            total_loss += float(losses.sum())
            total_samples += int(labels.shape[0])
        active_instances = executor.active_instances
        model_input_spec = executor.input_spec
        model_output_spec = executor.output_specs[0]

    summary = {
        "branch": artifact.branch,
        "model_name": artifact.model_name,
        "experiment_name": artifact.experiment_name,
        "artifact_path": to_repo_relative(artifact.model_path),
        "summary_path": to_repo_relative(artifact.summary_path),
        "samples": total_samples,
        "accuracy": confusion.accuracy(),
        "avg_loss": float(total_loss / total_samples) if total_samples else 0.0,
        "input_name": model_input_spec.name,
        "input_shape": list(model_input_spec.shape),
        "input_dtype": str(model_input_spec.dtype),
        "output_name": model_output_spec.name,
        "output_shape": list(model_output_spec.shape),
        "output_dtype": str(model_output_spec.dtype),
        "execution_mode": "serial" if args.num_instances == 1 else "multi_instance_concurrent",
        "num_instances": args.num_instances,
        "active_instances": active_instances,
        "buffer_depth": args.buffer_depth,
        "dispatch_policy": "serial" if args.num_instances == 1 else "modulo_round_robin",
        "ordering_policy": "serial" if args.num_instances == 1 else "strict_modulo_roundtrip",
        "confusion_matrix_csv": to_repo_relative(
            write_confusion_matrix_csv(artifact_dir / "confusion_matrix.csv", confusion.matrix, class_names)
        ),
    }
    if args.save_confusion_matrix:
        summary["confusion_matrix_png"] = to_repo_relative(
            plot_confusion_matrix(artifact_dir / "confusion_matrix.png", confusion.matrix, class_names)
        )
    if args.save_per_class_metrics:
        per_class_path = write_csv(
            artifact_dir / "per_class_metrics.csv",
            confusion.per_class_metrics(class_names),
        )
        summary["per_class_metrics_csv"] = to_repo_relative(per_class_path)
    write_json(artifact_dir / "summary.json", summary)
    return summary


if __name__ == "__main__":
    main()
