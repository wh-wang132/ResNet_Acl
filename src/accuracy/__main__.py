from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.acl_runner import ACLModelRunner, ACLRuntimeError
from src.common.args import parse_accuracy_args
from src.common.artifact_scanner import ArtifactRecord, resolve_artifacts
from src.common.data import load_npy_sample
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
    artifacts = resolve_artifacts(args.branch, artifact_path=args.artifact_path, scan_root=args.scan_root)
    manifest = load_manifest(args.split_manifest)
    class_names = [str(name) for name in manifest["class_names"]]
    records = build_test_records(manifest, args.data_dir)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise AccuracyRunError("test 集为空，无法执行精度评测")

    run_dir = create_run_directory(args.output_root, args.branch)
    summary_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for artifact in artifacts:
        try:
            summary_rows.append(run_accuracy_for_artifact(artifact, records, class_names, run_dir, args.device_id, args))
        except Exception as exc:
            if args.fail_fast:
                raise
            failures.append(
                {
                    "model_name": artifact.model_name,
                    "experiment_name": artifact.experiment_name,
                    "artifact_path": to_repo_relative(artifact.model_path),
                    "error": str(exc),
                }
            )

    write_json(
        run_dir / "summary.json",
        {
            "branch": args.branch,
            "run_dir": to_repo_relative(run_dir),
            "artifacts_total": len(artifacts),
            "artifacts_succeeded": len(summary_rows),
            "artifacts_failed": len(failures),
            "results": summary_rows,
            "failures": failures,
        },
    )
    write_csv(run_dir / "summary.csv", summary_rows)
    if failures:
        write_csv(run_dir / "failures.csv", failures)


def run_accuracy_for_artifact(
    artifact: ArtifactRecord,
    records,
    class_names: list[str],
    run_dir: Path,
    device_id: int,
    args,
) -> dict[str, object]:
    artifact_dir = run_dir / artifact.model_name / artifact.experiment_name
    confusion = ConfusionMatrixAccumulator(num_classes=len(class_names))
    total_loss = 0.0
    total_samples = 0

    with ACLModelRunner(artifact, device_id=device_id) as runner:
        for record in records:
            sample = load_npy_sample(record.file_path, artifact.input_spec.dtype)
            outputs, _ = runner.infer_with_timing(sample)
            logits = outputs[0]
            probabilities = softmax_np(logits, axis=1)
            predictions = argmax_predictions(probabilities, axis=1)
            labels = np.asarray([record.label_idx], dtype=np.int64)
            losses = cross_entropy_from_logits(logits.astype(np.float64, copy=False), labels)
            confusion.update(predictions, labels)
            total_loss += float(losses.sum())
            total_samples += int(labels.shape[0])

    summary = {
        "branch": artifact.branch,
        "model_name": artifact.model_name,
        "experiment_name": artifact.experiment_name,
        "artifact_path": to_repo_relative(artifact.model_path),
        "summary_path": to_repo_relative(artifact.summary_path),
        "samples": total_samples,
        "accuracy": confusion.accuracy(),
        "avg_loss": float(total_loss / total_samples) if total_samples else 0.0,
        "input_name": artifact.input_spec.name,
        "input_shape": list(artifact.input_spec.shape),
        "input_dtype": str(artifact.input_spec.dtype),
        "output_name": artifact.output_specs[0].name,
        "output_shape": list(artifact.output_specs[0].shape),
        "output_dtype": str(artifact.output_specs[0].dtype),
        "confusion_matrix_csv": to_repo_relative(write_confusion_matrix_csv(artifact_dir / "confusion_matrix.csv", confusion.matrix, class_names)),
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
