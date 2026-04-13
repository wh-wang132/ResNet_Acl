from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits64 = np.asarray(logits, dtype=np.float64)
    shifted = logits64 - np.max(logits64, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def argmax_predictions(values: np.ndarray, axis: int = 1) -> np.ndarray:
    return np.argmax(values, axis=axis).astype(np.int64, copy=False)


def cross_entropy_from_logits(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    logits64 = np.asarray(logits, dtype=np.float64)
    labels64 = np.asarray(labels, dtype=np.int64).reshape(-1)
    if logits64.ndim != 2:
        raise ValueError(f"logits 维度必须为 2，实际为 {logits64.ndim}")
    if logits64.shape[0] != labels64.shape[0]:
        raise ValueError("logits 与 labels 的 batch 维度不一致")

    max_logits = np.max(logits64, axis=1, keepdims=True)
    stabilized = logits64 - max_logits
    logsumexp = np.log(np.sum(np.exp(stabilized), axis=1)) + max_logits[:, 0]
    losses = -logits64[np.arange(labels64.shape[0]), labels64] + logsumexp
    return losses


class ConfusionMatrixAccumulator:
    def __init__(self, num_classes: int):
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, predictions: np.ndarray, labels: np.ndarray) -> None:
        preds = np.asarray(predictions, dtype=np.int64).reshape(-1)
        trues = np.asarray(labels, dtype=np.int64).reshape(-1)
        if preds.shape != trues.shape:
            raise ValueError("predictions 与 labels 的形状不一致")
        for prediction, label in zip(preds, trues, strict=True):
            self.matrix[prediction, label] += 1

    def total(self) -> int:
        return int(self.matrix.sum())

    def accuracy(self) -> float:
        total = self.total()
        if total == 0:
            return 0.0
        return float(np.trace(self.matrix) / total)

    def per_class_metrics(self, class_names: list[str]) -> list[dict[str, float | int | str]]:
        total = float(self.matrix.sum())
        rows: list[dict[str, float | int | str]] = []
        for index, class_name in enumerate(class_names):
            true_positive = float(self.matrix[index, index])
            false_positive = float(np.sum(self.matrix[index, :]) - true_positive)
            false_negative = float(np.sum(self.matrix[:, index]) - true_positive)
            true_negative = total - true_positive - false_positive - false_negative
            precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
            recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
            specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
            support = int(np.sum(self.matrix[:, index]))
            rows.append(
                {
                    "class_name": class_name,
                    "precision": precision,
                    "recall": recall,
                    "specificity": specificity,
                    "support": support,
                }
            )
        return rows


@dataclass
class LatencyMeter:
    infer_latencies_ms: list[float] = field(default_factory=list)
    end_to_end_latencies_ms: list[float] = field(default_factory=list)

    def record(self, infer_latency_ms: float, end_to_end_latency_ms: float) -> None:
        self.infer_latencies_ms.append(float(infer_latency_ms))
        self.end_to_end_latencies_ms.append(float(end_to_end_latency_ms))

    @staticmethod
    def _summary(values: list[float]) -> dict[str, float]:
        if not values:
            return {
                "avg_ms": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "total_ms": 0.0,
            }
        arr = np.asarray(values, dtype=np.float64)
        return {
            "avg_ms": float(np.mean(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "total_ms": float(np.sum(arr)),
        }

    def build_summary(self, samples: int) -> dict[str, float | int]:
        infer_summary = self._summary(self.infer_latencies_ms)
        end_to_end_summary = self._summary(self.end_to_end_latencies_ms)
        infer_seconds = infer_summary["total_ms"] / 1000.0
        end_to_end_seconds = end_to_end_summary["total_ms"] / 1000.0
        return {
            "samples": samples,
            "avg_latency_ms": infer_summary["avg_ms"],
            "p50_latency_ms": infer_summary["p50_ms"],
            "p95_latency_ms": infer_summary["p95_ms"],
            "p99_latency_ms": infer_summary["p99_ms"],
            "pure_infer_total_ms": infer_summary["total_ms"],
            "pure_infer_avg_latency_ms": infer_summary["avg_ms"],
            "pure_infer_p50_latency_ms": infer_summary["p50_ms"],
            "pure_infer_p95_latency_ms": infer_summary["p95_ms"],
            "pure_infer_p99_latency_ms": infer_summary["p99_ms"],
            "end_to_end_total_ms": end_to_end_summary["total_ms"],
            "end_to_end_avg_latency_ms": end_to_end_summary["avg_ms"],
            "end_to_end_p50_latency_ms": end_to_end_summary["p50_ms"],
            "end_to_end_p95_latency_ms": end_to_end_summary["p95_ms"],
            "end_to_end_p99_latency_ms": end_to_end_summary["p99_ms"],
            "pure_infer_throughput_samples_per_sec": float(samples / infer_seconds) if infer_seconds else 0.0,
            "end_to_end_throughput_samples_per_sec": float(samples / end_to_end_seconds) if end_to_end_seconds else 0.0,
        }
