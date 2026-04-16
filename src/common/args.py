from __future__ import annotations

import argparse
from pathlib import Path

from .branch_specs import REPO_ROOT, BRANCH_SPECS


DEFAULT_MANIFEST = REPO_ROOT / "input" / "splits" / "dataset_split__train0.60_val0.20_test0.20_seed42.json"
DEFAULT_DATA_DIR = REPO_ROOT / "Data"
DEFAULT_ACCURACY_OUTPUT = REPO_ROOT / "output" / "accuracy"
DEFAULT_EFFICIENCY_OUTPUT = REPO_ROOT / "output" / "efficiency"


def _base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--branch", required=True, choices=sorted(BRANCH_SPECS))
    parser.add_argument("--artifact_path", type=Path, required=True, help="单个 artifact 目录或 OM 文件路径")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split_manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_instances", type=int, default=1, help="并发 ACL 模型实例数，1 表示串行单实例")
    parser.add_argument("--buffer_depth", type=int, default=1, help="每个实例的输入/输出阻塞队列深度")
    return parser


def _finalize_common_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.batch_size != 1:
        raise ValueError("当前 ATC 产物按 batch_size=1 编译，推理端暂仅支持 --batch_size 1")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit 必须为正整数")
    if args.num_instances <= 0:
        raise ValueError("--num_instances 必须为正整数")
    if args.buffer_depth <= 0:
        raise ValueError("--buffer_depth 必须为正整数")
    args.data_dir = args.data_dir.resolve() if args.data_dir.is_absolute() else (REPO_ROOT / args.data_dir).resolve()
    args.split_manifest = (
        args.split_manifest.resolve()
        if args.split_manifest.is_absolute()
        else (REPO_ROOT / args.split_manifest).resolve()
    )
    if args.artifact_path is not None:
        args.artifact_path = (
            args.artifact_path.resolve()
            if args.artifact_path.is_absolute()
            else (REPO_ROOT / args.artifact_path).resolve()
        )
    args.output_root = (
        args.output_root.resolve()
        if args.output_root.is_absolute()
        else (REPO_ROOT / args.output_root).resolve()
    )
    return args


def parse_accuracy_args() -> argparse.Namespace:
    parser = _base_parser("基于 ACL 的 OM 精度测试")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=DEFAULT_ACCURACY_OUTPUT,
    )
    parser.add_argument(
        "--save_confusion_matrix",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save_per_class_metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return _finalize_common_args(parser.parse_args())


def parse_efficiency_args() -> argparse.Namespace:
    parser = _base_parser("基于 ACL 的 OM 效率测试")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=DEFAULT_EFFICIENCY_OUTPUT,
    )
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--time_mode",
        choices=["pure", "end_to_end", "both"],
        default="both",
    )
    args = _finalize_common_args(parser.parse_args())
    if args.warmup_steps < 0:
        raise ValueError("--warmup_steps 不能为负数")
    if args.repeat <= 0:
        raise ValueError("--repeat 必须为正整数")
    return args
