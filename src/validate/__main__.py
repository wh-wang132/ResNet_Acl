from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common.args import DEFAULT_DATA_DIR, DEFAULT_MANIFEST
from src.common.branch_specs import BRANCH_SPECS
from src.common.validation import (
    DEFAULT_SAMPLE_LIMIT,
    ValidationError,
    build_validation_report,
    resolve_cli_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ResNet_Acl 仓库预检（不触发真实 ACL 推理）")
    parser.add_argument("--branch", choices=sorted(BRANCH_SPECS), default=None)
    parser.add_argument("--artifact_path", type=Path, default=None, help="单个 artifact 目录或 OM 文件路径")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split_manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sample_limit", type=int, default=DEFAULT_SAMPLE_LIMIT, help="每个 shape/dtype 组合抽样校验的 test 样本数")
    parser.add_argument("--output_path", type=Path, default=None, help="可选，额外写出 JSON 校验报告")
    parser.add_argument(
        "--skip_complexity_precheck",
        action="store_true",
        help="跳过 efficiency 所需的 complexity 预检，仅执行纯数据/manifest/shape 契约检查",
    )
    return parser.parse_args()


def write_output(output_path: Path, payload: dict[str, object]) -> Path:
    resolved = resolve_cli_path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return resolved


def main() -> None:
    args = parse_args()
    try:
        report = build_validation_report(
            branch=args.branch,
            artifact_path=args.artifact_path,
            data_dir=args.data_dir,
            split_manifest=args.split_manifest,
            sample_limit=args.sample_limit,
            skip_complexity_precheck=args.skip_complexity_precheck,
        )
    except ValidationError as exc:
        raise SystemExit(str(exc)) from exc

    if args.output_path is not None:
        output_path = write_output(args.output_path, report)
        report["output_path"] = str(output_path)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
