from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BranchSpec:
    branch: str
    scan_root: Path
    model_filename: str
    summary_filename: str
    output_subdir: str


BRANCH_SPECS: dict[str, BranchSpec] = {
    "pruning_fp16": BranchSpec(
        branch="pruning_fp16",
        scan_root=REPO_ROOT / "input" / "atc" / "pruning_fp16",
        model_filename="model_fp16.om",
        summary_filename="atc_summary.json",
        output_subdir="pruning_fp16",
    ),
    "amct_deploy": BranchSpec(
        branch="amct_deploy",
        scan_root=REPO_ROOT / "input" / "atc" / "amct_deploy",
        model_filename="deploy_model.om",
        summary_filename="atc_summary.json",
        output_subdir="amct_deploy",
    ),
}


def get_branch_spec(branch: str) -> BranchSpec:
    try:
        return BRANCH_SPECS[branch]
    except KeyError as exc:
        raise ValueError(f"不支持的推理分支: {branch}") from exc
