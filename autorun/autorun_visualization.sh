#!/bin/sh
set -eu

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

print_section() {
    title=$1
    printf '\n============================================================\n'
    printf '%s\n' "$title"
    printf '============================================================\n'
}

cd "$ROOT_DIR"

print_section "开始生成可视化汇总"

pixi run python -m src.visualization \
    --plot_set all \
    --num_instances 1 \
    --buffer_depth 1 \
    "$@"

print_section "完成生成可视化汇总"
