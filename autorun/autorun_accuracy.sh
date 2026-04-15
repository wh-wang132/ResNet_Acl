#!/bin/sh
set -eu

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

print_section() {
    title=$1
    printf '\n============================================================\n'
    printf '%s\n' "$title"
    printf '============================================================\n'
}

run_branch() {
    branch=$1
    root_dir=$2
    shift 2
    tmp_list=$(mktemp)
    trap 'rm -f "$tmp_list"' EXIT HUP INT TERM

    print_section "开始遍历 ${branch}: ${root_dir}"

    find "$root_dir" -mindepth 2 -maxdepth 2 -type d | LC_ALL=C sort > "$tmp_list"

    while IFS= read -r artifact_dir; do
        [ -n "$artifact_dir" ] || continue
        printf '\n[%s] %s\n' "$branch" "$artifact_dir"
        pixi run python -m src.accuracy \
            --branch "$branch" \
            --artifact_path "$artifact_dir" \
            --num_instances 2 \
            --buffer_depth 1 \
            "$@"
    done < "$tmp_list"

    print_section "完成遍历 ${branch}: ${root_dir}"
}

cd "$ROOT_DIR"

run_branch "pruning_fp16" "input/atc/pruning_fp16" "$@"
run_branch "amct_deploy" "input/atc/amct_deploy" "$@"
