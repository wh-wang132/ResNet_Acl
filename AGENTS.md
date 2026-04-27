# ResNet_Acl 代理工作指南

本文面向在 `/root/ResNet_Acl` 中工作的 Codex/agent。回答和新增文档默认使用中文；代码、路径、字段名和命令保持原样。

## 项目定位

- 当前仓库是本科毕设“基于昇腾 AI 架构的高效化无人机射频信号识别”的 Ascend ACL 推理端仓库，云端仓库为 `wh-wang132/ResNet_Acl`。
- 上游训练端仓库为 `wh-wang132/ResNet`。训练端负责 base model 训练、剪枝、QAT、ONNX 导出、AMCT 转换和 ATC 编译；本仓库只消费已导入的 ATC/OM 产物，执行预检、ACL 推理评测和离线可视化。
- 不要把训练、剪枝、QAT、ONNX、AMCT 或 ATC 编译主流程迁入本仓库。若需要调整产物生产逻辑，应在上游训练端处理，并同步本仓库的消费契约。
- 主要数据流：
  `input/atc/<branch>/<model>/<experiment>/OM + atc_summary.json`
  -> `src.common.artifact_scanner`
  -> `input/splits/*.json + Data/**/*.npy`
  -> `src.common.data`
  -> `src.common.acl_runner`
  -> `src.accuracy` / `src.efficiency`
  -> `src.visualization`。

## 环境与常用命令

- 优先使用 `pixi run python -m src.<module>`，不要绕过 pixi 环境。`pixi.toml` 固定 Python `3.12`、CANN Toolkit `8.5.0` 和 Ascend 310B ops `8.5.0`。
- 运行 accuracy、efficiency 或默认 complexity 预检前，确保 `.envrc` 或等价环境已加载 Ascend `set_env.sh`，并设置 `PYTHONPATH=${REPO_ROOT}/src`。
- `src.accuracy` 和 `src.efficiency` 需要当前 Python 环境可导入 `acl`；`src.efficiency` 和默认 `src.validate` 需要 `atc` 在 `PATH` 中可用。
- 仓库预检：
  ```bash
  pixi run python -m src.validate --sample_limit 8
  pixi run python -m src.validate --sample_limit 8 --skip_complexity_precheck
  ```
- 单 artifact 精度评测：
  ```bash
  pixi run python -m src.accuracy \
    --branch pruning_fp16 \
    --artifact_path input/atc/pruning_fp16/resnet18_2d/from_ratio0.60_steps5_global_ft10_bs64 \
    --num_instances 1 \
    --buffer_depth 1
  ```
- 单 artifact 效率评测：
  ```bash
  pixi run python -m src.efficiency \
    --branch amct_deploy \
    --artifact_path input/atc/amct_deploy/resnet18_2d/from_ratio0.80_steps16_global_ft10_bs64 \
    --num_instances 1 \
    --buffer_depth 1 \
    --warmup_steps 20 \
    --repeat 5
  ```
- 离线可视化：
  ```bash
  pixi run python -m src.visualization --strict
  ```
- 批量脚本使用 `set -eu`，任一 artifact 失败会中止整体任务：
  ```bash
  sh autorun/autorun_accuracy.sh --limit 128
  sh autorun/autorun_efficiency.sh --limit 128 --repeat 3
  sh autorun/autorun_visualization.sh
  ```

## 模块边界

- `src/common` 是共享基础设施层，不放具体评测入口流程。
- `src/common/branch_specs.py` 是分支名、扫描根目录、OM 文件名、summary 文件名和输出子目录的唯一来源。
- `src/common/args.py` 维护 accuracy/efficiency 共享 CLI 参数和基础硬约束。
- `src/common/manifest.py` 只处理 split manifest 和 `SampleRecord` 构建。
- `src/common/artifact_scanner.py` 只处理 artifact 扫描、`atc_summary.json` 解析、`TensorSpec`/`ArtifactRecord` 构建和摘要契约校验。
- `src/common/data.py` 只处理 `.npy` 读取、补前导 `1` 维、dtype 转换、C contiguous 校验和运行时输入适配。
- `src/common/runtime_prep.py` 组合 artifact、manifest、样本预加载和 runner 启动前校验。
- `src/common/acl_runner.py` 封装 ACL 生命周期、单实例推理和多实例有序并发执行；不要让上层直接管理 ACL 资源。
- `src/common/model_complexity.py` 调用 `atc --mode=1` 解析 OM JSON，并只统计明确分类过的 Conv/MatMul MACs。
- `src/common/metrics.py` 放交叉熵、混淆矩阵和 latency summary。
- `src/common/report.py` 放输出目录、JSON/CSV、混淆矩阵图等写出工具。
- `src.accuracy` 只做精度评测编排，输出 `output/accuracy/<branch>/<model>/<experiment>/`。
- `src.efficiency` 只做效率评测编排，输出 `output/efficiency/<branch>/<model>/<experiment>/`。
- `src.validate` 是静态预检入口，不执行真实 ACL 推理；默认会做 efficiency 所需的 complexity 前置检查。
- `src.visualization` 是纯离线论文插图生成层，只读取 accuracy/efficiency 已生成的 summary、CSV 和图片路径；不得导入 ACL、调用 ATC 或重新计算 OM complexity。
- `autorun/*.sh` 只负责遍历 `<branch>/<model>/<experiment>` 两级 artifact 目录并调用 Python CLI。

## 接口契约

- 当前内置推理分支只有 `pruning_fp16` 和 `amct_deploy`。
- 当前只支持 `batch_size=1`、单输入、单输出 OM，以及 ACL dtype `float16` 和 `float32`。
- `pruning_fp16` 分支的 OM 文件名为 `model_fp16.om`；`amct_deploy` 分支的 OM 文件名为 `deploy_model.om`；summary 文件名均为 `atc_summary.json`。
- split manifest 默认路径为 `input/splits/dataset_split__train0.60_val0.20_test0.20_seed42.json`，必须至少包含 `class_names`、`class_to_idx`、`train_files`、`val_files`、`test_files`。
- manifest 中每条样本记录至少包含 `path`、`label_name`、`label_idx`。
- 数据目录默认为 `Data/`，按类别目录组织 `.npy` 文件；原始样本形状可通过补前导 `1` 维对齐到模型输入。
- `atc_summary.json` 强依赖字段包括 `stage`、`branch`、`source_interface`、`resolved_input_shape`、`source_architecture_signature.parameter_count`。
- `source_interface` 当前至少需要 `input_name`、`output_name`、`input_elem_type`、`output_elem_type`、`input_shape`、`output_shape`。
- `source_interface.input_shape` 在 `batch=1` 条件下解析后必须等于 `resolved_input_shape`。
- 推理前必须保持三方一致性校验：预加载样本、`atc_summary.json`、OM 实际 ACL 规格的 shape/dtype 必须一致。
- efficiency summary 文件名必须保持 `summary__instances{num_instances}_buffer{buffer_depth}.json`，可视化按 `num_instances` 和 `buffer_depth` 过滤。
- visualization 按 `ArtifactKey(branch, model_name, experiment_name)` 合并 accuracy 和 efficiency；缺任一侧进入 `missing_inputs`，`--strict` 下中止。
- experiment 名称解析依赖 `from_ratio{r}_steps{s}_{mode}_ft{n}_bs{b}` 格式，改名会影响 ratio、steps、pruning_mode、finetune_epochs、batch_size 派生字段。

## 输出契约

- accuracy 输出目录：`output/accuracy/<branch>/<model_name>/<experiment_name>/`。
- accuracy 关键文件：`summary.json`、`confusion_matrix.csv`、可选 `confusion_matrix.png`、可选 `per_class_metrics.csv`。
- accuracy summary 需要保留 branch/model/experiment、artifact/summary 路径、samples、accuracy、avg_loss、输入输出 name/shape/dtype、execution mode、实例数、buffer depth、dispatch/ordering policy 和产物路径。
- efficiency 输出目录：`output/efficiency/<branch>/<model_name>/<experiment_name>/`。
- efficiency 关键文件：`summary__instances{num_instances}_buffer{buffer_depth}.json`。
- efficiency summary 需要保留 `parameter_count`、`operation_count`、`operation_count_unit=MACs`、`operation_count_scope=per_forward_pass`、included/excluded op types、warmup/repeat、time mode、pure infer/end-to-end latency 与 throughput、H2D/execute/D2H/decode 分阶段耗时。
- visualization 默认输出目录：`output/visualization/paper/`。
- visualization 关键文件：`index.json`、`paper_summary.md`、`tables/paper_top5_candidates.csv`、`tables/paper_branch_pair_summary.csv`、`tables/paper_pareto_candidates.csv` 和 `plots/fig*.svg`。
- visualization 仅服务论文插图表达，不保留探索型 `plot_set`、旧版通用 Markdown 报告或长期维护型汇总表契约。

## 异常处理规范

- 底层库函数应抛领域异常，不要 `print` 后吞错；CLI 入口负责捕获并转换为 `SystemExit(str(exc))`。
- 继续使用现有领域异常：`ValidationError`、`ArtifactScanError`、`ManifestError`、`DataError`、`ACLRuntimeError`、`ModelComplexityError`、`VisualizationError`。
- 参数和运行前置条件要尽早校验；错误消息应包含具体路径、参数名、`expected`/`actual` 等可诊断信息。
- 外部命令必须显式检查命令是否存在、返回码和预期输出文件；失败时保留可诊断输出。`atc --mode=1` 失败时保留尾部输出即可。
- ACL C API 返回码统一通过 `_check_ret` 和 `_unwrap_with_ret` 转换为 `ACLRuntimeError`。
- ACL `open()` 失败必须调用 `close()` 回收已分配资源；清理阶段使用 `raise_on_error=False`，避免清理错误覆盖主异常。
- 推理必需输入缺失应失败；可视化辅助文件缺失可降级为空字段或 `None`，并通过 `missing_inputs` 记录。
- 并发推理新增逻辑必须让 worker 错误通过队列回到主线程，再由主线程包装为 `ACLRuntimeError`；不要让后台线程静默失败。
- Shell 自动化默认 fail-fast；不要默认忽略单个 artifact 失败，除非显式新增“跳过失败并汇总”的模式。

## 跨仓库同步

- 同步新产物时，优先从训练端 `wh-wang132/ResNet` 的 `output/atc/<branch>/<model>/<experiment>/` 复制到本仓库 `input/atc/<branch>/<model>/<experiment>/`。
- 同步 artifact 时必须保留 OM 文件、`atc_summary.json` 和 `fusion_result.json`。efficiency 的参数量和 MACs 统计依赖 summary 与 OM complexity 前置条件。
- 训练端若修改类别顺序、`class_to_idx`、样本布局、输入输出 shape/dtype、动态 shape、多输出、分支名、文件名、summary 字段或 experiment 命名，本仓库必须同步更新扫描、校验、推理和可视化逻辑。
- 跨仓库更新后的推荐验证顺序：
  ```bash
  pixi run python -m src.validate --branch <branch> --sample_limit 8
  pixi run python -m src.accuracy --branch <branch> --artifact_path <artifact_dir>
  pixi run python -m src.efficiency --branch <branch> --artifact_path <artifact_dir>
  ```
- 如果 validate 失败，优先检查目录布局、manifest 字段、summary 字段、shape/dtype 和 complexity 前置条件，不要先排查 ACL 运行时。

## 改动同步清单

- 新增或修改分支、OM 文件名、扫描目录时，同步 `src/common/branch_specs.py`、`docs/interface_contract.md`、`README.md` 和 `autorun/*.sh`。
- 修改 manifest 字段或样本记录格式时，同步 `src/common/manifest.py`、`src/common/validation.py`、`src/common/runtime_prep.py` 和 `docs/interface_contract.md`。
- 修改输入 shape/dtype、ACL elem type、动态 batch、多输入或多输出时，同步 `src/common/artifact_scanner.py`、`src/common/data.py`、`src/common/acl_runner.py`、`src/common/runtime_prep.py`、`src/accuracy/__main__.py`、`src/efficiency/__main__.py` 和 `docs/interface_contract.md`。
- 修改 CLI 参数时，同步 `src/common/args.py`、对应 `src/*/__main__.py`、`README.md` 和 `docs/interface_contract.md`。
- 修改 accuracy summary 字段时，同步 `src/accuracy/__main__.py`、`src/visualization/sources.py`、`src/visualization/tables.py`、`src/visualization/report.py` 和相关 plots。
- 修改 efficiency summary 字段时，同步 `src/efficiency/__main__.py`、`src/common/metrics.py`、`src/common/model_complexity.py`、`src/visualization/sources.py`、`src/visualization/tables.py` 和 `src/visualization/report.py`。
- 修改 visualization 论文图、paper 表字段或 index 结构时，同步 `src/visualization/schema.py`、`src/visualization/sources.py`、`src/visualization/tables.py`、`src/visualization/report.py`、`src/visualization/plots.py`、`README.md` 和 `docs/interface_contract.md`。
