# ResNet_Acl

本科毕设“基于昇腾 AI 架构的高效化无人机射频信号识别”的推理端代码仓库，对应云端仓库为 `wh-wang132/ResNet_Acl`，上游训练端仓库为 `wh-wang132/ResNet`。

当前仓库聚焦两类工作：

- `src.accuracy`：基于 ACL 的 OM 精度评测
- `src.efficiency`：基于 ACL 的 OM 效率评测
- `src.visualization`：离线聚合精度/效率产物并生成论文分析图表

同时提供一个不触发真实 ACL 推理的仓库预检入口：

- `src.validate`：校验数据集、split manifest、ATC 产物、样本 shape/dtype 契约，并预检 `src.efficiency` 所需的 complexity 前置条件

## 仓库结构

```text
ResNet_Acl/
├── Data/                         # 24 类无人机射频 .npy 样本
├── input/
│   ├── atc/
│   │   ├── pruning_fp16/         # 剪枝 + FP16 导出的 OM 产物
│   │   └── amct_deploy/          # AMCT 量化部署产物
│   └── splits/
│       └── dataset_split__train0.60_val0.20_test0.20_seed42.json
├── output/
│   ├── accuracy/                 # 精度评测输出
│   ├── efficiency/               # 效率评测输出
│   └── visualization/            # 论文插图输出
├── autorun/                      # 遍历全部 artifact 的批量脚本
└── src/
    ├── accuracy/
    ├── efficiency/
    ├── validate/
    └── common/
```

## 环境

项目使用 `pixi` 管理运行环境，`pixi.toml` 当前固定：

- Python `3.12`
- Ascend CANN Toolkit `8.5.0`
- Ascend 310B ops `8.5.0`

仓库运行时环境变量放在 `.envrc` 中，当前会完成：

- 设置 `REPO_ROOT`
- `source .pixi/envs/default/Ascend/ascend-toolkit/set_env.sh`
- 补充 `PYTHONPATH=${REPO_ROOT}/src`

推荐使用 `direnv` 自动加载该环境：

```bash
sudo apt-get install direnv
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
source ~/.bashrc
direnv allow
```

如果不使用 `direnv`，则在执行任何推理或校验命令前手动加载：

```bash
source .envrc
```

下文所有命令默认都建立在 `.envrc` 已加载的前提下。

安装后可先做仓库预检，不触发真实 ACL 推理：

```bash
pixi run python -m src.validate --sample_limit 8
```

默认 `src.validate` 也会预检 `src.efficiency` 所需的 complexity 条件，因此需要 `atc` 可用，并会调用 `atc --mode=1` 做离线 OM 解析。如果只想排查数据、manifest 和 shape/dtype 契约，可显式跳过：

```bash
pixi run python -m src.validate --sample_limit 8 --skip_complexity_precheck
```

如果只想检查某个分支或单个 artifact：

```bash
pixi run python -m src.validate --branch pruning_fp16 --sample_limit 8
pixi run python -m src.validate \
  --branch pruning_fp16 \
  --artifact_path input/atc/pruning_fp16/resnet18_2d/from_ratio0.60_steps5_global_ft10_bs64 \
  --sample_limit 8
```

## 运行入口

查看命令行参数：

```bash
pixi run python -m src.accuracy --help
pixi run python -m src.efficiency --help
pixi run python -m src.validate --help
pixi run python -m src.visualization --help
```

精度评测示例：

```bash
pixi run python -m src.accuracy \
  --branch pruning_fp16 \
  --artifact_path input/atc/pruning_fp16/resnet18_2d/from_ratio0.60_steps5_global_ft10_bs64 \
  --num_instances 1 \
  --buffer_depth 1
```

效率评测示例：

```bash
pixi run python -m src.efficiency \
  --branch amct_deploy \
  --artifact_path input/atc/amct_deploy/resnet18_2d/from_ratio0.80_steps16_global_ft10_bs64 \
  --num_instances 1 \
  --buffer_depth 1 \
  --warmup_steps 20 \
  --repeat 5
```

批量遍历全部 artifact：

```bash
sh autorun/autorun_accuracy.sh --limit 128
sh autorun/autorun_efficiency.sh --limit 128 --repeat 3
sh autorun/autorun_visualization.sh
```

离线生成论文插图：

```bash
pixi run python -m src.visualization --strict
```

## 当前硬约束

- 当前 ATC 产物按 `batch_size=1` 编译，推理端只接受 `--batch_size 1`
- 数据样本默认来自 `Data/**/*.npy`
- 默认 split manifest 为 `input/splits/dataset_split__train0.60_val0.20_test0.20_seed42.json`
- 当前内置推理分支只有 `pruning_fp16` 和 `amct_deploy`
- `atc_summary.json` 中 `stage`、`branch`、`source_interface`、`resolved_input_shape` 是推理端强依赖字段
- 目前 ACL dtype 映射仅支持 `float16` 和 `float32`

## 产物输出

`src.accuracy` 默认写到 `output/accuracy/<branch>/...`，核心输出包括：

- `summary.json`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `per_class_metrics.csv`

`src.efficiency` 默认写到 `output/efficiency/<branch>/...`，核心输出包括：

- `summary__instances{N}_buffer{M}.json`

其中 `summary__instances{N}_buffer{M}.json` 现在额外包含：

- `parameter_count`：直接透传 `atc_summary.json` 中的参数量
- `operation_count`：按单次前向的理论 `MACs` 统计
- `operation_count_included_op_types`：当前被计入 `MACs` 的 OM op type
- `operation_count_excluded_op_types`：当前显式忽略的 OM op type

当前 `operation_count` 只统计卷积和矩阵乘家族算子，不把 `Add`、`Cast`、`TransData`、量化/反量化等部署辅助算子混入理论 `MACs`。

`src.visualization` 默认写到 `output/visualization/paper/`，核心输出面向论文插图，包括：

- `index.json`
- `paper_summary.md`
- `tables/paper_top5_candidates.csv`
- `tables/paper_branch_pair_summary.csv`
- `tables/paper_pareto_candidates.csv`
- `plots/fig1_pareto_error_throughput.svg`
- `plots/fig2_pareto_error_latency.svg`
- `plots/fig3_fp16_int8_pair_summary.svg`
- `plots/fig4_complexity_tradeoff.svg`
- 可选 `plots/fig5_per_class_recall_delta.svg`

该模块只消费已有 `output/accuracy` 与 `output/efficiency`，不导入 ACL、不调用 ATC、不触发真实推理。论文图默认使用 SVG，可通过 `--format png --dpi 300` 生成位图版本。由于大多数模型精度高于 `0.99`，涉及精度对比的主图使用 `error_rate = 1 - accuracy` 并采用对数坐标轴；涉及计算量、参数量和 latency 的主图也采用对数坐标轴。当前图表会客观呈现本次测量中 FP16 pruning 与 INT8 AMCT 的精度、吞吐和延迟差异。

## 上下游契约

训练端 `wh-wang132/ResNet` 需要稳定提供：

- 类别顺序和 `class_to_idx`
- `.npy` 样本布局
- 导出 ONNX/ATC 所需的输入输出 shape 与 dtype
- `model_name / experiment_name` 维度下的实验命名

推理端 `wh-wang132/ResNet_Acl` 消费这些产物并执行 ACL 推理与评测。

详细目录约定和字段契约见 [docs/interface_contract.md](/root/ResNet_Acl/docs/interface_contract.md)。
