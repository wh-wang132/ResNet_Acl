# 上下游接口契约

本文档约束训练端 `wh-wang132/ResNet`、当前推理端仓库以及云端仓库 `wh-wang132/ResNet_Acl` 之间的产物组织方式。后续新增模型分支、调整导出流程或补实验时，应优先保持这些契约稳定。

## 1. 目录契约

### 1.1 数据集目录

推理端默认从 `Data/` 读取样本，按类别目录组织：

```text
Data/
├── 0/
├── 1/
├── ...
└── 23/
```

每个样本文件必须是 `.npy`，当前仓库中的原始样本形状为 `(543, 512)`，推理前允许补前导 `1` 维对齐到模型输入形状 `(1, 1, 543, 512)`。

### 1.2 split manifest

默认文件：

```text
input/splits/dataset_split__train0.60_val0.20_test0.20_seed42.json
```

必须至少包含以下字段：

- `class_names`
- `class_to_idx`
- `train_files`
- `val_files`
- `test_files`

其中每条样本记录至少包含：

- `path`
- `label_name`
- `label_idx`

### 1.3 ATC 产物目录

推理端固定按下面的目录结构消费产物：

```text
input/atc/<branch>/<model_name>/<experiment_name>/
├── atc_summary.json
├── fusion_result.json
└── <model_filename>
```

当前已内置的分支和模型文件名：

- `pruning_fp16` -> `model_fp16.om`
- `amct_deploy` -> `deploy_model.om`

新增分支时，需要同步补齐：

- 分支名
- 扫描根目录
- OM 文件名
- 摘要文件名
- 输出子目录

## 2. `atc_summary.json` 契约

推理端目前会直接校验以下字段：

- `stage` 必须为 `atc`
- `branch` 必须与命令行指定分支一致
- `source_interface` 必须存在且为对象
- `resolved_input_shape` 必须与 `source_interface.input_shape` 在 `batch=1` 条件下解析后的结果一致

`source_interface` 当前最少需要：

- `input_name`
- `output_name`
- `input_elem_type`
- `output_elem_type`
- `input_shape`
- `output_shape`

当前 ACL elem type 映射：

- `1` -> `float32`
- `10` -> `float16`

如果训练端或 ATC 导出端切换了 dtype、动态 shape 或多输出模型，推理端扫描与执行逻辑也必须同步修改。

## 3. 推理端当前约束

- 只支持 `batch_size=1`
- 只支持单输入、单输出 OM
- 静态校验阶段检查 summary、manifest、样本文件和样本 shape/dtype
- 真实推理阶段再通过 ACL 读取 OM 的真实输入输出规格，并做三方一致性校验：
  `预加载样本`、`atc_summary`、`OM 实际规格`

## 4. 输出契约

### 4.1 精度评测

输出目录：

```text
output/accuracy/<branch>/<model_name>/<experiment_name>/
```

关键文件：

- `summary.json`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `per_class_metrics.csv`

### 4.2 效率评测

输出目录：

```text
output/efficiency/<branch>/<model_name>/<experiment_name>/
```

关键文件：

- `summary__instances{num_instances}_buffer{buffer_depth}.json`

关键字段补充：

- `parameter_count`：从 `atc_summary.json -> source_architecture_signature.parameter_count` 透传
- `operation_count`：当前输入 shape 下、单次前向的理论 `MACs`
- `operation_count_unit`：固定为 `MACs`
- `operation_count_scope`：固定为 `per_forward_pass`
- `operation_count_included_op_types`：当前计入 `operation_count` 的 OM op type
- `operation_count_excluded_op_types`：当前显式忽略的 OM op type

当前 `operation_count` 只统计卷积与矩阵乘家族算子，不把 `Add`、`Cast`、`TransData`、量化/反量化、池化与逐元素激活算子混入理论 `MACs`。

## 5. 推荐校验顺序

训练端或云端同步新产物后，建议按下面顺序检查：

1. `pixi run python -m src.validate --branch <branch> --sample_limit 8`
2. `pixi run python -m src.accuracy --branch <branch> --artifact_path <artifact_dir> ...`
3. `pixi run python -m src.efficiency --branch <branch> --artifact_path <artifact_dir> ...`

如果第 1 步失败，优先修复目录布局、manifest 字段、摘要字段或样本 shape/dtype，而不是直接排查 ACL 运行时。
