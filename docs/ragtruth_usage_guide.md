# RAGTruth 幻觉检测任务使用指南

## 概述

RAGTruthTask 是一个用于评估大型语言模型（LLM）幻觉检测能力的任务管理器。它支持多任务类型（Summary/QA/Data2txt）的灵活加载和独立评估，并提供详细的 span-level 指标计算。

## 主要特性

- **多任务支持**：支持同时评估 Summary、QA、Data2txt 三种任务类型的任意组合
- **灵活的数据加载**：支持从 train 或 test 分割加载数据，支持固定数量采样或比例采样
- **精细的指标计算**：基于 span-level 的 F1、Precision、Recall 计算
- **综合报告生成**：生成分任务报告和汇总报告

## 快速开始

### 安装依赖

确保已安装必要的依赖：

```bash
uv pip install datasets pandas pydantic pyyaml
```

### 基础使用

#### 示例 1：评估单个任务（Summary）

```python
from queryplan_eval.tasks import RAGTruthTask
from queryplan_eval.core.base_llm import BaseLLM

# 初始化 LLM（假设已配置）
llm = BaseLLM(model="your-model-name")

# 创建任务，只评估 Summary 任务
task = RAGTruthTask(
    task_types=["Summary"],
    split="test",
    llm=llm,
    output_dir="outputs/ragtruth_summary"
)

# 运行评估
metrics = task.run_evaluation(temperature=0.0)

# 查看结果
print(metrics)
```

#### 示例 2：多任务评估

```python
# 评估 Summary 和 QA 两个任务
task = RAGTruthTask(
    task_types=["Summary", "QA"],
    split="test",
    use_cot=False,  # 不使用 Chain-of-Thought
    llm=llm,
    output_dir="outputs/ragtruth_multi"
)

metrics = task.run_evaluation()
```

#### 示例 3：从 train 分割采样数据

```python
# 从 train 分割中采样 100 个样本
task = RAGTruthTask(
    task_types=["Summary", "QA", "Data2txt"],
    split="train",
    sample_n=100,  # 采样 100 个样本
    use_cot=True,  # 启用 Chain-of-Thought
    llm=llm,
    output_dir="outputs/ragtruth_train_sample"
)

metrics = task.run_evaluation()
```

#### 示例 4：按比例采样

```python
# 采样 20% 的数据
task = RAGTruthTask(
    task_types=["Summary", "QA"],
    split="test",
    sample_ratio=0.2,  # 采样 20%
    llm=llm,
    output_dir="outputs/ragtruth_sampled"
)

metrics = task.run_evaluation()
```

## 参数详解

### RAGTruthTask 初始化参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `task_types` | `List[str]` | ✓ | 任务类型列表，如 `["Summary", "QA", "Data2txt"]` |
| `split` | `str` | - | 数据分割，"train" 或 "test"，默认 "test" |
| `sample_n` | `Optional[int]` | - | 固定采样数量，与 `sample_ratio` 互斥 |
| `sample_ratio` | `Optional[float]` | - | 采样比例 (0-1)，与 `sample_n` 互斥 |
| `use_cot` | `bool` | - | 是否使用 Chain-of-Thought 推理，默认 False |
| `cache_dir` | `Optional[Path]` | - | 数据集缓存目录 |
| `llm` | `BaseLLM` | ✓ | LLM 实例 |
| `output_dir` | `str` | - | 输出目录，默认 "outputs/ragtruth" |

## 输出文件

RAGTruthTask 生成以下输出文件：

### 详细结果

- **results.csv**：所有样本的详细评估结果，包含以下字段：
  - `idx`：样本索引
  - `task_type`：任务类型
  - `context_length`：参考文本长度
  - `output_length`：模型输出长度
  - `predicted_hallucinations`：模型预测的幻觉片段列表（JSON）
  - `ok`：是否成功解析
  - `latency_sec`：执行耗时
  - `precision`、`recall`、`f1`：span-level 指标

- **results_{TaskType}.csv**：按任务类型分割的结果

### 指标文件

- **metrics.json**：综合指标，包含：
  - `overall`：全局平均指标
  - `by_task`：按任务类型的指标
  - `parse_success_rate`：解析成功率

### 报告文件

- **summary.txt**：人类可读的综合摘要，包含配置信息、总体结果和分任务指标
- **report_{TaskType}.txt**：按任务类型的详细报告

## 指标说明

### Span-Level 指标

RAGTruthTask 使用字符级别的 span 匹配来计算指标，基于以下公式：

- **Precision** = |预测 spans ∩ 真实 spans| / |预测 spans|
- **Recall** = |预测 spans ∩ 真实 spans| / |真实 spans|
- **F1** = 2 × Precision × Recall / (Precision + Recall)

其中：
- 预测 spans：模型识别出的幻觉片段在输出文本中的字符位置
- 真实 spans：标注数据中的幻觉片段位置

### 支持的任务类型

#### Summary（摘要任务）

| 参数 | 必需 | 说明 |
|------|------|------|
| `reference` | ✓ | 原始文档 |
| `response` | ✓ | 摘要文本 |

#### QA（问答任务）

| 参数 | 必需 | 说明 |
|------|------|------|
| `question` | ✓ | 问题文本 |
| `reference` | ✓ | 参考段落 |
| `response` | ✓ | 回答文本 |

#### Data2txt（数据转文本任务）

| 参数 | 必需 | 说明 |
|------|------|------|
| `reference` | ✓ | 结构化数据（JSON 格式） |
| `response` | ✓ | 生成的文本 |

## 高级用法

### 自定义采样

```python
# 从 test 分割采样前 50 个样本
task = RAGTruthTask(
    task_types=["Summary"],
    split="test",
    sample_n=50,
    llm=llm,
    output_dir="outputs/ragtruth_custom"
)
```

### 启用 Chain-of-Thought

```python
# 使用 CoT 进行推理，需要模型支持
task = RAGTruthTask(
    task_types=["Summary", "QA"],
    split="test",
    use_cot=True,
    llm=llm,
    output_dir="outputs/ragtruth_cot"
)
```

### 处理多个任务组合

```python
# 分别评估，然后查看分任务指标
combinations = [
    (["Summary"], "summary_only"),
    (["QA"], "qa_only"),
    (["Data2txt"], "data2txt_only"),
    (["Summary", "QA"], "summary_and_qa"),
    (["Summary", "QA", "Data2txt"], "all_tasks"),
]

for task_types, output_name in combinations:
    task = RAGTruthTask(
        task_types=task_types,
        split="test",
        sample_n=100,
        llm=llm,
        output_dir=f"outputs/ragtruth_{output_name}"
    )
    metrics = task.run_evaluation()
    print(f"{output_name}: {metrics['overall']['f1']:.4f}")
```

## 常见问题

### Q: 如何处理大型数据集？

A: 使用采样参数来减少数据量：

```python
# 方式 1：采样固定数量
task = RAGTruthTask(..., sample_n=1000, ...)

# 方式 2：采样比例
task = RAGTruthTask(..., sample_ratio=0.1, ...)
```

### Q: 如何查看详细的错误信息？

A: 查看输出 CSV 中的 `error` 和 `raw_response` 字段。

### Q: 指标为 0 是什么原因？

A: 可能是：
1. 解析失败（`ok=False`）
2. 模型输出的幻觉片段与真实数据完全不匹配
3. 模型没有识别到任何幻觉（预测 spans 为空）

## 实现细节

### 数据加载流程

1. 根据 `task_types` 参数加载多个任务的数据
2. 使用 HuggingFace Dataset 合并多个任务的数据
3. 应用采样策略（`sample_n` 或 `sample_ratio`）

### Prompt 构建

- 使用 `RAGPromptManager` 根据任务类型选择对应的模板
- QA 任务需要额外的 `question` 字段
- 支持 CoT 和 non-CoT 两种模式

### Span 解析

- 使用 `parse_spans_from_text()` 将文本片段转换为字符位置
- 支持标点符号差异的模糊匹配

## 文件结构

```
src/queryplan_eval/
├── schemas.py              # 添加了 HallucinationResult
├── tasks/
│   └── ragtruth_task.py    # RAGTruthTask 实现
└── metrics/
    └── ragtruth_metrics.py # 指标计算函数
```

## 相关模块

- `RAGTruthDataset`：数据集加载，参考 `src/queryplan_eval/datasets/ragtruth.py`
- `RAGPromptManager`：Prompt 管理，参考 `src/queryplan_eval/core/prompt_manager.py`
- `span_utils`：Span 处理工具，参考 `src/queryplan_eval/metrics/span_utils.py`
