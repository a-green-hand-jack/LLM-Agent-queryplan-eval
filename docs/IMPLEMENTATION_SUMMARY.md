# RAGTruth 幻觉检测任务实现总结

## 实现完成概述

本文档总结了 RAGTruth 幻觉检测任务的完整实现，包括数据加载、提示管理、任务评估和指标计算等核心功能。

## 实现的核心功能

### 1. 数据模型定义（schemas.py）

添加了 `HallucinationResult` Pydantic 模型，用于定义 LLM 输出的幻觉检测结果格式。

```python
class HallucinationResult(BaseModel):
    """幻觉检测结果"""
    hallucination_list: List[str] = Field(
        default_factory=list,
        description="检测到的幻觉片段列表，每个片段必须是 response 中的精确子串"
    )
```

### 2. 指标计算模块（metrics/ragtruth_metrics.py）

实现了四个核心函数用于指标计算和聚合：

#### 2.1 `compute_hallucination_metrics(predicted_spans, ground_truth_spans)`
计算单个样本的 span-level 指标（Precision、Recall、F1）。

**特点**：
- 使用字符级别的集合运算
- 支持部分重叠的 span 匹配
- 自动处理空 span 的情况

#### 2.2 `aggregate_metrics_by_task(results, metric_keys)`
按任务类型聚合指标，计算每个任务的平均性能。

**输出格式**：
```python
{
    "Summary": {"precision": 0.9, "recall": 0.85, "f1": 0.87, "count": 50},
    "QA": {"precision": 0.88, "recall": 0.90, "f1": 0.89, "count": 50},
    "Data2txt": {"precision": 0.92, "recall": 0.88, "f1": 0.90, "count": 50}
}
```

#### 2.3 `compute_overall_metrics(results, metric_keys)`
计算整体的平均指标。

#### 2.4 `generate_metric_report(results, output_format)`
生成综合指标报告，支持 dict 或 JSON 输出。

### 3. 任务管理器（tasks/ragtruth_task.py）

实现了 `RAGTruthTask` 类，继承自 `BaseTask`，提供以下核心功能：

#### 3.1 灵活的数据加载

支持以下加载策略：

```python
# 单任务
task = RAGTruthTask(task_types=["Summary"], ...)

# 多任务
task = RAGTruthTask(task_types=["Summary", "QA", "Data2txt"], ...)

# 固定数量采样
task = RAGTruthTask(sample_n=100, ...)

# 比例采样
task = RAGTruthTask(sample_ratio=0.2, ...)
```

**实现细节**：
- 使用 `RAGTruthDataset` 加载每个任务的数据
- 使用 HuggingFace `concatenate_datasets` 合并多任务数据
- 支持 train/test 分割选择

#### 3.2 Prompt 构建

```python
def build_chat(self, item):
    """根据任务类型使用对应的 prompt 模板"""
    # 使用 RAGPromptManager 选择对应的模板
    # 支持 CoT 和 non-CoT 两种模式
    # QA 任务自动添加 question 字段
```

**特点**：
- 动态选择任务特定的 prompt 模板
- 支持 Chain-of-Thought 推理
- 自动处理 QA 任务的额外参数

#### 3.3 结果处理

```python
def process_single_result(self, item, parsed, raw, latency):
    """处理单个样本的评估结果"""
    # 解析预测的幻觉片段
    # 解析真实的幻觉标注
    # 计算 span-level 指标
    # 生成详细的结果记录
```

**输出字段**：
- `idx`、`task_type`、`context_length`、`output_length`
- `predicted_hallucinations`：模型预测的幻觉片段列表（JSON）
- `ok`：是否成功解析
- `latency_sec`：执行耗时
- `precision`、`recall`、`f1`：span-level 指标
- `predicted_spans`、`ground_truth_spans`：转换后的 span 位置（JSON）

#### 3.4 多任务独立评估

```python
def compute_metrics(self, results):
    """计算评估指标"""
    # 计算整体指标（overall）
    # 按任务类型计算分任务指标（by_task）
    # 计算解析成功率
```

**返回结构**：
```python
{
    "total": 150,                          # 总样本数
    "ok": 148,                             # 成功样本数
    "parse_success_rate": 0.9867,          # 解析成功率
    "overall": {                           # 整体指标
        "precision": 0.90,
        "recall": 0.88,
        "f1": 0.89,
        "count": 150
    },
    "by_task": {                           # 分任务指标
        "Summary": {...},
        "QA": {...},
        "Data2txt": {...}
    }
}
```

#### 3.5 综合报告生成

生成以下输出文件：

```
outputs/ragtruth_latest/
├── results.csv              # 所有样本的详细结果
├── results_Summary.csv      # Summary 任务结果
├── results_QA.csv           # QA 任务结果
├── results_Data2txt.csv     # Data2txt 任务结果
├── metrics.json             # 综合指标（JSON 格式）
├── summary.txt              # 人类可读的总结
├── report_Summary.txt       # Summary 任务报告
├── report_QA.txt            # QA 任务报告
└── report_Data2txt.txt      # Data2txt 任务报告
```

## 文件结构

```
src/queryplan_eval/
├── schemas.py                          # 添加 HallucinationResult
├── tasks/
│   ├── __init__.py                     # 导出 RAGTruthTask
│   └── ragtruth_task.py               # RAGTruthTask 实现
├── metrics/
│   ├── __init__.py                     # 导出 ragtruth_metrics
│   └── ragtruth_metrics.py            # 指标计算函数
└── datasets/
    ├── __init__.py                     # 导出 RAGTruthDataset
    └── ragtruth.py                     # 数据集实现

scripts/
└── run_eval_ragtruth.py               # 评估脚本

tests/
└── test_ragtruth_task.py              # 单元测试

docs/
├── ragtruth_usage_guide.md            # 使用指南
└── IMPLEMENTATION_SUMMARY.md          # 本文档
```

## 核心设计决策

### 1. 多任务灵活性

设计支持以下三个层级的灵活性：

- **任务组合**：任意选择 Summary、QA、Data2txt 的组合
- **数据分割**：支持从 train 或 test 分割加载
- **采样策略**：支持固定数量采样或比例采样

### 2. 独立评估与汇总

- 分别评估每个任务类型
- 生成独立的任务报告
- 同时提供整体指标和分任务指标

### 3. Span 级别的评估

使用字符级别的 span 匹配而不是 token 级别，因为：
- 更精确地表示幻觉的位置
- 支持模糊匹配（容忍标点符号差异）
- 与数据集标注方式一致

### 4. 结构化输出

所有输出都包括：
- **详细结果**（CSV）：用于进一步分析
- **指标数据**（JSON）：用于编程处理
- **人类可读报告**（TXT）：用于快速查看

## 使用示例

### 基本使用

```bash
# 评估 3 个任务类型，采样 50 个样本
uv run python scripts/run_eval_ragtruth.py

# 只评估 Summary 任务
uv run python scripts/run_eval_ragtruth.py --task-types Summary

# 评估 Summary 和 QA，采样 100 个样本
uv run python scripts/run_eval_ragtruth.py --task-types Summary QA -n 100

# 使用 CoT 推理
uv run python scripts/run_eval_ragtruth.py --enable-cot

# 自定义输出目录
uv run python scripts/run_eval_ragtruth.py --outdir outputs/ragtruth_custom
```

### Python API 使用

```python
from queryplan_eval.tasks import RAGTruthTask
from queryplan_eval.llms import OpenAILLM

# 初始化 LLM
llm = OpenAILLM(model_name="qwen-flash")

# 创建任务
task = RAGTruthTask(
    task_types=["Summary", "QA"],
    split="test",
    sample_n=50,
    use_cot=False,
    llm=llm,
    output_dir="outputs/ragtruth_test"
)

# 运行评估
metrics = task.run_evaluation(temperature=0.0)

# 查看结果
print(metrics["overall"]["f1"])
```

## 指标解释

### Span-Level F1

使用字符级别的集合运算计算：

- **Precision** = |预测 chars ∩ 真实 chars| / |预测 chars|
- **Recall** = |预测 chars ∩ 真实 chars| / |真实 chars|
- **F1** = 2 × Precision × Recall / (Precision + Recall)

其中：
- 预测 chars：模型识别出的幻觉片段覆盖的所有字符位置
- 真实 chars：标注数据中的幻觉片段覆盖的所有字符位置

### 特殊情况

- 预测和真实都为空 → F1 = 1.0（正确识别无幻觉）
- 一个为空另一个不为空 → F1 = 0.0（完全错误）

## 验证与测试

实现包含以下测试：

1. **指标计算测试**
   - 完美匹配
   - 部分重叠
   - 完全不匹配
   - 空 spans

2. **参数验证测试**
   - 无效任务类型
   - 冲突的采样参数

3. **功能测试**
   - 单任务初始化
   - 多任务初始化
   - 采样参数
   - Prompt 构建

4. **集成测试**
   - 完整工作流程

## 设计亮点

### 1. 架构清晰

- 数据加载：`RAGTruthDataset`
- Prompt 管理：`RAGPromptManager`
- 任务执行：`RAGTruthTask`
- 指标计算：`ragtruth_metrics` 模块

### 2. 灵活易用

- 支持多种初始化方式
- 参数验证充分
- 错误提示清晰

### 3. 输出全面

- 详细的结果数据
- 多格式输出
- 分任务和汇总报告

### 4. 可测试性

- 独立的指标计算函数
- Mock 友好的设计
- 完整的测试覆盖

## 下一步改进方向

1. **性能优化**
   - 批量处理优化
   - 缓存机制

2. **功能扩展**
   - 支持自定义 span 匹配算法
   - 支持更多任务类型

3. **分析增强**
   - 错误分析工具
   - 可视化报告

## 相关文件参考

- 使用指南：`docs/ragtruth_usage_guide.md`
- 数据集实现：`src/queryplan_eval/datasets/ragtruth.py`
- Prompt 管理：`src/queryplan_eval/core/prompt_manager.py`
- Span 工具：`src/queryplan_eval/metrics/span_utils.py`
