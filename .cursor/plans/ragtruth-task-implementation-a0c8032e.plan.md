<!-- a0c8032e-7579-4b07-a2a5-0cbb3039f961 f0aeecfb-611d-4de6-b2d0-c692139b2559 -->
# RAGTruth Task Implementation Plan

## 任务目标

实现 RAGTruthTask 类，用于评估 LLM 的幻觉检测能力，支持多任务类型（Summary/QA/Data2txt）的灵活加载和独立评估。

## 核心设计

### 1. Schema 定义

在 `src/queryplan_eval/schemas.py` 中添加：

```python
class HallucinationResult(BaseModel):
    """幻觉检测结果"""
    hallucination_list: List[str] = Field(
        default_factory=list,
        description="检测到的幻觉片段列表，每个片段必须是response中的精确子串"
    )
```

### 2. RAGTruthTask 类实现

在 `src/queryplan_eval/tasks/ragtruth_task.py` 创建新文件：

**核心功能**：

- 灵活的数据加载策略（支持多任务组合、split选择、采样）
- 使用 RAGPromptManager 构建 prompt
- 基于 span-level 的指标计算（F1, Precision, Recall）
- 多任务独立评估和汇总报告

**关键方法**：

```python
class RAGTruthTask(BaseTask):
    def __init__(
        self,
        task_types: List[str],  # 如 ["Summary", "QA"]
        split: str = "test",  # "train" 或 "test"
        sample_n: Optional[int] = None,  # 采样数量
        sample_ratio: Optional[float] = None,  # 采样比例
        use_cot: bool = False,  # 是否使用 CoT
        **kwargs
    ):
        # 初始化 RAGPromptManager
        # 加载数据集
    
    def load_dataset(self, path: str) -> Dataset:
        # 根据 task_types 加载并合并多个任务的数据
        # 支持采样策略
    
    def build_chat(self, item: RAGTruthItem) -> list[dict]:
        # 使用 RAGPromptManager.get_prompt() 构建消息
        # 根据 task_type 选择对应的 prompt 模板
    
    def process_single_result(self, item, parsed, raw, latency) -> Dict:
        # 解析 hallucination_list
        # 使用 parse_spans_from_text 转换为 spans
        # 调用 calculate_span_f1 计算指标
    
    def compute_metrics(self, results: List[Dict]) -> Dict:
        # 计算总体指标
        # 按任务类型分组计算
    
    def save_results(self, results: List[Dict], metrics: Dict):
        # 保存详细结果 CSV
        # 保存分任务报告
        # 保存汇总报告
```

### 3. 数据加载策略实现

```python
def _load_flexible_dataset(
    task_types: List[str],
    split: str,
    sample_n: Optional[int],
    sample_ratio: Optional[float],
    cache_dir: Optional[Path]
) -> Dataset:
    """灵活加载数据集
    
    支持：
 - 单任务或多任务组合
 - train/test split 选择
 - 固定数量采样或比例采样
    """
```

### 4. 指标计算增强

在 `src/queryplan_eval/metrics/ragtruth_metrics.py` 创建新文件：

```python
def compute_hallucination_metrics(
    predicted_spans: List[Span],
    ground_truth_spans: List[Span]
) -> Dict[str, float]:
    """计算单个样本的幻觉检测指标"""
    precision, recall, f1 = calculate_span_f1(predicted_spans, ground_truth_spans)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def aggregate_metrics_by_task(
    results: List[Dict]
) -> Dict[str, Dict[str, float]]:
    """按任务类型聚合指标"""
```

### 5. 多任务评估流程

```python
def run_evaluation(self, temperature: float = 0.0) -> Dict:
    """运行评估并生成分任务报告"""
    # 1. 对所有数据进行推理
    # 2. 按 task_type 分组
    # 3. 分别计算每个任务的指标
    # 4. 生成独立报告
    # 5. 汇总总体指标
```

## 文件结构

```
src/queryplan_eval/
├── schemas.py (修改：添加 HallucinationResult)
├── tasks/
│   ├── ragtruth_task.py (新建)
│   └── __init__.py (更新导出)
└── metrics/
    ├── ragtruth_metrics.py (新建)
    └── __init__.py (更新导出)
```

## 关键实现细节

1. **数据加载灵活性**：

   - 通过 `task_types: List[str]` 参数支持任意任务组合
   - 支持 `sample_n` 和 `sample_ratio` 互斥的采样策略
   - 使用 HuggingFace Dataset 的 `concatenate_datasets` 合并多任务数据

2. **Prompt 构建**：

   - 根据 `item.task_type` 动态选择 prompt 模板
   - 传递正确的变量（QA 需要 question，其他只需 reference 和 response）

3. **Span 解析**：

   - 使用 `parse_spans_from_text(hallucination_list, response)` 转换文本到 spans
   - 处理标点符号差异的模糊匹配

4. **指标计算**：

   - 样本级：precision, recall, f1
   - 任务级：按 task_type 分组的平均指标
   - 全局级：所有样本的总体指标

5. **报告生成**：

   - `results.csv`: 所有样本的详细结果
   - `metrics_by_task.json`: 分任务指标
   - `summary.txt`: 人类可读的汇总报告

## 使用示例

```python
# 示例 1: 只评估 Summary 任务
task = RAGTruthTask(
    task_types=["Summary"],
    split="test",
    llm=llm,
    prompt_manager=None,  # 内部创建 RAGPromptManager
    output_dir="outputs/ragtruth_summary"
)

# 示例 2: 评估 Summary + QA，从 train 中采样 100 个
task = RAGTruthTask(
    task_types=["Summary", "QA"],
    split="train",
    sample_n=100,
    use_cot=True,
    llm=llm,
    prompt_manager=None,
    output_dir="outputs/ragtruth_multi"
)

# 示例 3: 评估所有任务，采样 20%
task = RAGTruthTask(
    task_types=["Summary", "QA", "Data2txt"],
    split="test",
    sample_ratio=0.2,
    llm=llm,
    prompt_manager=None,
    output_dir="outputs/ragtruth_all"
)

# 运行评估
metrics = task.run_evaluation(temperature=0.0)
```

### To-dos

- [ ] 在 schemas.py 中添加 HallucinationResult 模型
- [ ] 创建 metrics/ragtruth_metrics.py 实现指标计算函数
- [ ] 创建 tasks/ragtruth_task.py 实现 RAGTruthTask 类
- [ ] 更新 tasks/__init__.py 和 metrics/__init__.py 导出新模块
- [ ] 编写单元测试验证多任务加载、指标计算和报告生成