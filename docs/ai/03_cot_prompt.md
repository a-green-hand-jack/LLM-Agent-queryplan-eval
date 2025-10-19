# CoT Prompt 设计与实现

**日期**: 2025-10-19  
**目标**: 为 QueryPlan-LLM 系统添加 Chain-of-Thought (CoT) 推理能力，提升参数提取的准确性

---

## 📋 任务分析

### 当前状态
- **Prompt 版本**: v5 (`queryplan_system_prompt_v5.j2`)
- **输出模式**: 直接结构化输出（使用 Outlines 库）
- **Schema**: `QueryResult` 包含 `plans`, `refused`, `refuse_reason`
- **问题**: 缺少中间推理过程，模型可能在复杂查询时出错

### CoT 优势
1. **提升复杂推理**: 多步骤时间计算、多域判断等场景
2. **可解释性**: 输出推理过程，便于调试和分析
3. **减少错误**: 通过显式推理减少直接跳跃到错误结论
4. **提升拒答准确性**: 明确推理是否应该拒答

---

## 🎯 设计方案

### 方案选择
**选定方案**: 在 `QueryResult` schema 中添加 `reasoning` 字段

**理由**:
- 与现有 Outlines 结构化输出兼容
- 最小化代码改动
- 保持单次 API 调用（相比两阶段方法更高效）

### CoT 结构设计

```json
{
  "reasoning": {
    "query_analysis": "分析用户问题的意图和关键信息",
    "domain_identification": "识别问题所属的领域（健康/运动/其他）",
    "time_calculation": "如有时间相关内容，展示计算过程",
    "refuse_check": "检查是否应该拒答及原因",
    "final_decision": "最终决策及参数提取结果"
  },
  "plans": [...],
  "refused": false,
  "refuse_reason": ""
}
```

---

## 🔧 实现步骤

### Step 1: 修改 Schema

**文件**: `src/queryplan_eval/schemas.py`

需要添加：
1. `ReasoningSteps` 类：存储推理过程的各个步骤
2. 修改 `QueryResult` 添加 `reasoning` 字段
3. 更新 `normalize_result` 函数以处理 reasoning

### Step 2: 创建 CoT Prompt

**文件**: `src/queryplan_eval/prompts/queryplan_system_prompt_v6_cot.j2`

关键改进：
1. 在输出格式说明中添加 `reasoning` 字段要求
2. 提供 CoT 推理模板和示例
3. 强调先推理后输出的流程

### Step 3: 修改运行脚本

**文件**: `scripts/run_eval.py`

可能需要：
1. 在 CSV 输出中添加 reasoning 列
2. 在 summary 中统计 reasoning 的使用情况

### Step 4: 测试与验证

运行评估并对比：
- v5 (无 CoT) vs v6 (CoT) 的准确率
- reasoning 质量分析
- 时间开销对比

---

## 📝 实现记录

### 1. Schema 修改 ✅

**文件**: `src/queryplan_eval/schemas.py`

**新增类**:
```python
class ReasoningSteps(BaseModel):
    """CoT 推理步骤"""
    query_analysis: str  # 分析用户问题的意图和关键信息
    domain_identification: str  # 识别问题所属的领域
    time_calculation: Optional[str]  # 如有时间相关内容，展示计算过程
    refuse_check: str  # 检查是否应该拒答及原因
    final_decision: str  # 最终决策及参数提取结果
```

**修改 QueryResult 类**:
- 添加 `reasoning: Optional[ReasoningSteps]` 字段
- 更新 `model_config` 的示例以包含 reasoning

**修改 normalize_result 函数**:
- 添加 `include_reasoning: bool = False` 参数
- 当启用时将 reasoning 包含在输出中

**时间**: 2025-10-19 完成

---

### 2. CoT Prompt 创建 ✅

**文件**: `src/queryplan_eval/prompts/queryplan_system_prompt_v6_cot.j2`

**关键设计**:

#### 输出格式要求
在 JSON 中添加 `reasoning` 对象，包含 5 个推理步骤：
1. `query_analysis`: 分析问题意图、主体、关键信息
2. `domain_identification`: 判断领域和具体场景
3. `time_calculation`: （可选）展示详细的时间计算过程
4. `refuse_check`: 明确检查拒答条件
5. `final_decision`: 综合分析给出最终决策

#### 推理步骤详细说明
为每个步骤提供：
- 明确的目标和要求
- 具体的操作方法
- 示例说明

#### 增强的示例
提供 6 个详细示例，每个都包含完整的 reasoning 过程：
1. 单个计划（基础查询）
2. 时间计算示例（近三年）
3. 非个人问题（健康咨询）
4. 拒答示例（预测类问题）
5. 多个计划（不同 domain）
6. 多时间段对比（同一 domain）

**关键改进点**:
- 强调"先推理再输出"的流程
- 在时间计算步骤中要求展示详细过程
- 在拒答检查中要求明确说明是否符合拒答条件
- 所有示例都包含详细的 reasoning 过程

**时间**: 2025-10-19 完成

---

### 3. 运行脚本修改 ✅

**文件**: `scripts/run_eval.py`

**主要修改**:

#### 添加命令行参数
```python
parser.add_argument(
    "--enable-cot",
    action="store_true",
    help="启用 Chain-of-Thought 推理（使用 v6 CoT prompt）"
)
```

#### Prompt 选择逻辑
```python
if args.enable_cot:
    cot_prompt_path = str(.../ "queryplan_system_prompt_v6_cot.j2")
    system_new = render_system_prompt(cot_prompt_path, today=today)
else:
    system_new = render_system_prompt(args.new_prompt, today=today)
```

#### CSV 输出扩展
启用 CoT 时，添加 5 个 reasoning 相关列：
- `reasoning_query_analysis`
- `reasoning_domain_identification`
- `reasoning_time_calculation`
- `reasoning_refuse_check`
- `reasoning_final_decision`

#### 数据提取
从 `parsed.reasoning` 对象中提取各个步骤并写入 CSV

**时间**: 2025-10-19 完成

---

### 4. Lint 检查与修复 ✅

**发现的问题**:
1. `run_eval.py`: `reasoning_fields` 在异常分支未初始化
2. `schemas.py`: 类型推断问题

**修复方案**:
1. 在异常处理分支添加 `reasoning_fields = {}`
2. 在 `normalize_result` 中显式声明 `result: dict = {}`

**验证**: 所有 lint 错误已清除

**时间**: 2025-10-19 完成

---

## 🚀 使用方法

### 基本用法（不启用 CoT）
```bash
python scripts/run_eval.py \
  --data data/summary_train_v3.xlsx \
  --n 50 \
  --outdir outputs/v5_no_cot
```

### 启用 CoT
```bash
python scripts/run_eval.py \
  --data data/summary_train_v3.xlsx \
  --n 50 \
  --enable-cot \
  --outdir outputs/v6_cot
```

### 参数说明
- `--enable-cot`: 启用 Chain-of-Thought 推理
- `--data`: 数据集路径
- `--n`: 评估样本数量
- `--outdir`: 输出目录
- `--model`: 使用的模型（默认: qwen-flash）
- `--temperature`: 采样温度（默认: 0.0）

### 输出文件

#### 标准输出（无 CoT）
- `eval_results.csv`: 包含 idx, variant, query, raw_response, ok, type, n_plans, latency_sec, parsed, gold_label, error
- `summary.txt`: 统计摘要
- `diffs.csv`: new vs old 的差异

#### CoT 输出（启用 --enable-cot）
在标准输出基础上，`eval_results.csv` 额外包含：
- `reasoning_query_analysis`
- `reasoning_domain_identification`
- `reasoning_time_calculation`
- `reasoning_refuse_check`
- `reasoning_final_decision`

---

## 📊 预期效果分析

### CoT 能够改善的场景

#### 1. 复杂时间计算
**无 CoT**:
- 可能直接给出错误的时间区间
- 缺少计算过程，难以发现错误

**有 CoT**:
- 展示"基准日期 → 计算步骤 → 最终结果"
- 模型被迫逐步计算，减少计算错误
- 可追溯和验证每一步

#### 2. 拒答判断
**无 CoT**:
- 可能误拒答正常查询（如健康咨询）
- 或者遗漏应该拒答的情况

**有 CoT**:
- 在 `refuse_check` 步骤明确检查拒答条件
- 对不拒答的情况也要说明理由
- 减少误判

#### 3. 多域/多时间段问题
**无 CoT**:
- 可能混淆不同 domain 的处理方式
- 时间对比可能错误拆分或合并

**有 CoT**:
- 在 `domain_identification` 明确识别涉及的所有 domain
- 在 `final_decision` 明确说明是拆分还是合并

#### 4. 边界情况
**无 CoT**:
- 容易输出未来日期
- 可能忽略特殊场景规则（如睡眠场景）

**有 CoT**:
- 在推理过程中多次强调约束条件
- 模型更可能遵守规则

---

## 🔍 下一步：测试与验证

### 小规模测试
```bash
# 测试 10 个样本，快速验证
python scripts/run_eval.py \
  --data data/summary_train_v3.xlsx \
  --n 10 \
  --enable-cot \
  --outdir outputs/cot_test_10
```

### 对比实验
```bash
# v5 无 CoT (50 samples)
python scripts/run_eval.py \
  --data data/summary_train_v3.xlsx \
  --n 50 \
  --outdir outputs/v5_baseline

# v6 有 CoT (50 samples)
python scripts/run_eval.py \
  --data data/summary_train_v3.xlsx \
  --n 50 \
  --enable-cot \
  --outdir outputs/v6_cot

# 对比结果
python scripts/llm_judge.py \
  --old-csv outputs/v5_baseline/eval_results.csv \
  --new-csv outputs/v6_cot/eval_results.csv \
  --outdir outputs/v5_vs_v6_analysis
```

### 分析维度
1. **准确率对比**: CoT vs 无 CoT 的整体准确率
2. **时间计算准确性**: 特别关注涉及时间计算的样本
3. **拒答准确性**: 是否减少误拒答和漏拒答
4. **推理质量**: 人工抽查 reasoning 的逻辑性和完整性
5. **延迟影响**: CoT 是否显著增加推理时间

---

## 📈 预期结果

### 性能预期
- **准确率提升**: 预期在复杂查询上提升 5-10%
- **拒答准确性**: 误拒答率降低 20-30%
- **时间计算**: 错误率降低 30-40%
- **延迟增加**: 预期增加 10-30% 的推理时间（因为输出更长）

### 风险与限制
1. **Token 消耗增加**: reasoning 会增加输出 token 数量
2. **格式遵守性**: 部分模型可能难以严格遵守复杂的 JSON 格式
3. **过度推理**: 模型可能在简单问题上也进行复杂推理，影响效率

---

## 💡 后续优化方向

### 1. 自适应 CoT
根据问题复杂度决定是否启用 CoT：
- 简单问题（如"我今天的步数"）→ 无 CoT
- 复杂问题（如多时间段对比）→ 启用 CoT

### 2. 推理步骤优化
- 合并相关性强的步骤
- 针对特定场景定制推理模板

### 3. Few-Shot CoT
在 prompt 中提供 2-3 个详细的 CoT 示例，帮助模型理解推理方式

### 4. 推理验证
添加自动验证逻辑，检查 reasoning 与最终输出的一致性

---

## ✅ 完成清单

- [x] 设计 CoT 结构（ReasoningSteps schema）
- [x] 修改 QueryResult schema 添加 reasoning 字段
- [x] 创建 v6 CoT prompt
- [x] 修改 run_eval.py 支持 CoT
- [x] 添加 --enable-cot 命令行参数
- [x] 扩展 CSV 输出包含 reasoning 字段
- [x] 修复 lint 错误
- [x] 编写使用文档

---

## 🎯 总结

通过引入 Chain-of-Thought (CoT) 推理，我们为 QueryPlan-LLM 系统添加了显式的推理能力。主要改进包括：

1. **结构化推理**: 5 步推理框架覆盖从问题分析到最终决策的完整过程
2. **最小化改动**: 兼容现有的 Outlines 结构化输出机制
3. **可选启用**: 通过 `--enable-cot` 参数灵活控制
4. **完整记录**: reasoning 过程完整保存在 CSV 中，便于分析和调试

下一步需要进行实际测试，对比 v5（无 CoT）和 v6（有 CoT）的效果，并根据结果进一步优化。

---

## 🧪 测试验证

### 基础功能测试 ✅

**测试脚本**: `scripts/test_cot_basic.py`

**运行命令**:
```bash
uv run python scripts/test_cot_basic.py
```

**测试结果** (2025-10-19):
```
============================================================
CoT 功能基础测试
============================================================

测试 1: Schema 定义 ✅
- QueryResult with reasoning 创建成功
- 从 JSON 解析成功
- reasoning.query_analysis 正确提取

测试 2: CoT Prompt 渲染 ✅
- CoT Prompt 渲染成功
- Prompt 长度: 11944 字符
- 包含 'reasoning': True
- 包含 'query_analysis': True
- 包含 'Chain-of-Thought': True

测试 3: JSON 示例有效性 ✅
- JSON 示例验证成功
- Plans 数量: 1
- Reasoning 存在: True
- Query Analysis 正确解析

============================================================
✅ 所有基础测试通过！
============================================================
```

**结论**: 
- Schema 定义正确，可以创建和解析带 reasoning 的 QueryResult
- CoT Prompt 正确渲染，包含所有必要的指令
- JSON 示例格式正确，符合 Pydantic schema

---

## 📚 快速开始指南

### 1. 验证环境
```bash
# 确保在项目根目录
cd /path/to/LLM-Agent-queryplan-eval

# 运行基础测试
uv run python scripts/test_cot_basic.py
```

### 2. 小规模测试（5 样本）
```bash
# 测试 CoT 功能（需要 API key）
uv run python scripts/run_eval.py \
  --data data/summary_train_v3.xlsx \
  --n 5 \
  --enable-cot \
  --outdir outputs/cot_quick_test
```

### 3. 查看结果
```bash
# 查看 CSV 输出（包含 reasoning 列）
head -n 3 outputs/cot_quick_test/eval_results.csv

# 查看统计摘要
cat outputs/cot_quick_test/summary.txt
```

### 4. 对比测试
```bash
# 无 CoT baseline
uv run python scripts/run_eval.py \
  --data data/summary_train_v3.xlsx \
  --n 20 \
  --outdir outputs/v5_baseline_20

# 有 CoT
uv run python scripts/run_eval.py \
  --data data/summary_train_v3.xlsx \
  --n 20 \
  --enable-cot \
  --outdir outputs/v6_cot_20

# 对比分析（使用 LLM judge）
uv run python scripts/llm_judge.py \
  --old-csv outputs/v5_baseline_20/eval_results.csv \
  --new-csv outputs/v6_cot_20/eval_results.csv \
  --outdir outputs/cot_comparison
```

---

## 🔧 故障排查

### 问题 1: ModuleNotFoundError
```
ModuleNotFoundError: No module named 'queryplan_eval'
```

**解决方案**: 使用 `uv run` 而不是直接运行 python
```bash
uv run python scripts/test_cot_basic.py
```

### 问题 2: API Key 未设置
```
Missing API key. Set qwen_key in .env or OPENAI_API_KEY.
```

**解决方案**: 在项目根目录创建 `.env` 文件
```bash
echo "qwen_key=your_api_key_here" > .env
```

### 问题 3: 数据集路径错误
```
FileNotFoundError: data/summary_train_v3.xlsx
```

**解决方案**: 确认数据集路径
```bash
ls -l data/summary_train_v3.xlsx
```

---

## 📝 实现文件清单

### 核心文件
1. ✅ `src/queryplan_eval/schemas.py` - 添加 ReasoningSteps 和修改 QueryResult
2. ✅ `src/queryplan_eval/prompts/queryplan_system_prompt_v6_cot.j2` - CoT prompt
3. ✅ `scripts/run_eval.py` - 支持 --enable-cot 参数
4. ✅ `scripts/test_cot_basic.py` - 基础功能测试

### 文档文件
5. ✅ `docs/ai/03_cot_prompt.md` - 完整的设计文档和使用指南

### 无需修改的文件
- `src/queryplan_eval/renderer.py` - 无需修改，已支持新模板
- `src/queryplan_eval/datasets/` - 无需修改
- `scripts/llm_judge.py` - 无需修改，可直接用于对比分析

---

## 🎓 技术要点

### 1. 为什么选择在 Schema 中添加 reasoning？
- **兼容性**: 与现有的 Outlines 结构化输出机制无缝集成
- **单次调用**: 避免两阶段方法（先推理再输出）的额外开销
- **类型安全**: Pydantic schema 提供自动验证和类型检查
- **可选性**: reasoning 字段为 Optional，不影响旧版本

### 2. 为什么设计 5 步推理？
- **覆盖全流程**: 从问题分析到最终决策的完整链路
- **模块化**: 每步有明确职责，便于调试和优化
- **灵活性**: time_calculation 可选，适应不同问题类型
- **可解释**: 每步输出可追溯，便于理解模型决策

### 3. 如何处理 CoT 增加的 Token 开销？
- **可选启用**: 通过 `--enable-cot` 参数控制，简单问题可不启用
- **有针对性**: 重点在复杂问题（时间计算、拒答判断）上使用
- **未来优化**: 可以实现自适应 CoT（根据问题复杂度动态决定）

---

## 📊 预期下一步

### 立即可做
1. ✅ 基础功能验证（已完成）
2. 🔄 小规模评估（5-10 样本）
3. 🔄 中等规模对比（50 样本）
4. 🔄 分析 reasoning 质量

### 后续计划
1. 大规模评估（200+ 样本）
2. 针对性优化（根据错误模式调整 prompt）
3. 自适应 CoT 实现
4. Few-Shot CoT 实验

