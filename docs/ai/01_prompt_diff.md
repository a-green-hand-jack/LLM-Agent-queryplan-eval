# QueryPlan Prompt A/B 测试计划与执行记录

## 第一部分：项目理解与计划

### 当前状态分析

**数据文件**: `./data/summary_train_v3.xlsx`
- 总列数: 19 列
- 关键列:
  - `query`: 用户查询（输入数据）
  - `plan`: 预期的计划输出（金标签，当前都是 "REFUSE"）
  - 其他列: 相关的上下文信息（domain, suggest, rag 等）

**现有代码**: `scripts/run_eval.py`
- 已有功能:
  - 支持从 Excel 加载查询数据
  - 使用 Outlines 调用 OpenAI 兼容的 API（Qwen）
  - 支持结构化输出（QueryResult Schema）
  - 比较两个 system prompt 的效果
  - 输出评估结果、差异对比

- 已有 Schema (`QueryResult`):
  - `plans`: List[Plan] - 计划列表
  - `refused`: bool - 是否拒绝
  - `refuse_reason`: str - 拒绝原因

### 目标

1. **修改 run_eval.py**:
   - 读取 `./data/summary_train_v3.xlsx` 而非之前的数据文件
   - 使用 `query` 列作为模型输入
   - 预期模型输出与 `plan` 列比对

2. **比较两个 Prompt 的效果**:
   - 两个 system prompt 在相同查询上的输出质量
   - 统计指标: 成功率、拒绝率、错误率、响应延迟
   - 找出两个 prompt 输出不同的查询案例

3. **执行流程**:
   - 从 Excel 加载一定数量的查询样本
   - 对每个查询，分别使用新旧 prompt 调用模型
   - 记录所有结果（原始输出、解析结果、执行时间）
   - 生成对比报告

### 计划步骤

#### Step 1: 数据加载与集成 ✅
- [x] 确认 Excel 文件结构和数据质量
- [ ] 修改 `data_utils.py` 的 `load_queries()` 函数以支持更灵活的列选择
- [ ] 或直接在 `run_eval.py` 中扩展数据加载逻辑

#### Step 2: 代码修改与适配
- [ ] 更新 `run_eval.py` 命令行参数
- [ ] 实现新的数据加载逻辑
- [ ] 适配现有的评估流程

#### Step 3: 本地测试执行
- [ ] 使用小样本（n=5-10）进行测试
- [ ] 验证模型调用、数据流、结果保存
- [ ] 检查输出格式正确性

#### Step 4: 完整运行与结果分析
- [ ] 使用完整数据集运行评估（或 n=50）
- [ ] 分析 prompt 效果差异
- [ ] 生成对比分析报告

#### Step 5: 文档更新
- [ ] 记录执行结果和关键发现
- [ ] 更新本文档的结果部分

---

## 第二部分：实现与执行

### Step 1: 数据加载分析

**Excel 数据特性**:
- 总行数: 427 行（不含表头）
- `query` 列: 用户查询，主要是关于健身、睡眠、健康监测等个人数据查询
- `plan` 列: **预期的结构化输出**，格式为 JSON 或 "REFUSE"
  - **正常计划** (占多数): JSON 格式，包含 domain、time、query 数组、is_personal 等字段
  - **拒绝** (36 条): "REFUSE" 字符串
  - **319 种不同的 JSON 变体**: 针对不同领域和查询类型的结构化计划

**数据分布统计**:
- plan 列值: 319 种不同值（几乎每条记录都不同）
- 拒绝比例: 36/427 ≈ 8.4%
- 主要领域: 睡眠、跑步、步行、体温、心脏健康等个人健康指标

**关键认知**: 
> 这是一个**个人健康数据查询数据集**，而非医疗诊断咨询。模型应该：
> 1. 识别查询中的健康指标领域（domain）
> 2. 提取时间范围（time）
> 3. 识别查询的具体内容（query 数组）
> 4. 判断是否涉及个人隐私（is_personal）
> 5. 对于超出能力范围的查询（如"帮我煮杯红糖水"）则拒绝

### Step 2: 代码修改记录

**修改位置**: `scripts/run_eval.py`

#### 2.1 命令行参数调整
- 保持 `--data` 参数指向 `./data/summary_train_v3.xlsx`
- 保持现有的 `-n` 参数

#### 2.2 数据加载修改
```python
# 现有逻辑 (load_queries 只返回 query 列)
df = load_queries(args.data, n=args.n)  # 返回 DataFrame[query]

# 需要扩展: 支持加载 plan 列作为金标签
df = load_queries(args.data, n=args.n)  # 仍然返回 query
# 但需要同时加载 plan 列
```

#### 2.3 评估结果扩展
- 新增 `gold_label` 字段用于存储预期的 plan
- 新增 `system_prompt` 字段保存调用的 system prompt 完整内容
- 新增 `raw_response` 字段保存模型的原始返回值（未解析前）
- 在评估结果 CSV 中对比模型输出与金标签

#### 2.4 输出结果格式

生成的 CSV 包含以下字段：
```
- idx: 查询索引
- variant: 'new' 或 'old'
- query: 用户查询文本
- system_prompt: 完整的 system prompt 内容
- raw_response: 模型原始返回值（字符串）
- ok: 是否成功解析
- type: 输出类型 (plans/refuse/parse_error/exception)
- n_plans: 生成的计划数量
- latency_sec: 响应延迟
- parsed: 解析后的 JSON 对象
- gold_label: 预期的 plan 值
- error: 错误信息
```

### 执行日志

#### 2024-11 实现阶段

**✅ 完成事项**:

1. **数据分析** (2024-11-01)
   - 随机抽取20行样本数据
   - 发现数据特性：427行，319种不同的plan值
   - 确认数据是个人健康数据查询集，而非医疗诊断

2. **代码修改** (2024-11-01)
   - 修改 `data_utils.py`:
     - 新增 `load_queries_with_gold_labels()` 函数
     - 同时加载 query 和 plan 列
   
   - 修改 `scripts/run_eval.py`:
     - 导入新的 `load_queries_with_gold_labels` 函数
     - 修改主循环以加载 gold_label
     - 保存 `system_prompt` 字段（完整的系统提示）
     - 保存 `raw_response` 字段（模型原始返回值）
     - 保存 `gold_label` 字段（预期的 plan 值）
     - 更新结果列名从 `normalized` 改为 `parsed`
     - 更新 pivot 表逻辑

3. **代码验证** (2024-11-01)
   - ✅ 语法检查通过
   - ✅ 数据加载函数测试通过
   - ✅ 成功加载前3行数据并验证列结构

#### 下一步计划

- [ ] 配置 API 密钥和环境变量（.env 文件）
- [ ] 运行 n=5-10 的小样本测试
- [ ] 验证完整的评估流程
- [ ] 运行完整数据集评估（n=50 或更多）
- [ ] 分析 prompt 效果差异
- [ ] 生成最终报告

#### 技术改进 (2024-10-18)

**✅ 流式 CSV 写入** - 提升用户体验:
- 改为逐条记录写入 CSV，每处理完一条就立即保存
- 使用 Python `csv.DictWriter` 替代 DataFrame 批量写入
- 加入 `csv_file.flush()` 确保数据立即写入磁盘
- 优势：
  - 可以实时查看进度和中间结果
  - 即使程序中途中断，也能保留已处理的结果
  - 支持非常大的数据集而不需占用大量内存

**实现细节**:
```python
# 初始化
csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
csv_writer.writeheader()

# 循环中每处理一条记录
csv_writer.writerow(record)
csv_file.flush()  # 立即保存到磁盘
```

#### 技术细节

**新的输出 CSV 结构**:
- `idx`: 查询在数据集中的索引
- `variant`: 'new' 或 'old' 
- `query`: 用户查询文本
- `system_prompt`: 完整的 system prompt 内容（用于审计和调试）
- `raw_response`: 模型原始返回值（未经任何解析）
- `ok`: 是否成功解析
- `type`: 输出类型 (plans/refuse/parse_error/exception)
- `n_plans`: 生成的计划数量
- `latency_sec`: 响应延迟（秒）
- `parsed`: 解析后的 JSON 对象（规范化后的结构）
- `gold_label`: 数据集中的预期输出值
- `error`: 如有错误的错误信息

**使用示例**:
```bash
# 运行小样本测试
uv run python scripts/run_eval.py --data data/summary_train_v3.xlsx --n 5 --temperature 0.0

# 运行完整数据集
uv run python scripts/run_eval.py --data data/summary_train_v3.xlsx --n 50 --temperature 0.0
```
