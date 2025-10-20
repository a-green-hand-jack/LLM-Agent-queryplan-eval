<!-- 1c5af75d-0b8a-4008-9eff-65b7116d1cf0 53a871e5-9bd8-4287-8862-d06ac6a1bd79 -->
# RAGTruth Dataset 数据转换改进计划

## 问题诊断

### 旧版本 (RL4HS) 的设计

1. **早期转换**: 在数据加载时就进行字段映射（context → input_text, output → response）
2. **信息处理**: 将 JSON 字符串的 hallucination_labels 解析为元组列表，提供结构化数据
3. **集成式设计**: 数据转换、过滤、划分紧密耦合

### 当前版本 (QueryPlan-Eval) 的设计

1. **保留原始字段**: 直接使用 context, output, hallucination_labels
2. **延迟处理**: 保留原始 JSON 字符串，需要时才解析
3. **模块化设计**: 数据加载、访问、转换分离

### 发现的关键问题

1. **缺少核心数据转换**: hallucination_labels 应该转换为 hallucination_spans (List[Tuple[int, int]])
2. **缺少分层抽样**: split_train_val 使用简单随机分割，未考虑幻觉类别不平衡
3. **幻觉判断重复解析**: 在 compute_sample_weights 中每次都重新解析 JSON
4. **缺少幻觉比例验证**: 未验证分层效果

## 改进方案 - 清晰的分层设计

### 第一层：数据转换 (Dataset 层) - 核心改进

在 RAGTruthItem 中添加转换后的字段：

- 保留原始: `hallucination_labels` (JSON 字符串，供参考)
- 添加转换: `hallucination_spans` (List[Tuple[int, int]]，结构化数据)
- 在 RAGTruthDataset.**getitem**() 时完成转换

### 第二层：数据分割 (Dataset 层)

改进 split_train_val：

- 使用已转换的 hallucination_spans 判断是否有幻觉
- 实现分层抽样，基于幻觉类别平衡
- 验证分层效果（对比前后幻觉比例）

### 第三层：业务逻辑 (Task 层) - 不在 Dataset 层

关键决策：`_has_hallucination()` 在 Task 层实现

- 数据层只负责数据转换和访问
- 任务层实现幻觉检测逻辑，基于 hallucination_spans
- 这样保持 Dataset 的纯粹性和可复用性

## 实施步骤

1. **添加 hallucination_spans 转换** 

- 在 RAGTruthItem 中添加 hallucination_spans 字段
- 在 RAGTruthDataset.**getitem**() 中完成 JSON→元组列表的转换
- 使用已有的 _parse_hallucination_labels 函数

2. **改进 split_train_val** 

- 添加分层标签生成逻辑（基于 hallucination_spans 是否非空）
- 使用 sklearn 的 train_test_split 进行分层抽样
- 添加 _verify_stratification 验证分层效果

3. **优化 compute_sample_weights**

- 使用已转换的 hallucination_spans 判断（改为检查列表是否非空）
- 避免重复的 JSON 解析

4. **编写集成测试**

- 验证 hallucination_spans 正确转换
- 验证分层抽样的幻觉比例分布
- 验证权重计算基于转换后的数据

5. **文档说明**

- 明确 Dataset 层的责任：数据加载、转换、访问
- 说明 Task 层的责任：业务逻辑、Prompt 生成、幻觉检测

## 关键决策

- **核心改变**: 添加 hallucination_spans 作为转换后的结构化数据
- **早期转换**: 在 Dataset 访问时就做好转换，提供给下游清晰的数据
- **清晰分层**: Dataset = 数据处理，Task = 业务逻辑
- **向后兼容**: 保留原始 hallucination_labels，方便调试和参考
- **性能考虑**: 在 **getitem**() 时转换一次，避免重复解析

## 文件修改范围

- `src/queryplan_eval/datasets/ragtruth.py` - 添加转换、改进分层
- `tests/test_ragtruth.py` - 添加转换和分层效果验证
- 后续 `src/queryplan_eval/tasks/ragtruth_task.py` 中实现幻觉检测逻辑

### To-dos

- [ ] 添加 RAGTruthItem dataclass 定义和必要的导入
- [ ] 重写 load_ragtruth_from_hf 函数，使用原始字段，返回 HuggingFace Dataset
- [ ] 重写 RAGTruthDataset 类为 HuggingFace 风格，实现类型安全访问
- [ ] 简化 split_train_val 函数，使用 HuggingFace Dataset API
- [ ] 调整辅助函数适配新的数据结构和字段名