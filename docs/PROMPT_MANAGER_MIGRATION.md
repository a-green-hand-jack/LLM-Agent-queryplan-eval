# Prompt Manager 升级指南

本文档说明如何从旧的 prompt manager 迁移到新的生产级 prompt-manager 系统。

## 概述

项目已升级到使用生产级的 [prompt-manager](https://github.com/a-green-hand-jack/PromptManager) 库，提供以下优势：

- **版本管理**: 支持多版本共存，轻松进行 A/B 测试和回滚
- **高性能缓存**: 多级缓存系统，提升 50-90% 的渲染性能
- **类型安全**: 参数验证和类型转换
- **开发友好**: 开发模式支持热重载
- **统一接口**: 所有任务使用统一的 API

## 目录结构变化

### 旧结构
```
src/queryplan_eval/
├── prompts/
│   ├── query_plan/
│   │   ├── v5.j2
│   │   ├── v6_cot.j2
│   ├── ragtruth/
│   │   ├── templates/
│   │   └── prompt_config.yaml
│   ├── patent/
│   └── judgement/
└── core/
    └── prompt_manager.py  # 旧的实现
```

### 新结构
```
prompts/                    # 项目根目录下
├── query_plan/
│   ├── configs/
│   │   └── query_plan.yaml  # 配置文件
│   └── templates/
│       ├── system/          # 系统提示词
│       │   ├── v5.jinja2
│       │   ├── v6_cot.jinja2
│       │   └── ...
│       ├── user/            # 用户提示词（可选）
│       └── common/          # 可重用组件（可选）
├── ragtruth/
│   ├── configs/
│   │   └── ragtruth.yaml
│   └── templates/
│       └── system/
│           ├── summarization_with_cot.jinja2
│           ├── summarization_without_cot.jinja2
│           └── ...
├── patent/
│   ├── configs/
│   │   └── patent.yaml
│   └── templates/
│       └── system/
│           ├── v1.jinja2
│           └── v2.jinja2
└── judgement/
    ├── configs/
    │   └── judgement.yaml
    └── templates/
        └── system/
            └── v1.jinja2
```

## 代码迁移示例

### 1. 旧代码（Query Plan）

```python
from queryplan_eval.core.prompt_manager import PromptManager

# 旧的用法
manager = PromptManager(task_name="query_plan", version="v6_cot")
prompt = manager.load(today="2025年10月30日")
```

### 新代码（Query Plan）

```python
from queryplan_eval.core.unified_prompt_manager import UnifiedPromptManager

# 新的用法
manager = UnifiedPromptManager()
messages = manager.render_query_plan(
    version="v6_cot",
    today="2025年10月30日"
)

# 获取 LLM 配置
llm_config = manager.get_query_plan_llm_config()
```

### 2. 旧代码（RAGTruth）

```python
from queryplan_eval.core.prompt_manager import RAGPromptManager

# 旧的用法
manager = RAGPromptManager()
prompt = manager.get_prompt(
    task_type="Summary",
    use_cot=True,
    reference="原文...",
    response="摘要..."
)
```

### 新代码（RAGTruth）

```python
from queryplan_eval.core.unified_prompt_manager import UnifiedPromptManager

# 新的用法
manager = UnifiedPromptManager()
messages = manager.render_ragtruth(
    task_type="Summary",
    use_cot=True,
    reference="原文...",
    response="摘要..."
)

# 获取 LLM 配置
llm_config = manager.get_ragtruth_llm_config()
```

### 3. 旧代码（Patent）

```python
from queryplan_eval.core.prompt_manager import PatentPromptManager

# 旧的用法
manager = PatentPromptManager(version="v2")
prompt = manager.load()
```

### 新代码（Patent）

```python
from queryplan_eval.core.unified_prompt_manager import UnifiedPromptManager

# 新的用法
manager = UnifiedPromptManager()
messages = manager.render_patent(
    version="v2",
    sequence="Ala Ser Lys",
    features=[...]
)

# 获取 LLM 配置
llm_config = manager.get_patent_llm_config()
```

## 主要 API 变化

### 初始化

**旧:**
```python
# 每个任务需要单独初始化
query_manager = PromptManager("query_plan", "v6_cot")
rag_manager = RAGPromptManager()
patent_manager = PatentPromptManager("v2")
```

**新:**
```python
# 统一的 manager 管理所有任务
manager = UnifiedPromptManager(
    dev_mode=False,        # 生产模式
    enable_cache=True      # 启用缓存
)
```

### 渲染提示词

**旧:**
```python
# 返回字符串
prompt = manager.load(**params)
```

**新:**
```python
# 返回 OpenAI 格式的消息列表
messages = manager.render_query_plan(**params)
# [{"role": "system", "content": "..."}]
```

### LLM 配置

**新增功能:**
```python
# 获取每个任务的 LLM 配置
config = manager.get_query_plan_llm_config()
# {'model': 'gpt-4', 'temperature': 0.7, ...}

# 直接用于 OpenAI API
import openai
response = openai.chat.completions.create(
    model=config["model"],
    messages=messages,
    temperature=config["temperature"]
)
```

## 配置文件格式

每个任务的配置文件 (`configs/{task_name}.yaml`) 包含：

```yaml
metadata:
  name: task_name
  description: "任务描述"
  author: "作者"
  current_version: "v1"
  tags: ["tag1", "tag2"]

parameters:
  param_name:
    type: str  # str, int, float, bool, list, dict
    required: true
    default: null
    description: "参数描述"

llm_config:
  model: "gpt-4"
  temperature: 0.7
  max_tokens: 2000
  response_format:
    type: "json_object"
```

## 高级功能

### 1. 开发模式（热重载）

```python
# 开发模式：禁用缓存，自动重新加载
manager = UnifiedPromptManager(dev_mode=True)
```

在开发模式下，对配置文件或模板的修改会立即生效，无需重启应用。

### 2. 缓存管理

```python
# 获取缓存统计
stats = manager.cache_stats()
print(stats)

# 清除所有缓存
manager.clear_cache()

# 重新加载所有配置
manager.reload()
```

### 3. 版本管理

```python
# 列出所有可用版本
versions = manager.list_versions("query_plan", "system")
print(versions)  # ['v5', 'v6_cot', 'original', ...]

# 使用不同版本进行 A/B 测试
messages_v5 = manager.render_query_plan(version="v5", **params)
messages_v6 = manager.render_query_plan(version="v6_cot", **params)
```

### 4. 全局单例

```python
from queryplan_eval.core.unified_prompt_manager import get_unified_manager

# 获取全局单例
manager = get_unified_manager()

# 在其他地方使用同一个实例
manager = get_unified_manager()  # 返回相同的实例
```

## 测试

运行测试脚本验证迁移：

```bash
uv run python tests/test_unified_prompt_manager.py
```

测试覆盖：
- ✅ Query Plan 任务（7个版本）
- ✅ RAGTruth 任务（6个变体）
- ✅ Patent 任务（2个版本）
- ✅ Judgement 任务
- ✅ 缓存功能
- ✅ 版本管理

## 性能优化

新系统带来的性能提升：

1. **模板缓存**: 编译后的 Jinja2 模板被缓存，避免重复解析
2. **渲染缓存**: 相同参数的渲染结果被缓存
3. **参数验证**: 提前验证和转换参数类型
4. **懒加载**: 配置和模板按需加载

预期性能提升：**50-90%** （取决于缓存命中率）

## 常见问题

### Q: 旧的 prompt_manager.py 还能用吗？

A: 可以，但建议尽快迁移到新系统。旧代码位于 `src/queryplan_eval/core/prompt_manager.py`，不会被删除，以保证向后兼容。

### Q: 如何添加新的任务？

A:
1. 在 `prompts/` 下创建任务目录
2. 创建 `configs/{task_name}.yaml` 配置文件
3. 创建 `templates/system/{version}.jinja2` 模板
4. 在 `UnifiedPromptManager` 中添加任务特定的方法（可选）

### Q: 模板文件必须是 .jinja2 后缀吗？

A: 是的，新系统使用 `.jinja2` 后缀，而不是旧的 `.j2`。这是为了与 prompt-manager 库保持一致。

### Q: 可以在模板中使用哪些 Jinja2 功能？

A: 支持所有标准 Jinja2 功能：
- 变量替换: `{{ variable }}`
- 条件语句: `{% if condition %}...{% endif %}`
- 循环: `{% for item in list %}...{% endfor %}`
- 包含: `{% include 'common/component.jinja2' %}`
- 过滤器: `{{ text | upper }}`

### Q: 如何调试模板渲染错误？

A:
1. 启用开发模式: `UnifiedPromptManager(dev_mode=True)`
2. 查看详细的错误堆栈信息
3. 使用测试脚本单独测试: `tests/test_unified_prompt_manager.py`

## 贡献指南

如需添加新功能或修复问题，请：

1. 在 `prompt_manager` 分支上开发
2. 更新相关配置和模板
3. 添加或更新测试用例
4. 更新文档
5. 提交 Pull Request

## 相关资源

- [Prompt-Manager 官方文档](https://github.com/a-green-hand-jack/PromptManager)
- [Jinja2 文档](https://jinja.palletsprojects.com/)
- [项目测试脚本](../tests/test_unified_prompt_manager.py)
- [UnifiedPromptManager 源码](../src/queryplan_eval/core/unified_prompt_manager.py)

## 总结

新的 prompt manager 系统提供了：

✅ **更好的性能**: 50-90% 的性能提升
✅ **版本管理**: 轻松进行 A/B 测试和回滚
✅ **统一接口**: 所有任务使用相同的 API
✅ **类型安全**: 自动参数验证和转换
✅ **开发友好**: 热重载和清晰的错误信息
✅ **生产就绪**: 经过实战检验的设计模式

欢迎开始使用新系统！如有问题，请查阅本文档或联系维护者。
