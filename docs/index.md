# QueryPlan-LLM 评测工具文档

欢迎来到 QueryPlan-LLM 项目！本文档提供了项目的完整信息。

## 📚 文档导航

### 核心文档
- [Prompt 工程蓝图](prompt_design.md) - 详细的 Prompt 设计规范和规则定义

### 快速开始
详见项目根目录的 [README.md](../README.md)

## 🎯 项目概述

QueryPlan-LLM 是一个 **Prompt A/B 评测工具**，用于：

1. **结构化 Plan 抽取** - 从自然语言查询中稳定抽取 5 个关键字段：
   - `domain`: 领域（health/exercise）
   - `sub`: 子类别
   - `is_personal`: 是否针对本人
   - `time`: 时间范围
   - `food`: 涉及的食物

2. **Prompt 对比评测** - 对比两个提示词变体在结构化抽取上的表现

3. **高稳定性设计** - 通过强约束系统提示词 + 规则化抽取 + 结构化生成保证一致性

## 🔧 核心特性

- ✅ 使用 Pydantic 定义清晰的数据模式
- ✅ 集成 Outlines 实现结构化生成
- ✅ 支持 OpenAI 兼容 API（DashScope/Qwen）
- ✅ Jinja2 模板系统提示词
- ✅ 完整的 A/B 评测管道

## 📋 项目结构

```
src/queryplan_eval/          # 核心包
  ├── data_utils.py          # 数据加载与处理
  ├── renderer.py            # 提示词渲染
  ├── schemas.py             # Pydantic 数据模型
  ├── run_eval.py            # 评测主流程
  └── prompts/               # 提示词模板
      ├── original_system_prompt.txt
      └── queryplan_system_prompt.j2

tests/                        # 测试代码
docs/                        # 项目文档
```

## 📖 后续阅读

- 详细的 Prompt 规范和规则定义：[prompt_design.md](prompt_design.md)
- 项目使用说明：[../README.md](../README.md)
