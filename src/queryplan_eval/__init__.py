"""
QueryPlan-LLM 评测工具包。

本包提供了一套完整的查询计划抽取和评测框架，支持：
- 从自然语言查询中稳定抽取结构化 Plan（domain, sub, is_personal, time, food）
- 支持 A/B 测试两个提示词变体
- 使用 Pydantic + Outlines 保证输出结构
- 与 OpenAI 兼容的 API（如 DashScope/Qwen）集成
- 批量处理支持，通过 Batch API 降低成本
"""

__version__ = "0.1.0"
__author__ = "SFB Lab, KAUST"

# 导出核心模块
from . import data_utils, renderer, schemas
from .datasets import QueryPlanDataset, QueryPlanItem
from .batch_handler import (
    BatchRequest,
    BatchResult,
    BatchRequestBuilder,
    BatchResponseProcessor,
    BatchExecutor,
    batch_split,
)

__all__ = [
    "data_utils",
    "renderer",
    "schemas",
    "QueryPlanDataset",
    "QueryPlanItem",
    "BatchRequest",
    "BatchResult",
    "BatchRequestBuilder",
    "BatchResponseProcessor",
    "BatchExecutor",
    "batch_split",
]
