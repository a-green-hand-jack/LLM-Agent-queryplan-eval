"""
QueryPlan-LLM 评测工具包 - 重构版本

本包提供了一套完整的查询计划抽取和评测框架，支持：
- 从自然语言查询中稳定抽取结构化 Plan（domain, sub, is_personal, time, food）
- 支持 Chain-of-Thought 推理
- 多种 LLM 后端（API 和本地模型）
- 灵活的任务框架，支持快速扩展新任务
- 成对比较工具，用于 prompt 或模型效果评估
"""

__version__ = "0.2.0"
__author__ = "SFB Lab, KAUST"

# 核心组件
from .core import BaseLLM, BaseTask, PromptManager

# LLM 实现
from .llms import OpenAILLM # HuggingFaceLLM

# 任务
from .tasks import QueryPlanTask

# 工具
from .tools import PairwiseJudge

# 数据集
from .datasets import QueryPlanDataset, QueryPlanItem, EvalResultsDataset, SplitConfig, split_dataset, take_samples

# Schema
from .schemas import QueryResult, JudgementResult

# 向后兼容性
from . import data_utils, renderer, schemas
from .batch_handler import (
    BatchRequest,
    BatchResult,
    BatchRequestBuilder,
    BatchResponseProcessor,
    BatchExecutor,
    batch_split,
)

__all__ = [
    # 核心
    "BaseLLM",
    "BaseTask",
    "PromptManager",
    # LLM 实现
    "OpenAILLM",
    # "HuggingFaceLLM",
    # 任务
    "QueryPlanTask",
    # 工具
    "PairwiseJudge",
    # 数据集
    "QueryPlanDataset",
    "QueryPlanItem",
    "EvalResultsDataset",
    "SplitConfig",
    "split_dataset",
    "take_samples",
    # Schema
    "QueryResult",
    "JudgementResult",
    # 向后兼容
    "data_utils",
    "renderer",
    "schemas",
    "BatchRequest",
    "BatchResult",
    "BatchRequestBuilder",
    "BatchResponseProcessor",
    "BatchExecutor",
    "batch_split",
]
