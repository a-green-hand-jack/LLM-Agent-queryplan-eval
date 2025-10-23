"""查,计划数据集模块

提供 HuggingFace 风格的数据集加载和访问接口，支持类型安全的数据访问
和高级数据操作（如 map、filter 等）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from datasets import Dataset

logger = logging.getLogger(__name__)


@dataclass
class QueryPlanItem:
    """查询计划数据项
    
    表示数据集中的单个查询及其对应的金标签计划。
    支持通过属性访问数据，便于类型检查和 IDE 自动补全。
    
    Attributes:
        idx: 数据的索引（行号）
        query: 查询文本
        plan: 预期的查询计划（可选，如果数据中不存在则为 None）
    """

    idx: int
    query: str
    plan: Optional[str]


class QueryPlanDataset:
    """HuggingFace 风格的查询计划数据集
    
    从 Excel 文件加载查询和金标签数据，提供类似 HuggingFace Datasets
    的高级操作接口（如 map、filter 等），同时支持类型安全的数据访问。
    
    Example:
        >>> dataset = QueryPlanDataset("data.xlsx", n=50)
        >>> len(dataset)
        50
        >>> item = dataset[0]
        >>> print(item.query, item.plan)
        >>> for item in dataset:
        ...     print(item.idx, item.query)
    """

    def __init__(self, xlsx_path: str, n: Optional[int] = None) -> None:
        """初始化数据集
        
        Args:
            xlsx_path: Excel 文件路径
            n: 返回行数。如果为 None 则返回全部；如果指定，使用固定随机种子
               随机采样 n 行以确保可复现性。
            
        Raises:
            FileNotFoundError: 如果 Excel 文件不存在
            ValueError: 如果 Excel 缺少必需的列（query 或 plan）
        """
        xlsx_path_obj = Path(xlsx_path)
        if not xlsx_path_obj.exists():
            raise FileNotFoundError(f"Excel 文件不存在: {xlsx_path}")

        # 加载 Excel 并规范化列名
        df = pd.read_excel(xlsx_path)
        cols = [c.strip().lower() for c in df.columns]
        df.columns = cols

        # 检查必需的列
        if "query" not in df.columns:
            raise ValueError(
                f"Excel 缺少 'query' 列。可用列: {df.columns.tolist()}"
            )
        if "plan" not in df.columns:
            raise ValueError(
                f"Excel 缺少 'plan' 列。可用列: {df.columns.tolist()}"
            )

        # 采样或使用全部数据
        if n is not None:
            df = df.sample(n, random_state=42)
            logger.info(f"从 {xlsx_path} 中随机采样了 {n} 个查询")
        else:
            logger.info(f"从 {xlsx_path} 中加载了全部 {len(df)} 个查询")

        # 转换为 HuggingFace Dataset 所需的字典格式
        data_dicts = {
            "idx": list(range(len(df))),
            "query": [str(row).strip() for row in df["query"]],
            "plan": [
                str(row).strip() if not pd.isna(row) else None for row in df["plan"]
            ],
        }

        # 创建内部的 HuggingFace Dataset
        self._dataset = Dataset.from_dict(data_dicts)

    def __len__(self) -> int:
        """返回数据集中的样本数量"""
        return len(self._dataset)

    def __getitem__(self, idx: int) -> QueryPlanItem:
        """获取指定索引的数据项
        
        Args:
            idx: 数据项的索引（0 到 len-1）
            
        Returns:
            QueryPlanItem 对象，包含 idx、query 和 plan 字段
            
        Raises:
            IndexError: 如果索引超出范围
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"索引 {idx} 超出范围 [0, {len(self) - 1}]")

        row = self._dataset[idx]
        return QueryPlanItem(
            idx=row["idx"], query=row["query"], plan=row["plan"]
        )

    def __iter__(self):
        """迭代数据集中的所有项
        
        Yields:
            QueryPlanItem: 数据集中的每一个数据项
        """
        for i in range(len(self)):
            yield self[i]

    def __getattr__(self, name: str) -> Any:
        """代理不存在的属性到内部的 HuggingFace Dataset
        
        这允许用户直接调用 HuggingFace Dataset 的方法，如 map、filter 等，
        同时保留自定义的类型安全的 __getitem__ 接口。
        
        Args:
            name: 属性名
            
        Returns:
            HuggingFace Dataset 的属性或方法
            
        Raises:
            AttributeError: 如果属性不存在
        """
        return getattr(self._dataset, name)
