"""肽段专利数据集模块

提供 HuggingFace 风格的数据集加载和访问接口，用于加载肽段相关的专利数据。
支持类型安全的数据访问和高级数据操作（如 map、filter 等）。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from datasets import Dataset

logger = logging.getLogger(__name__)


@dataclass
class PeptideItem:
    """肽段数据项
    
    表示数据集中的单个肽段记录及其对应的序列信息。
    支持通过属性访问数据，便于类型检查和 IDE 自动补全。
    
    Attributes:
        idx: 数据的索引（行号）
        sequence: 肽段序列信息，包含特征、长度、类型等元数据的字典
    """

    idx: int
    sequence: dict[str, Any]


class PeptideDataset:
    """HuggingFace 风格的肽段数据集
    
    从 CSV 文件加载肽段专利数据，提供类似 HuggingFace Datasets
    的高级操作接口（如 map、filter 等），同时支持类型安全的数据访问。
    
    SEQUENCE 列中的 JSON 字符串会被自动解析为 Python 字典，便于后续处理。
    
    Example:
        >>> dataset = PeptideDataset("data/patents/US11111272B2.csv", n=50)
        >>> len(dataset)
        50
        >>> item = dataset[0]
        >>> print(item.idx, item.sequence['length'])
        >>> for item in dataset:
        ...     print(item.sequence.get('sequence'))
    """

    def __init__(self, csv_path: str, n: Optional[int] = None) -> None:
        """初始化数据集
        
        Args:
            csv_path: CSV 文件路径
            n: 返回行数。如果为 None 则返回全部；如果指定，使用固定随机种子
               随机采样 n 行以确保可复现性。
            
        Raises:
            FileNotFoundError: 如果 CSV 文件不存在
            ValueError: 如果 CSV 缺少必需的列（SEQUENCE）或 SEQUENCE 解析失败
        """
        csv_path_obj = Path(csv_path)
        if not csv_path_obj.exists():
            raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

        # 加载 CSV 并规范化列名
        df = pd.read_csv(csv_path_obj)
        cols = [c.strip().upper() for c in df.columns]
        df.columns = cols

        # 检查必需的列
        if "SEQUENCE" not in df.columns:
            raise ValueError(
                f"CSV 缺少 'SEQUENCE' 列。可用列: {df.columns.tolist()}"
            )

        # 采样或使用全部数据
        if n is not None:
            df = df.sample(n=n, random_state=42)
            logger.info(f"从 {csv_path} 中随机采样了 {n} 个肽段")
        else:
            logger.info(f"从 {csv_path} 中加载了全部 {len(df)} 个肽段")

        # 转换为 HuggingFace Dataset 所需的字典格式
        sequences = []
        for row in df["SEQUENCE"]:
            try:
                # 将 JSON 字符串解析为 Python 字典
                seq_dict = json.loads(row) if isinstance(row, str) else row
                sequences.append(seq_dict)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(
                    f"无法解析 SEQUENCE 字段: {str(row)[:100]}... "
                    f"错误: {e}"
                )
                # 对于无法解析的行，使用空字典作为占位符
                sequences.append({})

        data_dicts = {
            "idx": list(range(len(df))),
            "sequence": sequences,
        }

        # 创建内部的 HuggingFace Dataset
        self._dataset = Dataset.from_dict(data_dicts)

    def __len__(self) -> int:
        """返回数据集中的样本数量"""
        return len(self._dataset)

    def __getitem__(self, idx: int) -> PeptideItem:
        """获取指定索引的数据项
        
        Args:
            idx: 数据项的索引（0 到 len-1）
            
        Returns:
            PeptideItem 对象，包含 idx 和 sequence 字段
            
        Raises:
            IndexError: 如果索引超出范围
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"索引 {idx} 超出范围 [0, {len(self) - 1}]")

        row = self._dataset[idx]
        return PeptideItem(idx=row["idx"], sequence=row["sequence"])

    def __iter__(self):
        """迭代数据集中的所有项
        
        Yields:
            PeptideItem: 数据集中的每一个数据项
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
