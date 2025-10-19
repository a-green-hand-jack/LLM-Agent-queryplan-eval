"""评估结果数据集"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from datasets import Dataset

logger = logging.getLogger(__name__)


class EvalResultsDataset:
    """评估结果数据集
    
    用于加载和处理评估结果 CSV，提供数据访问接口
    """
    
    def __init__(self, csv_path: str):
        """初始化评估结果数据集
        
        Args:
            csv_path: 评估结果 CSV 文件路径
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"文件不存在: {csv_path}")
        
        # 加载 CSV 到内存
        self._df = pd.read_csv(csv_path)
        logger.info(f"加载评估结果: {len(self._df)} 行")
        
        # 转换为 HuggingFace Dataset
        self._dataset = Dataset.from_pandas(self._df)
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self._dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取指定索引的记录"""
        return self._dataset[idx]
    
    def __iter__(self):
        """迭代数据集"""
        for i in range(len(self)):
            yield self[i]
    
    def get_row_by_idx(self, data_idx: int) -> pd.Series:
        """按数据中的 idx 列获取行
        
        Args:
            data_idx: 数据中的 idx 列值
            
        Returns:
            对应的 pandas Series
            
        Raises:
            IndexError: 如果 idx 不存在
        """
        rows = self._df[self._df['idx'] == data_idx]
        if len(rows) == 0:
            raise IndexError(f"idx {data_idx} not found in dataset")
        return rows.iloc[0]
