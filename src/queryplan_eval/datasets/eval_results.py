"""评估结果数据集"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from datasets import Dataset

logger = logging.getLogger(__name__)


class EvalResultsDataset:
    """评估结果数据集
    
    用于加载和处理 run_eval 生成的 eval_results.csv，
    提供 pairwise 比较所需的数据访问接口
    """
    
    def __init__(self, csv_path: str):
        """初始化评估结果数据集
        
        Args:
            csv_path: 评估结果 CSV 文件路径
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"文件不存在: {csv_path}")
        
        # 加载 CSV
        df = pd.read_csv(csv_path)
        logger.info(f"加载评估结果: {len(df)} 行")
        
        # 转换为 HuggingFace Dataset
        self._dataset = Dataset.from_pandas(df)
    
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
    
    def prepare_comparison_pairs(self) -> List[Dict[str, Any]]:
        """准备需要比较的样本对
        
        从 eval_results 中提取 new 和 old 变体，组成对比样本对
        
        Returns:
            样本对列表，每个包含 idx, query, gold_label, new_response, old_response
        """
        # 将数据按 idx 分组
        data = pd.read_csv(self.csv_path)
        grouped = data.groupby("idx")
        
        pairs = []
        for idx, group in grouped:
            # 检查是否有 new 和 old 两个变体
            if "variant" not in group.columns:
                logger.warning(f"跳过 idx={idx}，数据中没有 variant 列")
                continue
            
            variants = group["variant"].unique()
            if len(variants) < 2:
                logger.debug(f"跳过 idx={idx}，只有单个变体")
                continue
            
            try:
                # 提取 new 和 old 行
                new_row = group[group["variant"] == "new"].iloc[0]
                old_row = group[group["variant"] == "old"].iloc[0]
            except IndexError:
                logger.debug(f"跳过 idx={idx}，无法找到 new 或 old 变体")
                continue
            
            # 跳过两者都失败的情况
            if not new_row.get("ok", False) and not old_row.get("ok", False):
                logger.debug(f"跳过 idx={idx}，两个候选都失败")
                continue
            
            # 构造样本对
            pair = {
                "idx": int(idx),
                "query": str(new_row.get("query", "")),
                "gold_label": str(new_row.get("gold_label", "")) if pd.notna(new_row.get("gold_label")) else "{}",
                "new_response": str(new_row.get("raw_response", "")) if pd.notna(new_row.get("raw_response")) else "{}",
                "old_response": str(old_row.get("raw_response", "")) if pd.notna(old_row.get("raw_response")) else "{}",
                "new_ok": bool(new_row.get("ok", False)),
                "old_ok": bool(old_row.get("ok", False))
            }
            pairs.append(pair)
        
        logger.info(f"准备了 {len(pairs)} 个比较样本对")
        return pairs
