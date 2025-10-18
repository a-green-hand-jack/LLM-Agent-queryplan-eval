from __future__ import annotations
import pandas as pd
from typing import Iterable

REQUIRED_COLS = ["query"]

def load_queries(xlsx_path: str, n: int | None = None) -> pd.DataFrame:
    """加载查询数据，仅返回 query 列
    
    Args:
        xlsx_path: Excel 文件路径
        n: 返回行数，如果为 None 则返回全部
        
    Returns:
        包含 query 列的 DataFrame
    """
    df = pd.read_excel(xlsx_path)
    # 规范化列名
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    assert "query" in df.columns, f"Excel missing 'query' column. Got: {df.columns.tolist()}"
    if n:
        df = df.head(n)
    return df[["query"]].copy()


def load_queries_with_gold_labels(xlsx_path: str, n: int | None = None) -> pd.DataFrame:
    """加载查询数据及其金标签（预期的计划输出）
    
    Args:
        xlsx_path: Excel 文件路径
        n: 返回行数，如果为 None 则返回全部
        
    Returns:
        包含 query 和 plan 列的 DataFrame
    """
    df = pd.read_excel(xlsx_path)
    # 规范化列名
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    assert "query" in df.columns, f"Excel missing 'query' column. Got: {df.columns.tolist()}"
    
    # 检查 plan 列是否存在
    if "plan" not in df.columns:
        raise ValueError(f"Excel missing 'plan' column for gold labels. Got: {df.columns.tolist()}")
    
    if n:
        df = df.head(n)
    
    return df[["query", "plan"]].copy()
