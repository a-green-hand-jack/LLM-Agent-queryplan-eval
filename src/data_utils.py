from __future__ import annotations
import pandas as pd
from typing import Iterable

REQUIRED_COLS = ["query"]

def load_queries(xlsx_path: str, n: int | None = None) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    # Normalize expected columns
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    assert "query" in df.columns, f"Excel missing 'query' column. Got: {df.columns.tolist()}"
    if n:
        df = df.head(n)
    return df[["query"]].copy()
