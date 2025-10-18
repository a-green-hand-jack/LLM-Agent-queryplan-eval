"""Pytest 全局配置和 fixtures。"""

import sys
from pathlib import Path

import pytest

# 将 src 目录添加到 Python 路径，以便导入项目模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


@pytest.fixture
def sample_query() -> dict:
    """提供示例查询数据。"""
    return {
        "history": [],
        "question": "我昨天的睡眠质量怎么样？",
    }


@pytest.fixture
def sample_data_dir() -> Path:
    """提供示例数据目录路径。"""
    return project_root / "data"
