"""肽段专利数据集测试模块"""

from __future__ import annotations

import pytest
from pathlib import Path

from queryplan_eval.datasets.peptide_patent import PeptideDataset, PeptideItem


class TestPeptideItem:
    """PeptideItem 数据项的测试"""

    def test_peptide_item_creation(self) -> None:
        """测试 PeptideItem 的创建"""
        seq_dict = {"length": "10", "type": "PRT", "sequence": "Xaa Arg Ser"}
        item = PeptideItem(idx=0, sequence=seq_dict)
        
        assert item.idx == 0
        assert item.sequence == seq_dict
        assert item.sequence["length"] == "10"


class TestPeptideDataset:
    """PeptideDataset 数据集的测试"""

    @pytest.fixture
    def csv_path(self) -> str:
        """获取测试用的 CSV 文件路径"""
        data_dir = Path(__file__).resolve().parent.parent / "data" / "patents"
        return str(data_dir / "US11111272B2_SIF_SGF_RawData_with_sequence.csv")

    def test_dataset_loading(self, csv_path: str) -> None:
        """测试数据集加载"""
        if not Path(csv_path).exists():
            pytest.skip(f"测试数据文件不存在: {csv_path}")
        
        dataset = PeptideDataset(csv_path)
        assert len(dataset) > 0

    def test_dataset_getitem(self, csv_path: str) -> None:
        """测试通过索引访问数据项"""
        if not Path(csv_path).exists():
            pytest.skip(f"测试数据文件不存在: {csv_path}")
        
        dataset = PeptideDataset(csv_path)
        item = dataset[0]
        
        assert isinstance(item, PeptideItem)
        assert isinstance(item.idx, int)
        assert isinstance(item.sequence, dict)

    def test_dataset_iteration(self, csv_path: str) -> None:
        """测试数据集迭代"""
        if not Path(csv_path).exists():
            pytest.skip(f"测试数据文件不存在: {csv_path}")
        
        dataset = PeptideDataset(csv_path, n=2)
        items = list(dataset)
        
        assert len(items) == 2
        for item in items:
            assert isinstance(item, PeptideItem)

    def test_dataset_sampling(self, csv_path: str) -> None:
        """测试数据采样"""
        if not Path(csv_path).exists():
            pytest.skip(f"测试数据文件不存在: {csv_path}")
        
        dataset1 = PeptideDataset(csv_path, n=3)
        dataset2 = PeptideDataset(csv_path, n=3)
        
        # 相同的 random_state 应该得到相同的数据
        assert len(dataset1) == len(dataset2) == 3
        assert dataset1[0].idx == dataset2[0].idx

    def test_dataset_sequence_is_dict(self, csv_path: str) -> None:
        """测试 SEQUENCE 字段被正确解析为字典"""
        if not Path(csv_path).exists():
            pytest.skip(f"测试数据文件不存在: {csv_path}")
        
        dataset = PeptideDataset(csv_path)
        item = dataset[0]
        
        # 验证 sequence 是字典且包含关键字段
        assert isinstance(item.sequence, dict)
        # 根据实际 CSV 数据，应该包含 'features', 'length', 'type' 等字段
        assert len(item.sequence) > 0

    def test_dataset_index_error(self, csv_path: str) -> None:
        """测试索引超出范围时的异常"""
        if not Path(csv_path).exists():
            pytest.skip(f"测试数据文件不存在: {csv_path}")
        
        dataset = PeptideDataset(csv_path)
        
        with pytest.raises(IndexError):
            _ = dataset[len(dataset)]
        
        with pytest.raises(IndexError):
            _ = dataset[-1]

    def test_dataset_file_not_found(self) -> None:
        """测试文件不存在时的异常"""
        with pytest.raises(FileNotFoundError):
            PeptideDataset("nonexistent_file.csv")

    def test_dataset_missing_sequence_column(self, tmp_path) -> None:
        """测试缺少 SEQUENCE 列时的异常"""
        # 创建一个没有 SEQUENCE 列的临时 CSV
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("SEQ_ID,SIF\n1,value")
        
        with pytest.raises(ValueError, match="SEQUENCE"):
            PeptideDataset(str(csv_file))
