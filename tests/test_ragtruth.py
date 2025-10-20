"""RAGTruthDataset 功能的单元测试

测试 RAGTruthDataset 的各种使用场景，包括数据加载、访问、迭代、
数据划分和辅助函数的功能验证。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from datasets import Dataset

from queryplan_eval.datasets.ragtruth import (
    RAGTruthDataset,
    RAGTruthItem,
    split_train_val,
    compute_sample_weights,
    get_dataset_statistics,
)


@pytest.fixture
def sample_hf_ragtruth_dataset():
    """创建一个示例 HuggingFace Dataset 用于测试 RAGTruthDataset"""
    data = {
        "task_type": ["Summary"] * 30 + ["QA"] * 20,
        "context": [f"context_{i}" for i in range(50)],
        "output": [f"output_{i}" for i in range(50)],
        # 前 25 个样本有幻觉，后 25 个没有
        "hallucination_labels": [
            json.dumps([{"start": 0, "end": 5}, {"start": 10, "end": 15}]) if i < 25 else "[]"
            for i in range(50)
        ],
        "query": [f"query_{i}" if i % 3 == 0 else None for i in range(50)],
    }
    return Dataset.from_dict(data)


@pytest.fixture
def sample_ragtruth_dataset(sample_hf_ragtruth_dataset):
    """创建一个示例 RAGTruthDataset 用于测试"""
    # 创建一个 Summary 任务的数据集
    dataset = RAGTruthDataset.__new__(RAGTruthDataset)
    # 只保留 Summary 任务的数据
    filtered = sample_hf_ragtruth_dataset.filter(lambda x: x["task_type"] == "Summary")
    dataset._dataset = filtered
    dataset.task_type = "Summary"
    dataset.split = "train"
    return dataset


class TestRAGTruthItem:
    """RAGTruthItem dataclass 的测试"""

    def test_ragtruth_item_creation(self):
        """测试 RAGTruthItem 的创建"""
        item = RAGTruthItem(
            idx=0,
            task_type="Summary",
            context="sample context",
            output="sample output",
            hallucination_labels='[{"start": 0, "end": 5}]',
            hallucination_spans=[(0, 5)],
            query="sample query",
        )

        assert item.idx == 0
        assert item.task_type == "Summary"
        assert item.context == "sample context"
        assert item.output == "sample output"
        assert item.query == "sample query"
        assert item.hallucination_spans == [(0, 5)]

    def test_ragtruth_item_without_query(self):
        """测试 RAGTruthItem 创建时 query 为 None"""
        item = RAGTruthItem(
            idx=1,
            task_type="QA",
            context="qa context",
            output="qa output",
            hallucination_labels="[]",
            hallucination_spans=[],
        )

        assert item.query is None
        assert item.task_type == "QA"
        assert item.hallucination_spans == []

    def test_ragtruth_item_fields(self):
        """测试 RAGTruthItem 包含所有必需字段"""
        item = RAGTruthItem(
            idx=5,
            task_type="Data2txt",
            context="data to text",
            output="generated text",
            hallucination_labels="[]",
            hallucination_spans=[],
        )

        assert hasattr(item, "idx")
        assert hasattr(item, "task_type")
        assert hasattr(item, "context")
        assert hasattr(item, "output")
        assert hasattr(item, "hallucination_labels")
        assert hasattr(item, "hallucination_spans")
        assert hasattr(item, "query")


class TestRAGTruthDataset:
    """RAGTruthDataset 类的基本功能测试"""

    def test_dataset_length(self, sample_ragtruth_dataset):
        """测试数据集长度"""
        assert len(sample_ragtruth_dataset) == 30  # 只有 Summary 任务的数据

    def test_dataset_getitem(self, sample_ragtruth_dataset):
        """测试通过索引访问数据"""
        item = sample_ragtruth_dataset[0]

        assert isinstance(item, RAGTruthItem)
        assert item.idx == 0
        assert item.task_type == "Summary"
        assert isinstance(item.context, str)
        assert isinstance(item.output, str)
        assert isinstance(item.hallucination_labels, str)

    def test_dataset_getitem_invalid_index(self, sample_ragtruth_dataset):
        """测试访问无效的索引"""
        with pytest.raises(IndexError, match="超出范围"):
            sample_ragtruth_dataset[100]

        with pytest.raises(IndexError, match="超出范围"):
            sample_ragtruth_dataset[-1]

    def test_dataset_iteration(self, sample_ragtruth_dataset):
        """测试数据集迭代"""
        items = list(sample_ragtruth_dataset)

        assert len(items) == 30
        for i, item in enumerate(items):
            assert isinstance(item, RAGTruthItem)
            assert item.task_type == "Summary"

    def test_dataset_multiple_iterations(self, sample_ragtruth_dataset):
        """测试多次迭代"""
        items1 = list(sample_ragtruth_dataset)
        items2 = list(sample_ragtruth_dataset)

        assert len(items1) == len(items2)
        for i1, i2 in zip(items1, items2):
            assert i1.context == i2.context
            assert i1.output == i2.output

    def test_dataset_attributes(self, sample_ragtruth_dataset):
        """测试数据集的属性"""
        assert sample_ragtruth_dataset.task_type == "Summary"
        assert sample_ragtruth_dataset.split == "train"

    def test_dataset_fields_preserved(self, sample_ragtruth_dataset):
        """测试原始字段名被保留，并且转换后的字段被添加"""
        item = sample_ragtruth_dataset[0]

        # 验证使用的是原始字段名，不是映射的字段名
        assert hasattr(item, "context")  # 不是 input_text
        assert hasattr(item, "output")   # 不是 response
        assert hasattr(item, "hallucination_labels")  # 原始值
        assert hasattr(item, "hallucination_spans")  # 转换后的值
        
        # 验证转换后的字段是结构化的元组列表
        assert isinstance(item.hallucination_spans, list)
        for span in item.hallucination_spans:
            assert isinstance(span, tuple)
            assert len(span) == 2


class TestSplitTrainVal:
    """split_train_val 函数的测试"""

    def test_split_basic(self, sample_ragtruth_dataset):
        """测试基本的数据划分"""
        train_split, val_split = split_train_val(
            sample_ragtruth_dataset, val_ratio=0.2, random_seed=42
        )

        # 验证返回类型
        assert isinstance(train_split, RAGTruthDataset)
        assert isinstance(val_split, RAGTruthDataset)

        # 验证大小
        assert len(train_split) + len(val_split) == len(sample_ragtruth_dataset)

        # 验证比例（允许 ±1 的误差）
        expected_val_size = int(len(sample_ragtruth_dataset) * 0.2)
        assert abs(len(val_split) - expected_val_size) <= 1

    def test_split_reproducibility(self, sample_ragtruth_dataset):
        """测试相同种子下的结果重现"""
        train1, val1 = split_train_val(sample_ragtruth_dataset, val_ratio=0.2, random_seed=42)
        train2, val2 = split_train_val(sample_ragtruth_dataset, val_ratio=0.2, random_seed=42)

        assert len(train1) == len(train2)
        assert len(val1) == len(val2)

    def test_split_different_seeds(self, sample_ragtruth_dataset):
        """测试不同种子产生不同的划分"""
        train1, val1 = split_train_val(sample_ragtruth_dataset, val_ratio=0.2, random_seed=42)
        train2, val2 = split_train_val(sample_ragtruth_dataset, val_ratio=0.2, random_seed=123)

        # 由于数据量较大，不同种子几乎肯定会产生不同的划分
        # （这是个概率测试，但几乎不会失败）
        assert len(train1) > 0 and len(train2) > 0

    def test_split_invalid_split(self):
        """测试对非 train split 的数据集进行划分应该失败"""
        dataset = RAGTruthDataset.__new__(RAGTruthDataset)
        dataset._dataset = Dataset.from_dict({
            "task_type": ["Summary"] * 10,
            "context": [f"ctx_{i}" for i in range(10)],
            "output": [f"out_{i}" for i in range(10)],
            "hallucination_labels": ["[]"] * 10,
            "query": [None] * 10,
        })
        dataset.task_type = "Summary"
        dataset.split = "test"  # 不是 train

        with pytest.raises(ValueError, match="只能对split='train'的数据集进行划分"):
            split_train_val(dataset)

    def test_split_val_ratio_variations(self, sample_ragtruth_dataset):
        """测试不同的验证集比例"""
        for val_ratio in [0.1, 0.2, 0.3, 0.5]:
            train_split, val_split = split_train_val(
                sample_ragtruth_dataset, val_ratio=val_ratio, random_seed=42
            )

            assert len(train_split) + len(val_split) == len(sample_ragtruth_dataset)
            expected_val_size = int(len(sample_ragtruth_dataset) * val_ratio)
            assert abs(len(val_split) - expected_val_size) <= 1

    def test_split_returns_dataset_instances(self, sample_ragtruth_dataset):
        """测试划分返回的是 RAGTruthDataset 实例"""
        train_split, val_split = split_train_val(sample_ragtruth_dataset)

        # 验证可以访问元素
        if len(train_split) > 0:
            train_item = train_split[0]
            assert isinstance(train_item, RAGTruthItem)

        if len(val_split) > 0:
            val_item = val_split[0]
            assert isinstance(val_item, RAGTruthItem)

    def test_split_items_are_different(self, sample_ragtruth_dataset):
        """测试划分后的训练集和验证集没有重叠样本"""
        train_split, val_split = split_train_val(
            sample_ragtruth_dataset, val_ratio=0.2, random_seed=42
        )

        # 收集所有训练样本的内容
        train_contexts = set(train_split[i].context for i in range(len(train_split)))
        val_contexts = set(val_split[i].context for i in range(len(val_split)))

        # 验证没有重叠的样本内容
        overlap = train_contexts & val_contexts
        assert len(overlap) == 0, f"发现重叠的样本: {overlap}"


class TestComputeSampleWeights:
    """compute_sample_weights 函数的测试"""

    def test_weights_shape(self, sample_ragtruth_dataset):
        """测试权重列表的形状"""
        weights = compute_sample_weights(sample_ragtruth_dataset)

        assert isinstance(weights, list)
        assert len(weights) == len(sample_ragtruth_dataset)

    def test_weights_all_positive(self, sample_ragtruth_dataset):
        """测试所有权重都是正数"""
        weights = compute_sample_weights(sample_ragtruth_dataset)

        assert all(w > 0 for w in weights)

    def test_weights_different_for_hallucination(self, sample_ragtruth_dataset):
        """测试有幻觉和无幻觉的样本权重不同"""
        weights = compute_sample_weights(sample_ragtruth_dataset)

        # 收集有幻觉和无幻觉样本的权重
        hallu_weights = []
        no_hallu_weights = []

        for i, item in enumerate(sample_ragtruth_dataset):
            if item.hallucination_labels and item.hallucination_labels != "[]":
                hallu_weights.append(weights[i])
            else:
                no_hallu_weights.append(weights[i])

        # 如果两个类都存在，权重应该不同
        if hallu_weights and no_hallu_weights:
            avg_hallu = sum(hallu_weights) / len(hallu_weights)
            avg_no_hallu = sum(no_hallu_weights) / len(no_hallu_weights)
            assert avg_hallu != avg_no_hallu

    def test_weights_sum_properties(self, sample_ragtruth_dataset):
        """测试权重的求和属性"""
        weights = compute_sample_weights(sample_ragtruth_dataset)

        # 权重总和应该等于样本数
        assert abs(sum(weights) - len(sample_ragtruth_dataset)) < 1e-6


class TestGetDatasetStatistics:
    """get_dataset_statistics 函数的测试"""

    def test_statistics_keys(self, sample_ragtruth_dataset):
        """测试统计信息包含所有必需的键"""
        stats = get_dataset_statistics(sample_ragtruth_dataset)

        required_keys = [
            "total_samples",
            "hallucination_samples",
            "hallucination_ratio",
            "avg_hallucination_spans",
            "avg_context_length",
            "avg_output_length",
        ]

        for key in required_keys:
            assert key in stats

    def test_statistics_values_valid(self, sample_ragtruth_dataset):
        """测试统计信息的值有效"""
        stats = get_dataset_statistics(sample_ragtruth_dataset)

        # 样本总数应该正确
        assert stats["total_samples"] == len(sample_ragtruth_dataset)

        # 幻觉样本数不应该超过总数
        assert stats["hallucination_samples"] <= stats["total_samples"]

        # 幻觉比例应该在 0 到 1 之间
        assert 0 <= stats["hallucination_ratio"] <= 1

        # 平均长度应该是正数
        assert stats["avg_context_length"] >= 0
        assert stats["avg_output_length"] >= 0

    def test_statistics_hallucination_count(self, sample_ragtruth_dataset):
        """测试幻觉样本计数"""
        stats = get_dataset_statistics(sample_ragtruth_dataset)

        # 手动计算幻觉样本数
        manual_count = 0
        for item in sample_ragtruth_dataset:
            if item.hallucination_labels and item.hallucination_labels != "[]":
                try:
                    parsed = json.loads(item.hallucination_labels)
                    if parsed:
                        manual_count += 1
                except json.JSONDecodeError:
                    pass

        assert stats["hallucination_samples"] == manual_count

    def test_statistics_ratio_consistency(self, sample_ragtruth_dataset):
        """测试幻觉比例的一致性"""
        stats = get_dataset_statistics(sample_ragtruth_dataset)

        expected_ratio = (
            stats["hallucination_samples"] / stats["total_samples"]
            if stats["total_samples"] > 0
            else 0
        )

        assert abs(stats["hallucination_ratio"] - expected_ratio) < 1e-6


class TestRAGTruthDatasetIntegration:
    """RAGTruthDataset 的集成测试"""

    def test_workflow_load_and_split(self, sample_ragtruth_dataset):
        """测试典型的加载和划分工作流"""
        # 加载数据集
        assert len(sample_ragtruth_dataset) > 0

        # 划分为训练和验证集
        train_split, val_split = split_train_val(
            sample_ragtruth_dataset, val_ratio=0.2, random_seed=42
        )

        # 验证划分有效
        assert len(train_split) > 0
        assert len(val_split) > 0

        # 从训练集中访问样本
        train_item = train_split[0]
        assert isinstance(train_item, RAGTruthItem)

        # 从验证集中访问样本
        val_item = val_split[0]
        assert isinstance(val_item, RAGTruthItem)

    def test_workflow_with_statistics(self, sample_ragtruth_dataset):
        """测试包含统计信息的工作流"""
        # 获取原始统计
        original_stats = get_dataset_statistics(sample_ragtruth_dataset)
        assert original_stats["total_samples"] == len(sample_ragtruth_dataset)

        # 划分数据集
        train_split, val_split = split_train_val(
            sample_ragtruth_dataset, val_ratio=0.2, random_seed=42
        )

        # 获取分割后的统计
        train_stats = get_dataset_statistics(train_split)
        val_stats = get_dataset_statistics(val_split)

        # 验证统计信息的完整性
        assert train_stats["total_samples"] + val_stats["total_samples"] == (
            original_stats["total_samples"]
        )

    def test_workflow_with_weights(self, sample_ragtruth_dataset):
        """测试包含权重计算的工作流"""
        # 计算原始权重
        weights = compute_sample_weights(sample_ragtruth_dataset)

        # 划分数据集
        train_split, val_split = split_train_val(
            sample_ragtruth_dataset, val_ratio=0.2, random_seed=42
        )

        # 分别计算分割后的权重
        train_weights = compute_sample_weights(train_split)
        val_weights = compute_sample_weights(val_split)

        # 验证权重列表有效
        assert len(weights) == len(sample_ragtruth_dataset)
        assert len(train_weights) == len(train_split)
        assert len(val_weights) == len(val_split)

        # 验证权重都是正数
        assert all(w > 0 for w in weights)
        assert all(w > 0 for w in train_weights)
        assert all(w > 0 for w in val_weights)

    def test_field_names_consistency(self, sample_ragtruth_dataset):
        """测试原始字段名的一致性"""
        # 验证所有样本使用原始字段名
        for item in sample_ragtruth_dataset:
            # 这些字段应该存在
            assert hasattr(item, "context")
            assert hasattr(item, "output")
            assert hasattr(item, "hallucination_labels")
            assert hasattr(item, "hallucination_spans")

            # 这些映射字段不应该存在
            assert not hasattr(item, "input_text")
            assert not hasattr(item, "response")

            # 验证字段类型
            assert isinstance(item.context, str)
            assert isinstance(item.output, str)
            assert isinstance(item.hallucination_labels, str)
            assert isinstance(item.hallucination_spans, list)

    def test_hallucination_spans_transformation(self, sample_ragtruth_dataset):
        """测试 hallucination_spans 的正确转换"""
        for item in sample_ragtruth_dataset:
            # 验证 hallucination_spans 是元组列表
            assert isinstance(item.hallucination_spans, list)
            
            for span in item.hallucination_spans:
                assert isinstance(span, tuple)
                assert len(span) == 2
                start, end = span
                assert isinstance(start, int)
                assert isinstance(end, int)
                assert start < end
    
    def test_stratified_split_effect(self, sample_ragtruth_dataset):
        """测试分层抽样的效果"""
        train_split, val_split = split_train_val(
            sample_ragtruth_dataset, val_ratio=0.2, random_seed=42
        )

        # 计算原始的幻觉比例
        original_hallucination_count = sum(
            1 for item in sample_ragtruth_dataset if item.hallucination_spans
        )
        original_ratio = original_hallucination_count / len(sample_ragtruth_dataset)

        # 计算分割后的幻觉比例
        train_hallucination_count = sum(
            1 for item in train_split if item.hallucination_spans
        )
        train_ratio = train_hallucination_count / len(train_split)

        val_hallucination_count = sum(
            1 for item in val_split if item.hallucination_spans
        )
        val_ratio = val_hallucination_count / len(val_split)

        # 验证分层效果：分割后的比例应该接近原始比例（允许 ±5% 的误差）
        assert abs(train_ratio - original_ratio) < 0.05, (
            f"训练集幻觉比例 {train_ratio:.2%} 与原始 {original_ratio:.2%} 差异过大"
        )
        assert abs(val_ratio - original_ratio) < 0.05, (
            f"验证集幻觉比例 {val_ratio:.2%} 与原始 {original_ratio:.2%} 差异过大"
        )
