"""RAGTruthTask 单元测试"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from queryplan_eval.tasks import RAGTruthTask
from queryplan_eval.schemas import HallucinationResult
from queryplan_eval.metrics.ragtruth_metrics import (
    compute_hallucination_metrics,
    aggregate_metrics_by_task,
    compute_overall_metrics,
)


class TestRAGTruthMetrics:
    """测试指标计算函数"""
    
    def test_compute_hallucination_metrics_perfect_match(self):
        """测试完美匹配的指标计算"""
        metrics = compute_hallucination_metrics(
            predicted_spans=[(0, 10)],
            ground_truth_spans=[(0, 10)]
        )
        
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
    
    def test_compute_hallucination_metrics_partial_overlap(self):
        """测试部分重叠的指标计算"""
        metrics = compute_hallucination_metrics(
            predicted_spans=[(0, 10)],
            ground_truth_spans=[(5, 15)]
        )
        
        assert metrics["precision"] == 0.5  # 5 chars overlap / 10 predicted
        assert metrics["recall"] == 0.5      # 5 chars overlap / 10 ground truth
        assert metrics["f1"] == 0.5
    
    def test_compute_hallucination_metrics_empty_spans(self):
        """测试空 spans 的指标计算"""
        metrics = compute_hallucination_metrics(
            predicted_spans=[],
            ground_truth_spans=[]
        )
        
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
    
    def test_compute_hallucination_metrics_no_match(self):
        """测试完全不匹配的指标计算"""
        metrics = compute_hallucination_metrics(
            predicted_spans=[(0, 5)],
            ground_truth_spans=[(10, 15)]
        )
        
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0
    
    def test_aggregate_metrics_by_task(self):
        """测试按任务类型聚合指标"""
        results = [
            {
                "task_type": "Summary",
                "precision": 0.9,
                "recall": 0.8,
                "f1": 0.85,
            },
            {
                "task_type": "Summary",
                "precision": 0.85,
                "recall": 0.9,
                "f1": 0.875,
            },
            {
                "task_type": "QA",
                "precision": 0.8,
                "recall": 0.8,
                "f1": 0.8,
            },
        ]
        
        aggregated = aggregate_metrics_by_task(results)
        
        assert "Summary" in aggregated
        assert "QA" in aggregated
        assert aggregated["Summary"]["count"] == 2
        assert aggregated["QA"]["count"] == 1
        assert abs(aggregated["Summary"]["precision"] - 0.875) < 0.001
        assert abs(aggregated["Summary"]["f1"] - 0.8625) < 0.001
    
    def test_compute_overall_metrics(self):
        """测试整体指标计算"""
        results = [
            {"precision": 0.9, "recall": 0.8, "f1": 0.85},
            {"precision": 0.8, "recall": 0.9, "f1": 0.85},
        ]
        
        overall = compute_overall_metrics(results)
        
        assert abs(overall["precision"] - 0.85) < 0.001
        assert abs(overall["recall"] - 0.85) < 0.001
        assert abs(overall["f1"] - 0.85) < 0.001
        assert overall["count"] == 2


class TestRAGTruthTaskInitialization:
    """测试 RAGTruthTask 初始化"""
    
    def test_invalid_task_type(self):
        """测试无效的任务类型"""
        with pytest.raises(ValueError, match="task_type 必须是"):
            RAGTruthTask(
                task_types=["InvalidTask"],
                llm=Mock(),
                output_dir="/tmp/test"
            )
    
    def test_conflicting_sampling_parameters(self):
        """测试冲突的采样参数"""
        with pytest.raises(ValueError, match="sample_n 和 sample_ratio 不能同时指定"):
            RAGTruthTask(
                task_types=["Summary"],
                sample_n=100,
                sample_ratio=0.2,
                llm=Mock(),
                output_dir="/tmp/test"
            )
    
    @patch('queryplan_eval.tasks.ragtruth_task.RAGTruthDataset')
    def test_single_task_initialization(self, mock_dataset_class):
        """测试单任务初始化"""
        # 模拟数据集
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = Mock(return_value=iter([]))
        mock_dataset.__len__ = Mock(return_value=0)
        mock_dataset_class.return_value = mock_dataset
        
        task = RAGTruthTask(
            task_types=["Summary"],
            split="test",
            llm=Mock(),
            output_dir="/tmp/test_summary"
        )
        
        assert task.task_types == ["Summary"]
        assert task.split == "test"
        assert task.use_cot == False
        
        # 验证数据集被加载
        mock_dataset_class.assert_called()
    
    @patch('queryplan_eval.tasks.ragtruth_task.RAGTruthDataset')
    def test_multi_task_initialization(self, mock_dataset_class):
        """测试多任务初始化"""
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = Mock(return_value=iter([]))
        mock_dataset.__len__ = Mock(return_value=0)
        mock_dataset_class.return_value = mock_dataset
        
        task = RAGTruthTask(
            task_types=["Summary", "QA"],
            split="train",
            use_cot=True,
            llm=Mock(),
            output_dir="/tmp/test_multi"
        )
        
        assert task.task_types == ["Summary", "QA"]
        assert task.split == "train"
        assert task.use_cot == True
        
        # 验证数据集被加载两次（每个任务一次）
        assert mock_dataset_class.call_count >= 2
    
    @patch('queryplan_eval.tasks.ragtruth_task.RAGTruthDataset')
    def test_sampling_parameters(self, mock_dataset_class):
        """测试采样参数"""
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = Mock(return_value=iter([]))
        mock_dataset.__len__ = Mock(return_value=1000)
        mock_dataset.select = Mock(return_value=mock_dataset)
        mock_dataset_class.return_value = mock_dataset
        
        # 测试固定数量采样
        task_n = RAGTruthTask(
            task_types=["Summary"],
            sample_n=100,
            llm=Mock(),
            output_dir="/tmp/test_sample_n"
        )
        
        assert task_n.sample_n == 100
        assert task_n.sample_ratio is None
        
        # 测试比例采样
        task_ratio = RAGTruthTask(
            task_types=["Summary"],
            sample_ratio=0.2,
            llm=Mock(),
            output_dir="/tmp/test_sample_ratio"
        )
        
        assert task_ratio.sample_n is None
        assert task_ratio.sample_ratio == 0.2


class TestRAGTruthTaskProcessing:
    """测试 RAGTruthTask 的结果处理"""
    
    def test_hallucination_result_parsing(self):
        """测试幻觉结果解析"""
        result = HallucinationResult(
            hallucination_list=["spans", "substrings"]
        )
        
        assert len(result.hallucination_list) == 2
        assert "spans" in result.hallucination_list
    
    def test_hallucination_result_empty(self):
        """测试空幻觉结果"""
        result = HallucinationResult(hallucination_list=[])
        
        assert len(result.hallucination_list) == 0


class TestRAGTruthTaskOutputGeneration:
    """测试 RAGTruthTask 的输出生成"""
    
    def test_output_directory_creation(self, tmp_path):
        """测试输出目录创建"""
        output_dir = tmp_path / "test_output"
        
        with patch('queryplan_eval.tasks.ragtruth_task.RAGTruthDataset'):
            task = RAGTruthTask(
                task_types=["Summary"],
                llm=Mock(),
                output_dir=str(output_dir)
            )
        
        assert output_dir.exists()
    
    def test_metrics_computation_structure(self):
        """测试指标计算的数据结构"""
        results = [
            {
                "task_type": "Summary",
                "ok": True,
                "precision": 0.9,
                "recall": 0.8,
                "f1": 0.85,
            },
            {
                "task_type": "QA",
                "ok": True,
                "precision": 0.8,
                "recall": 0.9,
                "f1": 0.85,
            },
        ]
        
        # 计算指标
        overall = compute_overall_metrics(results)
        by_task = aggregate_metrics_by_task(results)
        
        # 验证数据结构
        assert "precision" in overall
        assert "recall" in overall
        assert "f1" in overall
        assert "count" in overall
        
        assert "Summary" in by_task
        assert "QA" in by_task
        assert "count" in by_task["Summary"]
        assert "count" in by_task["QA"]


class TestRAGTruthTaskPromptBuilding:
    """测试 RAGTruthTask 的 Prompt 构建"""
    
    @patch('queryplan_eval.tasks.ragtruth_task.RAGTruthDataset')
    @patch('queryplan_eval.tasks.ragtruth_task.RAGPromptManager')
    def test_build_chat_for_summary(self, mock_pm_class, mock_dataset_class):
        """测试 Summary 任务的 chat 构建"""
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = Mock(return_value=iter([]))
        mock_dataset.__len__ = Mock(return_value=0)
        mock_dataset_class.return_value = mock_dataset
        
        mock_pm = MagicMock()
        mock_pm.get_prompt.return_value = "test prompt"
        mock_pm_class.return_value = mock_pm
        
        task = RAGTruthTask(
            task_types=["Summary"],
            llm=Mock(),
            output_dir="/tmp/test_chat"
        )
        
        item = {
            "task_type": "Summary",
            "context": "original text",
            "output": "summary text",
            "query": None,
        }
        
        chat = task.build_chat(item)
        
        assert len(chat) == 2
        assert chat[0]["role"] == "system"
        assert chat[1]["role"] == "user"
        
        # 验证 Prompt Manager 被调用
        mock_pm.get_prompt.assert_called_once()
    
    @patch('queryplan_eval.tasks.ragtruth_task.RAGTruthDataset')
    @patch('queryplan_eval.tasks.ragtruth_task.RAGPromptManager')
    def test_build_chat_for_qa(self, mock_pm_class, mock_dataset_class):
        """测试 QA 任务的 chat 构建（需要 question）"""
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = Mock(return_value=iter([]))
        mock_dataset.__len__ = Mock(return_value=0)
        mock_dataset_class.return_value = mock_dataset
        
        mock_pm = MagicMock()
        mock_pm.get_prompt.return_value = "test prompt"
        mock_pm_class.return_value = mock_pm
        
        task = RAGTruthTask(
            task_types=["QA"],
            llm=Mock(),
            output_dir="/tmp/test_chat_qa"
        )
        
        item = {
            "task_type": "QA",
            "context": "reference paragraph",
            "output": "answer text",
            "query": "what is this?",
        }
        
        chat = task.build_chat(item)
        
        assert len(chat) == 2
        
        # 验证 question 被传递给 Prompt Manager
        call_kwargs = mock_pm.get_prompt.call_args[1]
        assert "question" in call_kwargs


class TestRAGTruthTaskIntegration:
    """集成测试"""
    
    @patch('queryplan_eval.tasks.ragtruth_task.RAGTruthDataset')
    def test_complete_workflow(self, mock_dataset_class, tmp_path):
        """测试完整工作流程"""
        # 创建模拟数据
        mock_items = [
            {
                "idx": 0,
                "task_type": "Summary",
                "context": "original",
                "output": "summary",
                "hallucination_labels": "[]",
                "query": None,
            }
        ]
        
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = Mock(return_value=iter(mock_items))
        mock_dataset.__len__ = Mock(return_value=1)
        mock_dataset.select = Mock(return_value=mock_dataset)
        mock_dataset_class.return_value = mock_dataset
        
        output_dir = tmp_path / "test_workflow"
        
        task = RAGTruthTask(
            task_types=["Summary"],
            split="test",
            llm=Mock(),
            output_dir=str(output_dir)
        )
        
        # 验证初始化
        assert task.dataset is not None
        assert len(task.task_types) == 1
        assert output_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
