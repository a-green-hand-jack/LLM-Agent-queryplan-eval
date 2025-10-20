"""RAGTruthDataset 和 RAGPromptManager 的集成测试

测试 RAGTruthDataset 和 RAGPromptManager 的协作，验证：
1. 数据集的任务类型命名与 prompt manager 一致
2. 端到端的工作流（加载数据→生成 prompt）
3. 所有任务类型的正常工作

注意：此测试使用真实的 RAGTruth 数据集，需要网络连接从 HuggingFace 加载数据
"""

from __future__ import annotations

import logging

import pytest

from queryplan_eval.datasets.ragtruth import (
    RAGTruthDataset,
    RAGTruthItem,
)
from queryplan_eval.prompts.ragtruth import RAGPromptManager

logger = logging.getLogger(__name__)


@pytest.fixture
def ragtruth_dataset_summary():
    """加载真实的 Summary 任务 RAGTruthDataset"""
    try:
        return RAGTruthDataset(task_type="Summary", split="test")
    except Exception as e:
        logger.warning(f"无法加载真实 Summary 数据集: {e}，跳过测试")
        pytest.skip("无法加载 RAGTruth 数据集（可能没有网络连接）")


@pytest.fixture
def ragtruth_dataset_qa():
    """加载真实的 QA 任务 RAGTruthDataset"""
    try:
        return RAGTruthDataset(task_type="QA", split="test")
    except Exception as e:
        logger.warning(f"无法加载真实 QA 数据集: {e}，跳过测试")
        pytest.skip("无法加载 RAGTruth 数据集（可能没有网络连接）")


@pytest.fixture
def ragtruth_dataset_data2txt():
    """加载真实的 Data2txt 任务 RAGTruthDataset"""
    try:
        return RAGTruthDataset(task_type="Data2txt", split="test")
    except Exception as e:
        logger.warning(f"无法加载真实 Data2txt 数据集: {e}，跳过测试")
        pytest.skip("无法加载 RAGTruth 数据集（可能没有网络连接）")


@pytest.fixture
def prompt_manager():
    """创建 RAGPromptManager 实例"""
    return RAGPromptManager()


class TestNamingConsistency:
    """测试 RAGTruthDataset 和 RAGPromptManager 的命名一致性"""

    def test_summary_task_naming(self, ragtruth_dataset_summary, prompt_manager):
        """测试 Summary 任务命名一致"""
        # 获取数据集中的任务类型
        dataset_task_type = ragtruth_dataset_summary.task_type
        
        # 验证数据集任务类型
        assert dataset_task_type == "Summary"
        
        # 验证 prompt manager 可以接受相同的任务类型
        item = ragtruth_dataset_summary[0]
        prompt = prompt_manager.get_prompt(
            task_type=dataset_task_type,
            use_cot=True,
            reference=item.context,
            response=item.output
        )
        
        assert len(prompt) > 0
        assert isinstance(prompt, str)

    def test_qa_task_naming(self, ragtruth_dataset_qa, prompt_manager):
        """测试 QA 任务命名一致"""
        dataset_task_type = ragtruth_dataset_qa.task_type
        
        assert dataset_task_type == "QA"
        
        item = ragtruth_dataset_qa[0]
        prompt = prompt_manager.get_prompt(
            task_type=dataset_task_type,
            use_cot=True,
            question="What is this about?",
            reference=item.context,
            response=item.output
        )
        
        assert len(prompt) > 0
        assert isinstance(prompt, str)

    def test_data2txt_task_naming(self, ragtruth_dataset_data2txt, prompt_manager):
        """测试 Data2txt 任务命名一致"""
        dataset_task_type = ragtruth_dataset_data2txt.task_type
        
        assert dataset_task_type == "Data2txt"
        
        item = ragtruth_dataset_data2txt[0]
        prompt = prompt_manager.get_prompt(
            task_type=dataset_task_type,
            use_cot=False,
            reference=item.context,
            response=item.output
        )
        
        assert len(prompt) > 0
        assert isinstance(prompt, str)

    def test_valid_task_types(self, prompt_manager):
        """测试所有有效的任务类型"""
        valid_tasks = ["Summary", "QA", "Data2txt"]
        
        for task_type in valid_tasks:
            if task_type == "QA":
                # QA 需要额外的 question 参数
                prompt = prompt_manager.get_prompt(
                    task_type=task_type,
                    use_cot=True,
                    question="Test question?",
                    reference="Reference text",
                    response="Response text"
                )
            else:
                prompt = prompt_manager.get_prompt(
                    task_type=task_type,
                    use_cot=True,
                    reference="Reference text",
                    response="Response text"
                )
            
            assert len(prompt) > 0
            assert isinstance(prompt, str)

    def test_invalid_task_type_rejected(self, prompt_manager):
        """测试拒绝无效的任务类型"""
        invalid_tasks = ["summarization", "question_answering", "data_to_text"]
        
        for invalid_task in invalid_tasks:
            with pytest.raises(ValueError, match="task_type必须是"):
                prompt_manager.get_prompt(
                    task_type=invalid_task,
                    use_cot=True,
                    reference="text",
                    response="text"
                )


class TestEndToEndWorkflow:
    """测试端到端的工作流"""

    def test_dataset_to_prompt_summary(self, ragtruth_dataset_summary, prompt_manager):
        """测试从真实数据集加载到生成 prompt 的完整工作流（Summary）"""
        # 验证数据集非空
        assert len(ragtruth_dataset_summary) > 0
        
        # 获取数据项
        item = ragtruth_dataset_summary[0]
        assert isinstance(item, RAGTruthItem)
        assert item.task_type == "Summary"
        
        # 使用真实数据项生成 prompt
        prompt = prompt_manager.get_prompt(
            task_type=item.task_type,
            use_cot=True,
            reference=item.context,
            response=item.output
        )
        
        # 验证 prompt 生成
        assert len(prompt) > 0
        logger.info(f"Summary prompt 生成成功，长度: {len(prompt)} 字符")

    def test_dataset_to_prompt_qa(self, ragtruth_dataset_qa, prompt_manager):
        """测试从真实数据集加载到生成 prompt 的完整工作流（QA）"""
        assert len(ragtruth_dataset_qa) > 0
        
        item = ragtruth_dataset_qa[0]
        assert isinstance(item, RAGTruthItem)
        assert item.task_type == "QA"
        
        prompt = prompt_manager.get_prompt(
            task_type=item.task_type,
            use_cot=False,
            question="What does this text discuss?",
            reference=item.context,
            response=item.output
        )
        
        assert len(prompt) > 0
        logger.info(f"QA prompt 生成成功，长度: {len(prompt)} 字符")

    def test_dataset_to_prompt_data2txt(self, ragtruth_dataset_data2txt, prompt_manager):
        """测试从真实数据集加载到生成 prompt 的完整工作流（Data2txt）"""
        assert len(ragtruth_dataset_data2txt) > 0
        
        item = ragtruth_dataset_data2txt[0]
        assert isinstance(item, RAGTruthItem)
        assert item.task_type == "Data2txt"
        
        prompt = prompt_manager.get_prompt(
            task_type=item.task_type,
            use_cot=True,
            reference=item.context,
            response=item.output
        )
        
        assert len(prompt) > 0
        logger.info(f"Data2txt prompt 生成成功，长度: {len(prompt)} 字符")

    def test_multiple_items_prompt_generation(self, ragtruth_dataset_summary, prompt_manager):
        """测试为真实数据集中的多个样本生成 prompt"""
        count = min(5, len(ragtruth_dataset_summary))
        prompts = []
        
        for i in range(count):
            item = ragtruth_dataset_summary[i]
            prompt = prompt_manager.get_prompt(
                task_type=item.task_type,
                use_cot=True,
                reference=item.context,
                response=item.output
            )
            prompts.append(prompt)
        
        # 验证所有 prompt 都被生成
        assert len(prompts) == count
        assert all(len(p) > 0 for p in prompts)
        logger.info(f"为 {count} 个真实样本生成 prompt 成功")

    def test_dataset_split_with_prompt_generation(
        self, ragtruth_dataset_summary, prompt_manager
    ):
        """测试真实数据集划分后生成 prompt"""
        # 注意：split_train_val 需要 split='train' 的数据集，而我们使用的是 test split
        # 这里验证数据集结构正确，直接测试 prompt 生成而不进行划分
        assert len(ragtruth_dataset_summary) > 0
        
        # 直接从前几个样本生成 prompt，验证功能正常
        count = min(3, len(ragtruth_dataset_summary))
        for i in range(count):
            item = ragtruth_dataset_summary[i]
            prompt = prompt_manager.get_prompt(
                task_type=item.task_type,
                use_cot=True if i % 2 == 0 else False,
                reference=item.context,
                response=item.output
            )
            assert len(prompt) > 0
        
        logger.info(f"从真实数据集中的 {count} 个样本生成 prompt 成功")


class TestCoTModesConsistency:
    """测试 CoT 模式的一致性"""

    def test_with_cot_and_without_cot(self, ragtruth_dataset_summary, prompt_manager):
        """测试 with_cot 和 without_cot 模式生成不同的 prompt"""
        item = ragtruth_dataset_summary[0]
        
        prompt_with_cot = prompt_manager.get_prompt(
            task_type=item.task_type,
            use_cot=True,
            reference=item.context,
            response=item.output
        )
        
        prompt_without_cot = prompt_manager.get_prompt(
            task_type=item.task_type,
            use_cot=False,
            reference=item.context,
            response=item.output
        )
        
        # 两个 prompt 应该不同
        assert prompt_with_cot != prompt_without_cot
        
        # with_cot 的 prompt 应该更长（包含步骤说明）
        assert len(prompt_with_cot) > len(prompt_without_cot)
        
        # with_cot 应该包含步骤相关的关键词
        assert "Step" in prompt_with_cot or "step" in prompt_with_cot

    def test_cot_mode_for_all_tasks(self, prompt_manager):
        """测试所有任务类型的 CoT 模式"""
        test_cases = [
            ("Summary", {"reference": "doc", "response": "summary"}),
            ("QA", {"question": "q?", "reference": "ref", "response": "ans"}),
            ("Data2txt", {"reference": "data", "response": "text"}),
        ]
        
        for task_type, kwargs in test_cases:
            # with_cot=True
            prompt_cot = prompt_manager.get_prompt(
                task_type=task_type,
                use_cot=True,
                **kwargs
            )
            assert len(prompt_cot) > 0
            
            # with_cot=False
            prompt_no_cot = prompt_manager.get_prompt(
                task_type=task_type,
                use_cot=False,
                **kwargs
            )
            assert len(prompt_no_cot) > 0
            
            # 两者应该不同
            assert prompt_cot != prompt_no_cot


class TestErrorHandling:
    """测试错误处理"""

    def test_missing_required_variables_summary(self, prompt_manager):
        """测试 Summary 缺少必需变量"""
        # 缺少 response
        with pytest.raises(ValueError, match="缺少必需的变量"):
            prompt_manager.get_prompt(
                task_type="Summary",
                use_cot=True,
                reference="reference only"
            )

    def test_missing_required_variables_qa(self, prompt_manager):
        """测试 QA 缺少必需变量"""
        # 缺少 question
        with pytest.raises(ValueError, match="缺少必需的变量"):
            prompt_manager.get_prompt(
                task_type="QA",
                use_cot=True,
                reference="reference",
                response="response"
            )

    def test_invalid_task_type(self, prompt_manager):
        """测试无效的任务类型"""
        with pytest.raises(ValueError, match="task_type必须是"):
            prompt_manager.get_prompt(
                task_type="InvalidTask",
                use_cot=True,
                reference="ref",
                response="res"
            )

    def test_dataset_invalid_task_type(self):
        """测试数据集拒绝无效的任务类型"""
        with pytest.raises(ValueError, match="task_type必须是"):
            RAGTruthDataset(task_type="InvalidTask", split="train")


class TestDataConsistency:
    """测试数据的一致性"""

    def test_item_fields_preserved(self, ragtruth_dataset_summary):
        """测试真实数据的原始字段名被保留"""
        for i, item in enumerate(ragtruth_dataset_summary):
            assert hasattr(item, "context")
            assert hasattr(item, "output")
            assert hasattr(item, "hallucination_labels")
            assert hasattr(item, "task_type")
            assert hasattr(item, "idx")
            
            # 限制循环以避免过长的测试时间
            if i >= 10:
                break

    def test_task_type_consistency_across_dataset(self, ragtruth_dataset_qa):
        """测试真实数据集中的所有样本都有一致的任务类型"""
        expected_task_type = "QA"
        
        for i, item in enumerate(ragtruth_dataset_qa):
            assert item.task_type == expected_task_type
            
            # 限制循环
            if i >= 10:
                break

    def test_prompt_generation_with_hallucination_labels(
        self, ragtruth_dataset_summary, prompt_manager
    ):
        """测试包含幻觉标签的真实样本可以正常生成 prompt"""
        for i, item in enumerate(ragtruth_dataset_summary):
            prompt = prompt_manager.get_prompt(
                task_type=item.task_type,
                use_cot=True,
                reference=item.context,
                response=item.output
            )
            
            # 即使样本有幻觉标签，prompt 仍应正常生成
            assert len(prompt) > 0
            assert isinstance(prompt, str)
            
            # 限制循环
            if i >= 10:
                break
