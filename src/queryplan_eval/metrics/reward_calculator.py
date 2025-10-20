"""
Reward计算器

根据预测的hallucination spans和真实spans计算reward
"""

import logging
from typing import List, Tuple

from .span_utils import calculate_span_f1

logger = logging.getLogger(__name__)

# Span类型定义
Span = Tuple[int, int]


class RewardCalculator:
    """
    Reward计算器
    
    实现论文中定义的reward函数:
    - 当预测和真实都为空时,reward = 1.0
    - 否则 reward = span-F1分数
    
    Examples:
        >>> calc = RewardCalculator()
        >>> reward = calc.calculate_reward(
        ...     predicted_spans=[(0, 10)],
        ...     ground_truth_spans=[(0, 10)]
        ... )
        >>> print(reward)  # 1.0 (完美匹配)
    """
    
    def __init__(self):
        pass
    
    def calculate_reward(
        self,
        predicted_spans: List[Span],
        ground_truth_spans: List[Span],
    ) -> float:
        """
        计算单个样本的reward
        
        Args:
            predicted_spans: 模型预测的hallucination spans
            ground_truth_spans: 真实的hallucination spans
        
        Returns:
            reward分数 [0.0, 1.0]
        """
        # 特殊情况:都为空
        if not predicted_spans and not ground_truth_spans:
            return 1.0
        
        # 计算span-F1
        precision, recall, f1 = calculate_span_f1(predicted_spans, ground_truth_spans)
        
        return f1
    
    def calculate_batch_rewards(
        self,
        batch_predicted_spans: List[List[Span]],
        batch_ground_truth_spans: List[List[Span]],
    ) -> List[float]:
        """
        批量计算reward
        
        Args:
            batch_predicted_spans: 一批预测spans
            batch_ground_truth_spans: 一批真实spans
        
        Returns:
            reward列表
        """
        if len(batch_predicted_spans) != len(batch_ground_truth_spans):
            raise ValueError(
                f"预测和真实的batch大小不匹配: "
                f"{len(batch_predicted_spans)} vs {len(batch_ground_truth_spans)}"
            )
        
        rewards = []
        for pred_spans, gt_spans in zip(batch_predicted_spans, batch_ground_truth_spans):
            reward = self.calculate_reward(pred_spans, gt_spans)
            rewards.append(reward)
        
        logger.debug(f"批次reward计算完成: mean={sum(rewards)/len(rewards):.4f}")
        
        return rewards
    
    def calculate_group_rewards(
        self,
        group_predicted_spans: List[List[Span]],
        ground_truth_spans: List[Span],
    ) -> List[float]:
        """
        计算一组样本的reward(用于GRPO的group sampling)
        
        在GRPO中,对同一个输入生成多个样本,这些样本共享同一个ground truth
        
        Args:
            group_predicted_spans: 一组预测spans(同一输入的多个生成)
            ground_truth_spans: 共享的真实spans
        
        Returns:
            reward列表
        """
        rewards = []
        for pred_spans in group_predicted_spans:
            reward = self.calculate_reward(pred_spans, ground_truth_spans)
            rewards.append(reward)
        
        logger.debug(
            f"Group reward计算完成: size={len(rewards)}, "
            f"mean={sum(rewards)/len(rewards):.4f}, "
            f"max={max(rewards):.4f}, min={min(rewards):.4f}"
        )
        
        return rewards

