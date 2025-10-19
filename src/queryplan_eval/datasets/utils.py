"""数据集分割工具模块

提供灵活的数据集分割功能，支持多种分割类型（train/eval/test）、
自定义比例和可复现的随机分割。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Literal, Optional

if TYPE_CHECKING:
    from .query_plan import QueryPlanDataset

logger = logging.getLogger(__name__)

SplitType = Literal["train_eval", "train_test", "eval_test", "train_eval_test"]


@dataclass
class SplitConfig:
    """数据集分割配置
    
    用于指定如何分割数据集，支持四种分割类型，可配置比例和随机种子。
    
    Attributes:
        split_type: 分割类型，支持以下值：
            - "train_eval": 分割为训练集和评估集
            - "train_test": 分割为训练集和测试集
            - "eval_test": 分割为评估集和测试集
            - "train_eval_test": 分割为训练集、评估集和测试集
        train_ratio: 训练集比例 (0.0 到 1.0)。用于 train_eval、train_test、train_eval_test
        eval_ratio: 评估集比例 (0.0 到 1.0)。仅用于 eval_test 和 train_eval_test
        test_ratio: 测试集比例 (0.0 到 1.0)。仅用于 train_test 和 train_eval_test
        random_seed: 随机数种子，用于可复现的分割（默认 42）
    
    Example:
        >>> # 分割为 70% 训练集和 30% 评估集
        >>> config = SplitConfig(
        ...     split_type="train_eval",
        ...     train_ratio=0.7,
        ...     random_seed=42
        ... )
        >>> 
        >>> # 分割为 60% 训练集、20% 评估集和 20% 测试集
        >>> config = SplitConfig(
        ...     split_type="train_eval_test",
        ...     train_ratio=0.6,
        ...     eval_ratio=0.2,
        ...     test_ratio=0.2,
        ...     random_seed=42
        ... )
    """

    split_type: SplitType
    train_ratio: Optional[float] = None
    eval_ratio: Optional[float] = None
    test_ratio: Optional[float] = None
    random_seed: int = 42

    def __post_init__(self) -> None:
        """验证配置的有效性
        
        Raises:
            ValueError: 如果配置不合法
        """
        valid_types = ("train_eval", "train_test", "eval_test", "train_eval_test")
        if self.split_type not in valid_types:
            raise ValueError(
                f"split_type 必须是 {valid_types} 之一，得到: {self.split_type}"
            )

        if self.split_type == "train_eval":
            if self.train_ratio is None:
                raise ValueError("train_eval 分割需要指定 train_ratio")
            if not 0 < self.train_ratio < 1:
                raise ValueError(f"train_ratio 必须在 (0, 1) 之间，得到: {self.train_ratio}")

        elif self.split_type == "train_test":
            if self.train_ratio is None:
                raise ValueError("train_test 分割需要指定 train_ratio")
            if not 0 < self.train_ratio < 1:
                raise ValueError(f"train_ratio 必须在 (0, 1) 之间，得到: {self.train_ratio}")

        elif self.split_type == "eval_test":
            if self.eval_ratio is None:
                raise ValueError("eval_test 分割需要指定 eval_ratio")
            if not 0 < self.eval_ratio < 1:
                raise ValueError(f"eval_ratio 必须在 (0, 1) 之间，得到: {self.eval_ratio}")

        elif self.split_type == "train_eval_test":
            if self.train_ratio is None:
                raise ValueError("train_eval_test 分割需要指定 train_ratio")
            if self.eval_ratio is None:
                raise ValueError("train_eval_test 分割需要指定 eval_ratio")
            if self.test_ratio is None:
                raise ValueError("train_eval_test 分割需要指定 test_ratio")

            total = self.train_ratio + self.eval_ratio + self.test_ratio
            if not (0.99 < total < 1.01):  # 允许浮点误差
                raise ValueError(
                    f"train_eval_test 三个比例之和必须为 1.0，得到: {total}"
                )
            if not (0 < self.train_ratio < 1 and 0 < self.eval_ratio < 1 and 0 < self.test_ratio < 1):
                raise ValueError("所有比例必须在 (0, 1) 之间")

    def __repr__(self) -> str:
        """返回配置的字符串表示"""
        if self.split_type == "train_eval":
            return f"SplitConfig({self.split_type}, train={self.train_ratio}, seed={self.random_seed})"
        elif self.split_type == "train_test":
            return f"SplitConfig({self.split_type}, train={self.train_ratio}, seed={self.random_seed})"
        elif self.split_type == "eval_test":
            return f"SplitConfig({self.split_type}, eval={self.eval_ratio}, seed={self.random_seed})"
        else:  # train_eval_test
            return f"SplitConfig({self.split_type}, train={self.train_ratio}, eval={self.eval_ratio}, test={self.test_ratio}, seed={self.random_seed})"


def split_dataset(
    dataset: QueryPlanDataset, config: SplitConfig
) -> Dict[str, QueryPlanDataset]:
    """根据配置分割数据集
    
    将输入的 QueryPlanDataset 根据 SplitConfig 中的参数分割为多个子数据集。
    支持 train/eval/test 的任意组合分割。
    
    Args:
        dataset: 要分割的 QueryPlanDataset 实例
        config: 分割配置对象
        
    Returns:
        字典，键为分割集的名称（"train"、"eval"、"test"），值为对应的 QueryPlanDataset 实例
        
    Raises:
        ValueError: 如果配置无效或分割失败
        
    Example:
        >>> dataset = QueryPlanDataset("data.xlsx")
        >>> config = SplitConfig(split_type="train_eval", train_ratio=0.7)
        >>> splits = split_dataset(dataset, config)
        >>> train_set = splits["train"]
        >>> eval_set = splits["eval"]
        >>> print(f"训练集大小: {len(train_set)}, 评估集大小: {len(eval_set)}")
    """
    # 导入 QueryPlanDataset（避免循环导入）
    from .query_plan import QueryPlanDataset

    logger.info(f"开始数据集分割: {config}")

    if config.split_type == "train_eval":
        return _split_train_eval(dataset, config)
    elif config.split_type == "train_test":
        return _split_train_test(dataset, config)
    elif config.split_type == "eval_test":
        return _split_eval_test(dataset, config)
    else:  # train_eval_test
        return _split_train_eval_test(dataset, config)


def _split_train_eval(
    dataset: QueryPlanDataset, config: SplitConfig
) -> Dict[str, QueryPlanDataset]:
    """分割为训练集和评估集"""
    from .query_plan import QueryPlanDataset

    train_test_ratio = config.train_ratio
    train_hf, eval_hf = dataset._dataset.train_test_split(
        test_size=1 - train_test_ratio, seed=config.random_seed
    ).values()

    train_dataset = _create_dataset_from_hf(train_hf)
    eval_dataset = _create_dataset_from_hf(eval_hf)

    logger.info(
        f"分割为 train ({len(train_dataset)}) 和 eval ({len(eval_dataset)})"
    )

    return {"train": train_dataset, "eval": eval_dataset}


def _split_train_test(
    dataset: QueryPlanDataset, config: SplitConfig
) -> Dict[str, QueryPlanDataset]:
    """分割为训练集和测试集"""
    from .query_plan import QueryPlanDataset

    train_test_ratio = config.train_ratio
    train_hf, test_hf = dataset._dataset.train_test_split(
        test_size=1 - train_test_ratio, seed=config.random_seed
    ).values()

    train_dataset = _create_dataset_from_hf(train_hf)
    test_dataset = _create_dataset_from_hf(test_hf)

    logger.info(
        f"分割为 train ({len(train_dataset)}) 和 test ({len(test_dataset)})"
    )

    return {"train": train_dataset, "test": test_dataset}


def _split_eval_test(
    dataset: QueryPlanDataset, config: SplitConfig
) -> Dict[str, QueryPlanDataset]:
    """分割为评估集和测试集"""
    from .query_plan import QueryPlanDataset

    eval_test_ratio = config.eval_ratio
    eval_hf, test_hf = dataset._dataset.train_test_split(
        test_size=1 - eval_test_ratio, seed=config.random_seed
    ).values()

    eval_dataset = _create_dataset_from_hf(eval_hf)
    test_dataset = _create_dataset_from_hf(test_hf)

    logger.info(
        f"分割为 eval ({len(eval_dataset)}) 和 test ({len(test_dataset)})"
    )

    return {"eval": eval_dataset, "test": test_dataset}


def _split_train_eval_test(
    dataset: QueryPlanDataset, config: SplitConfig
) -> Dict[str, QueryPlanDataset]:
    """分割为训练集、评估集和测试集
    
    使用两步分割：
    1. 先分割为 train 和 (eval+test)
    2. 再对 (eval+test) 进行分割
    """
    from .query_plan import QueryPlanDataset

    # 第一步：train vs (eval+test)
    train_ratio = config.train_ratio
    train_hf, eval_test_hf = dataset._dataset.train_test_split(
        test_size=1 - train_ratio, seed=config.random_seed
    ).values()

    # 第二步：eval vs test
    # 计算 eval 在 (eval+test) 中的比例
    eval_test_total = config.eval_ratio + config.test_ratio
    eval_in_split_ratio = config.eval_ratio / eval_test_total
    
    eval_hf, test_hf = eval_test_hf.train_test_split(
        test_size=1 - eval_in_split_ratio, seed=config.random_seed
    ).values()

    train_dataset = _create_dataset_from_hf(train_hf)
    eval_dataset = _create_dataset_from_hf(eval_hf)
    test_dataset = _create_dataset_from_hf(test_hf)

    logger.info(
        f"分割为 train ({len(train_dataset)})、eval ({len(eval_dataset)}) "
        f"和 test ({len(test_dataset)})"
    )

    return {
        "train": train_dataset,
        "eval": eval_dataset,
        "test": test_dataset,
    }


def _create_dataset_from_hf(hf_dataset) -> QueryPlanDataset:
    """从 HuggingFace Dataset 创建 QueryPlanDataset 实例
    
    Args:
        hf_dataset: HuggingFace Dataset 实例
        
    Returns:
        QueryPlanDataset 实例，其内部 _dataset 指向传入的 hf_dataset
    """
    from .query_plan import QueryPlanDataset

    # 创建一个空的 QueryPlanDataset 实例，不进行 Excel 加载
    dataset = QueryPlanDataset.__new__(QueryPlanDataset)
    
    # 直接设置内部的 HuggingFace Dataset
    dataset._dataset = hf_dataset
    
    return dataset


def take_samples(
    dataset: QueryPlanDataset, n: int, random: bool = True, seed: int = 42
) -> QueryPlanDataset:
    """从数据集中取 n 个样本
    
    用于从给定的 QueryPlanDataset 中提取部分样本用于快速验证或测试。
    
    Args:
        dataset: 源数据集
        n: 要取的样本数。如果 n 大于数据集大小则返回全部
        random: 是否随机选择样本（默认 True，随机选择 n 个）
        seed: 随机种子，仅在 random=True 时使用（默认 42）
        
    Returns:
        包含 n 个样本的新 QueryPlanDataset 实例
        
    Raises:
        ValueError: 如果 n <= 0
        
    Example:
        >>> dataset = QueryPlanDataset("data.xlsx", n=100)
        >>> 
        >>> # 取前 10 个样本
        >>> subset = take_samples(dataset, n=10)
        >>> 
        >>> # 随机取 10 个样本，使用种子 42
        >>> subset = take_samples(dataset, n=10, random=True, seed=42)
        >>> 
        >>> # 在循环中使用
        >>> for item in subset:
        ...     print(item.query)
    """
    if n <= 0:
        raise ValueError(f"n 必须大于 0，得到: {n}")
    
    # 确保不超过数据集大小
    n = min(n, len(dataset))
    
    if random:
        # 使用随机选择
        import random as py_random
        py_random.seed(seed)
        indices = sorted(py_random.sample(range(len(dataset)), n))
    else:
        # 顺序选择前 n 个
        indices = list(range(n))
    
    # 使用 HuggingFace Dataset 的 select 方法获取子集
    hf_subset = dataset._dataset.select(indices)
    
    # 从子集创建新的 QueryPlanDataset
    subset_dataset = _create_dataset_from_hf(hf_subset)
    
    logger.info(f"从数据集中取出 {n} 个样本（total={len(dataset)}, random={random}）")
    
    return subset_dataset
