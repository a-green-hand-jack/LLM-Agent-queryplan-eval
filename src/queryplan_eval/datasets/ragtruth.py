"""
RAGTruth数据集的HuggingFace风格实现

设计理念：
1. 使用原始字段名，不进行映射
2. 支持HuggingFace风格的高级操作（map、filter等）
3. 数据加载与访问分离
4. 类型安全的数据访问
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)


@dataclass
class RAGTruthItem:
    """RAGTruth数据项
    
    表示数据集中的单个样本，使用原始字段名。
    支持通过属性访问数据，便于类型检查和IDE自动补全。
    
    Attributes:
        idx: 数据的索引
        task_type: 任务类型 (Summary, QA, Data2txt)
        context: 上下文/参考文本
        output: 模型生成的输出
        hallucination_labels: 幻觉标注（JSON字符串）
        query: 问题/任务描述（可选）
    """
    idx: int
    task_type: str
    context: str
    output: str
    hallucination_labels: str
    query: Optional[str] = None


class RAGTruthDataset:
    """RAGTruth数据集的HuggingFace风格实现
    
    从HuggingFace Hub加载RAGTruth数据集，提供类似HuggingFace Datasets
    的高级操作接口（如map、filter等），同时支持类型安全的数据访问。
    
    Example:
        >>> # 加载数据集
        >>> train_dataset = RAGTruthDataset(task_type="Summary", split="train")
        >>> test_dataset = RAGTruthDataset(task_type="QA", split="test")
        >>> 
        >>> # 访问样本
        >>> len(train_dataset)
        >>> item = train_dataset[0]
        >>> print(item.context, item.output)
        >>> 
        >>> # 迭代
        >>> for item in train_dataset:
        ...     print(item.idx, item.task_type)
    """
    
    _dataset: Dataset
    task_type: str
    split: str
    
    def __init__(
        self,
        task_type: str = "Summary",
        split: str = "train",
        cache_dir: Optional[Path] = None,
    ) -> None:
        """初始化RAGTruth数据集
        
        Args:
            task_type: 任务类型 (Summary, QA, Data2txt)
            split: 数据集分割 (train 或 test)
            cache_dir: HuggingFace数据集缓存目录
            
        Raises:
            ValueError: 无效的task_type或split
            RuntimeError: 数据加载失败
        """
        valid_tasks = ["Summary", "QA", "Data2txt"]
        if task_type not in valid_tasks:
            raise ValueError(
                f"task_type必须是{valid_tasks}之一，得到: {task_type}"
            )
        
        valid_splits = ["train", "test"]
        if split not in valid_splits:
            raise ValueError(
                f"split必须是{valid_splits}之一，得到: {split}"
            )
        
        self.task_type = task_type
        self.split = split
        
        logger.info(f"加载RAGTruth数据集: task_type={task_type}, split={split}")
        
        try:
            # 加载HuggingFace数据集
            dataset = load_dataset(
                "wandb/RAGTruth-processed",
                split=split,
                cache_dir=str(cache_dir) if cache_dir else None,
                trust_remote_code=False,
            )
            
            # 过滤指定任务类型的数据
            filtered_dataset = dataset.filter(  # type: ignore
                lambda item: item.get("task_type") == task_type
            )
            self._dataset = filtered_dataset  # type: ignore
            
            logger.info(
                f"已加载 {len(self._dataset)} 个任务类型为 '{task_type}' 的样本"
            )
            
        except Exception as e:
            logger.error(f"加载RAGTruth数据集失败: {e}")
            raise RuntimeError(f"无法加载RAGTruth数据集: {e}")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self._dataset)

    def __getitem__(self, idx: int) -> RAGTruthItem:
        """获取指定索引的样本
        
        Args:
            idx: 样本索引
            
        Returns:
            RAGTruthItem对象
            
        Raises:
            IndexError: 如果索引超出范围
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"索引 {idx} 超出范围 [0, {len(self) - 1}]")
        
        row = self._dataset[idx]
        return RAGTruthItem(
            idx=idx,
            task_type=str(row["task_type"]),
            context=str(row["context"]),
            output=str(row["output"]),
            hallucination_labels=str(row["hallucination_labels"]),
            query=str(row.get("query")) if row.get("query") else None,
        )
    
    def __iter__(self):
        """迭代数据集中的所有项
        
        Yields:
            RAGTruthItem: 数据集中的每一个数据项
        """
        for i in range(len(self)):
            yield self[i]
    
    def __getattr__(self, name: str) -> Any:
        """代理不存在的属性到内部的HuggingFace Dataset
        
        这允许用户直接调用HuggingFace Dataset的方法，如map、filter等，
        同时保留自定义的类型安全的__getitem__接口。
        
        Args:
            name: 属性名
            
        Returns:
            HuggingFace Dataset的属性或方法
            
        Raises:
            AttributeError: 如果属性不存在
        """
        return getattr(self._dataset, name)


# ============================================================================
# 第二层：数据划分函数
# ============================================================================


def split_train_val(
    dataset: RAGTruthDataset,
    val_ratio: float = 0.2,
    random_seed: int = 42,
    cache_dir: Optional[Path] = None,
    force_resplit: bool = False,
) -> tuple[RAGTruthDataset, RAGTruthDataset]:
    """
    将RAGTruthDataset的train部分划分为训练集和验证集
    
    设计要点：
    1. 使用固定随机种子保证可重复性
    2. 将划分索引缓存到文件，确保跨运行的一致性
    3. 返回两个RAGTruthDataset实例
    
    Args:
        dataset: RAGTruthDataset实例
        val_ratio: 验证集比例，默认0.2
        random_seed: 随机种子，默认42
        cache_dir: 缓存目录，如果为None则不缓存
        force_resplit: 是否强制重新划分（忽略缓存）
        
    Returns:
        (train_split, val_split): 划分后的两个RAGTruthDataset实例
        
    Raises:
        ValueError: 如果数据集split不是train
    """
    if dataset.split != "train":
        raise ValueError(
            f"只能对split='train'的数据集进行划分，当前split='{dataset.split}'"
        )
    
    logger.info(
        f"开始划分数据集 "
        f"(task_type={dataset.task_type}, val_ratio={val_ratio})"
    )
    
    # 尝试从缓存加载
    if cache_dir is not None and not force_resplit:
        cached_indices = _load_split_indices_from_cache(
            cache_dir,
            dataset.task_type,
            len(dataset),
            val_ratio,
            random_seed
        )
        if cached_indices is not None:
            train_indices, val_indices = cached_indices
            logger.info("从缓存加载划分索引")
            train_split = _create_dataset_from_indices(
                dataset._dataset, train_indices, dataset.task_type, "train"
            )
            val_split = _create_dataset_from_indices(
                dataset._dataset, val_indices, dataset.task_type, "val"
            )
            logger.info(
                f"划分结果: train={len(train_split)}, val={len(val_split)}"
            )
            return train_split, val_split
    
    # 执行新的划分
    logger.info("执行新的数据划分...")
    
    # 使用HuggingFace Dataset的train_test_split方法
    splits = dataset._dataset.train_test_split(
        test_size=val_ratio,
        seed=random_seed,
        shuffle=True,
    )
    
    train_hf_dataset = splits["train"]
    val_hf_dataset = splits["test"]
    
    # 保存索引到缓存
    if cache_dir is not None:
        train_indices = list(range(len(train_hf_dataset)))
        val_indices = list(range(len(train_hf_dataset), len(train_hf_dataset) + len(val_hf_dataset)))
        _save_split_indices_to_cache(
            cache_dir,
            dataset.task_type,
            train_indices,
            val_indices,
            len(dataset),
            val_ratio,
            random_seed
        )
    
    # 创建RAGTruthDataset实例
    train_split = _create_rag_dataset_from_hf(
        train_hf_dataset, dataset.task_type, "train"
    )
    val_split = _create_rag_dataset_from_hf(
        val_hf_dataset, dataset.task_type, "val"
    )
    
    logger.info(f"划分完成: train={len(train_split)}, val={len(val_split)}")
    return train_split, val_split


def _load_split_indices_from_cache(
    cache_dir: Path,
    task_type: str,
    data_size: int,
    val_ratio: float,
    random_seed: int,
) -> Optional[tuple[List[int], List[int]]]:
    """
    从缓存加载划分索引
    
    Args:
        cache_dir: 缓存目录
        task_type: 任务类型
        data_size: 数据集大小
        val_ratio: 验证集比例
        random_seed: 随机种子
    
    Returns:
        (train_indices, val_indices) 或 None（如果缓存不存在或失效）
    """
    cache_file = cache_dir / f"split_indices_{task_type}.json"
    
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, "r") as f:
            cache_data = json.load(f)
        
        # 验证缓存的参数是否匹配
        if (cache_data.get("data_size") != data_size or
            abs(cache_data.get("val_ratio", 0) - val_ratio) > 1e-6 or
            cache_data.get("random_seed") != random_seed):
            logger.info("缓存参数不匹配，将重新划分")
            return None
        
        train_indices = cache_data.get("train_indices", [])
        val_indices = cache_data.get("val_indices", [])
        
        if not train_indices or not val_indices:
            logger.warning("缓存的索引无效，将重新划分")
            return None
        
        return train_indices, val_indices
        
    except Exception as e:
        logger.warning(f"加载缓存失败: {e}")
        return None


def _save_split_indices_to_cache(
    cache_dir: Path,
    task_type: str,
    train_indices: List[int],
    val_indices: List[int],
    data_size: int,
    val_ratio: float,
    random_seed: int,
) -> None:
    """
    将划分索引保存到缓存
    
    Args:
        cache_dir: 缓存目录
        task_type: 任务类型
        train_indices: 训练集索引
        val_indices: 验证集索引
        data_size: 数据集大小
        val_ratio: 验证集比例
        random_seed: 随机种子
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"split_indices_{task_type}.json"
    
    cache_data = {
        "task_type": task_type,
        "data_size": data_size,
        "val_ratio": val_ratio,
        "random_seed": random_seed,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
    }
    
    try:
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
        logger.info(f"划分索引已缓存到: {cache_file}")
    except Exception as e:
        logger.warning(f"保存缓存失败: {e}")


def _create_rag_dataset_from_hf(hf_dataset, task_type: str, split: str) -> RAGTruthDataset:
    """
    从HuggingFace Dataset创建RAGTruthDataset实例
    
    Args:
        hf_dataset: HuggingFace Dataset实例
        task_type: 任务类型
        split: 数据集分割标识（用于日志，不影响实际功能）
        
    Returns:
        RAGTruthDataset实例
    """
    # 创建一个空的RAGTruthDataset实例
    dataset = RAGTruthDataset.__new__(RAGTruthDataset)
    
    # 直接设置内部属性
    dataset._dataset = hf_dataset
    dataset.task_type = task_type
    dataset.split = split
    
    return dataset


def _create_dataset_from_indices(hf_dataset, indices: List[int], task_type: str, split: str) -> RAGTruthDataset:
    """
    从HuggingFace Dataset和索引列表创建RAGTruthDataset实例
    
    Args:
        hf_dataset: 原始HuggingFace Dataset
        indices: 要选择的索引列表
        task_type: 任务类型
        split: 数据集分割标识
        
    Returns:
        包含选定样本的RAGTruthDataset实例
    """
    selected_hf_dataset = hf_dataset.select(indices)
    return _create_rag_dataset_from_hf(selected_hf_dataset, task_type, split)


# ============================================================================
# 辅助函数
# ============================================================================


def compute_sample_weights(dataset: RAGTruthDataset) -> List[float]:
    """
    计算样本权重，用于处理类别不平衡
    
    策略：为包含幻觉的样本分配更高的权重
    
    Args:
        dataset: RAGTruthDataset实例
        
    Returns:
        样本权重列表，长度与dataset大小相同
    """
    # 统计包含幻觉的样本
    hallucination_count = 0
    for item in dataset:
        labels = item.hallucination_labels
        if labels and labels != "[]":
            try:
                parsed = json.loads(labels)
                if parsed:
                    hallucination_count += 1
            except json.JSONDecodeError:
                continue
    
    no_hallucination_count = len(dataset) - hallucination_count
    
    if hallucination_count == 0 or no_hallucination_count == 0:
        # 如果只有一个类别，返回均等权重
        return [1.0] * len(dataset)
    
    # 计算类别权重（少数类权重更高）
    total = len(dataset)
    hallucination_weight = total / (2 * hallucination_count)
    no_hallucination_weight = total / (2 * no_hallucination_count)
    
    # 为每个样本分配权重
    weights = []
    for item in dataset:
        labels = item.hallucination_labels
        has_hallucination = False
        
        if labels and labels != "[]":
            try:
                parsed = json.loads(labels)
                if parsed:
                    has_hallucination = True
            except json.JSONDecodeError:
                pass
        
        if has_hallucination:
            weights.append(hallucination_weight)
        else:
            weights.append(no_hallucination_weight)
    
    logger.info("类别权重计算:")
    logger.info(f"  有幻觉样本: {hallucination_count} 个, 权重: {hallucination_weight:.4f}")
    logger.info(
        f"  无幻觉样本: {no_hallucination_count} 个, 权重: {no_hallucination_weight:.4f}"
    )
    
    return weights


def get_dataset_statistics(dataset: RAGTruthDataset) -> Dict[str, Any]:
    """
    获取数据集统计信息
    
    Args:
        dataset: RAGTruthDataset实例
        
    Returns:
        统计信息字典
    """
    total = len(dataset)
    hallucination_count = 0
    total_spans = 0
    total_context_len = 0
    total_output_len = 0
    
    for item in dataset:
        # 统计有幻觉的样本
        labels = item.hallucination_labels
        if labels and labels != "[]":
            try:
                parsed = json.loads(labels)
                if parsed:
                    hallucination_count += 1
                    total_spans += len(parsed)
            except json.JSONDecodeError:
                pass
        
        # 统计文本长度
        total_context_len += len(item.context)
        total_output_len += len(item.output)
    
    return {
        "total_samples": total,
        "hallucination_samples": hallucination_count,
        "hallucination_ratio": hallucination_count / total if total > 0 else 0,
        "avg_hallucination_spans": total_spans / total if total > 0 else 0,
        "avg_context_length": total_context_len / total if total > 0 else 0,
        "avg_output_length": total_output_len / total if total > 0 else 0,
    }
