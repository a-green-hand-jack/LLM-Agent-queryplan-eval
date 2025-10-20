"""
RAGTruth 任务的指标计算模块

提供幻觉检测的 span-level 指标计算和聚合功能
"""

import json
import logging
from typing import Dict, List, Any

from .span_utils import calculate_span_f1, Span

logger = logging.getLogger(__name__)


def compute_hallucination_metrics(
    predicted_spans: List[Span],
    ground_truth_spans: List[Span],
) -> Dict[str, float]:
    """计算单个样本的幻觉检测指标
    
    Args:
        predicted_spans: 模型预测的幻觉 spans（字符级位置）
        ground_truth_spans: 真实的幻觉 spans（字符级位置）
    
    Returns:
        指标字典，包含 precision, recall, f1
        
    Examples:
        >>> metrics = compute_hallucination_metrics([(0, 10)], [(0, 10)])
        >>> print(metrics)
        {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    """
    precision, recall, f1 = calculate_span_f1(predicted_spans, ground_truth_spans)
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def aggregate_metrics_by_task(
    results: List[Dict],
    metric_keys: List[str] = ["precision", "recall", "f1"],
) -> Dict[str, Dict[str, float]]:
    """按任务类型聚合指标
    
    计算每个任务类型的平均指标。
    
    Args:
        results: 所有样本的结果列表，每个结果应包含 task_type 和指标字段
        metric_keys: 要聚合的指标键名列表
    
    Returns:
        按任务类型组织的指标字典
        {
            "Summary": {"precision": 0.9, "recall": 0.85, "f1": 0.87},
            "QA": {"precision": 0.88, "recall": 0.90, "f1": 0.89},
            ...
        }
    """
    # 按 task_type 分组
    tasks_metrics: Dict[str, List[Dict[str, float]]] = {}
    
    for result in results:
        task_type = result.get("task_type", "unknown")
        
        if task_type not in tasks_metrics:
            tasks_metrics[task_type] = []
        
        # 提取指标
        sample_metrics = {}
        for key in metric_keys:
            if key in result:
                sample_metrics[key] = result[key]
        
        if sample_metrics:
            tasks_metrics[task_type].append(sample_metrics)
    
    # 计算每个任务的平均指标
    aggregated: Dict[str, Dict[str, float]] = {}
    
    for task_type, metrics_list in tasks_metrics.items():
        if not metrics_list:
            continue
        
        aggregated[task_type] = {}
        
        for key in metric_keys:
            values = [m[key] for m in metrics_list if key in m]
            if values:
                aggregated[task_type][key] = sum(values) / len(values)
            else:
                aggregated[task_type][key] = 0.0
        
        # 添加样本数
        aggregated[task_type]["count"] = len(metrics_list)
    
    logger.info(f"按任务类型聚合指标完成: {list(aggregated.keys())}")
    
    return aggregated


def compute_overall_metrics(
    results: List[Dict],
    metric_keys: List[str] = ["precision", "recall", "f1"],
) -> Dict[str, float]:
    """计算整体指标
    
    对所有样本的指标进行平均计算。
    
    Args:
        results: 所有样本的结果列表
        metric_keys: 要聚合的指标键名列表
    
    Returns:
        整体指标字典，包含 count（样本数）
    """
    if not results:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "count": 0,
        }
    
    overall: Dict[str, float] = {}
    
    for key in metric_keys:
        values = [r[key] for r in results if key in r]
        if values:
            overall[key] = sum(values) / len(values)
        else:
            overall[key] = 0.0
    
    overall["count"] = len(results)
    
    logger.info(f"整体指标计算完成: F1={overall.get('f1', 0.0):.4f}")
    
    return overall


def compute_parse_success_rate(results: List[Dict]) -> float:
    """计算解析成功率
    
    Args:
        results: 所有样本的结果列表，每个结果应包含 "ok" 字段
    
    Returns:
        成功率（0-1 之间的浮点数）
    """
    if not results:
        return 0.0
    
    success_count = sum(1 for r in results if r.get("ok", False))
    
    return success_count / len(results)


def generate_metric_report(
    results: List[Dict],
    output_format: str = "dict",
) -> Any:
    """生成综合指标报告
    
    Args:
        results: 所有样本的结果列表
        output_format: 输出格式 ("dict" 或 "json")
    
    Returns:
        包含各个层级指标的报告
    """
    # 计算各个层级的指标
    overall = compute_overall_metrics(results)
    by_task = aggregate_metrics_by_task(results)
    parse_success = compute_parse_success_rate(results)
    
    report = {
        "overall": overall,
        "by_task": by_task,
        "parse_success_rate": parse_success,
        "total_samples": len(results),
    }
    
    if output_format == "json":
        return json.dumps(report, indent=2, ensure_ascii=False)
    else:
        return report
