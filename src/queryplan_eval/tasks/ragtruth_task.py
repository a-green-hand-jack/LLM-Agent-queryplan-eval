"""RAGTruth 幻觉检测任务"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..core.base_task import BaseTask
from ..core.prompt_manager import RAGPromptManager
from ..datasets import RAGTruthDataset, RAGTruthItem
from ..metrics.ragtruth_metrics import (
    compute_hallucination_metrics,
    aggregate_metrics_by_task,
    compute_overall_metrics,
)
from ..metrics.span_utils import parse_spans_from_text, Span
from ..schemas import HallucinationResult

logger = logging.getLogger(__name__)


class RAGTruthItemList:
    """RAGTruthItem 列表的 Dataset 风格包装
    
    模拟 HuggingFace Dataset 接口，但迭代返回 RAGTruthItem 对象而不是字典。
    与 QueryPlanDataset 保持一致的使用方式。
    """
    
    def __init__(self, items: List[RAGTruthItem]):
        self.items = items
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __iter__(self):
        return iter(self.items)
    
    def __getitem__(self, idx: int) -> RAGTruthItem:
        return self.items[idx]


class RAGTruthTask(BaseTask):
    """RAGTruth 幻觉检测任务
    
    用于评估 LLM 的幻觉检测能力，支持多任务类型（Summary/QA/Data2txt）
    的灵活加载和独立评估。
    
    特点：
    - 支持单任务或多任务组合评估
    - 支持从 train 或 test 分割加载数据
    - 支持固定数量采样（sample_n）或比例采样（sample_ratio）
    - 基于 span-level 的精细评估指标（F1, Precision, Recall）
    - 生成分任务报告和综合报告
    
    Examples:
        >>> # 只评估 Summary 任务
        >>> task = RAGTruthTask(
        ...     task_types=["Summary"],
        ...     split="test",
        ...     llm=llm,
        ...     output_dir="outputs/ragtruth_summary"
        ... )
        >>> metrics = task.run_evaluation()
        
        >>> # 评估多个任务，采样 100 个样本
        >>> task = RAGTruthTask(
        ...     task_types=["Summary", "QA"],
        ...     split="train",
        ...     sample_n=100,
        ...     use_cot=True,
        ...     llm=llm,
        ...     output_dir="outputs/ragtruth_multi"
        ... )
        >>> metrics = task.run_evaluation()
    """
    
    def __init__(
        self,
        task_types: List[str],
        split: str = "test",
        sample_n: Optional[int] = None,
        sample_ratio: Optional[float] = None,
        use_cot: bool = False,
        cache_dir: Optional[Path] = None,
        llm: Any = None,
        output_dir: str = "outputs/ragtruth",
        **kwargs
    ):
        """初始化 RAGTruth 任务
        
        Args:
            task_types: 任务类型列表，如 ["Summary", "QA", "Data2txt"]
                       必须是有效的任务类型组合
            split: 数据分割 "train" 或 "test"
            sample_n: 固定采样数量（与 sample_ratio 互斥）
            sample_ratio: 采样比例 (0-1)（与 sample_n 互斥）
            use_cot: 是否使用 Chain-of-Thought 推理
            cache_dir: 数据集缓存目录
            llm: LLM 实例
            output_dir: 输出目录
            **kwargs: 其他参数
        """
        # 验证 task_types
        valid_tasks = ["Summary", "QA", "Data2txt"]
        for task_type in task_types:
            if task_type not in valid_tasks:
                raise ValueError(
                    f"task_type 必须是 {valid_tasks} 之一，得到: {task_type}"
                )
        
        # 验证采样参数
        if sample_n is not None and sample_ratio is not None:
            raise ValueError(
                "sample_n 和 sample_ratio 不能同时指定，请只设置其中一个"
            )
        
        self.task_types = task_types
        self.split = split
        self.sample_n = sample_n
        self.sample_ratio = sample_ratio
        self.use_cot = use_cot
        self.cache_dir = cache_dir
        
        # 初始化 RAGPromptManager
        self.prompt_manager = RAGPromptManager()
        
        # 占位符，BaseTask 会调用 load_dataset
        # 这里先不调用 super().__init__，因为需要先设置一些参数
        self.llm = llm
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"初始化 RAGTruthTask: task_types={task_types}, split={split}, "
            f"use_cot={use_cot}, sample_n={sample_n}, sample_ratio={sample_ratio}"
        )
        
        # 加载数据集
        self.dataset = self.load_dataset(path="")
        logger.info(
            f"任务初始化完成，数据集大小: {len(self.dataset)}, 任务类型: {task_types}"
        )
    
    def load_dataset(self, path: str = "") -> Any:
        """加载灵活的 RAGTruth 数据集
        
        根据配置的 task_types、split 和采样参数加载数据。
        与 QueryPlanTask 保持一致，返回可以迭代出对象的数据集。
        
        Args:
            path: 未使用（兼容 BaseTask 接口）
        
        Returns:
            RAGTruthItemList，可以迭代出 RAGTruthItem 对象
        """
        logger.info(
            f"加载数据集: task_types={self.task_types}, split={self.split}"
        )
        
        # 收集所有 RAGTruthItem 对象
        all_items = []
        
        # 为每个任务类型加载数据
        for task_type in self.task_types:
            logger.info(f"  加载任务类型: {task_type}")
            
            rag_dataset = RAGTruthDataset(
                task_type=task_type,
                split=self.split,
                cache_dir=self.cache_dir,
            )
            
            # 直接收集 RAGTruthItem 对象，不转换为字典
            for item in rag_dataset:
                all_items.append(item)
        
        logger.info(f"合并后的数据集大小: {len(all_items)}")
        
        # 应用采样
        original_size = len(all_items)
        if self.sample_n is not None:
            if self.sample_n < len(all_items):
                all_items = all_items[:self.sample_n]
                logger.info(f"按数量采样: {len(all_items)}/{original_size}")
            else:
                logger.info(f"样本数量不足采样: {len(all_items)}")
        elif self.sample_ratio is not None:
            n = int(len(all_items) * self.sample_ratio)
            all_items = all_items[:n]
            logger.info(f"按比例采样: {n}/{original_size} ({self.sample_ratio:.1%})")
        
        logger.info(f"最终数据集大小: {len(all_items)}")
        
        # 返回包装类，表现得像 Dataset 但迭代返回 RAGTruthItem
        return RAGTruthItemList(all_items)
    
    def build_chat(self, item: Any) -> list[dict[str, str]]:
        """构建 chat 消息
        
        Args:
            item: RAGTruthItem 对象
        
        Returns:
            chat 消息列表
        """
        task_type = item.task_type
        
        # 准备 prompt 参数
        prompt_kwargs = {
            "reference": item.context,
            "response": item.output,
        }
        
        # QA 任务需要额外的 question 字段
        if task_type == "QA" and item.query:
            prompt_kwargs["question"] = item.query
        
        # 获取 prompt
        system_prompt = self.prompt_manager.get_prompt(
            task_type=task_type,
            use_cot=self.use_cot,
            **prompt_kwargs
        )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps({
                "task_type": task_type,
                "reference": item.context,
                "response": item.output,
            }, ensure_ascii=False)}
        ]
    
    def get_output_schema(self):
        """返回输出类型"""
        return HallucinationResult
    
    def process_single_result(
        self,
        item: RAGTruthItem,
        parsed: Optional[HallucinationResult],
        raw: Optional[str],
        latency: float
    ) -> Dict:
        """处理单个样本的结果
        
        Args:
            item: RAGTruthItem 对象
            parsed: 解析后的 HallucinationResult
            raw: 原始输出
            latency: 耗时
        
        Returns:
            处理后的结果记录
        """
        task_type = item.task_type
        response = item.output
        
        ok = parsed is not None
        predicted_spans: List[Span] = []
        ground_truth_spans: List[Span] = []
        metrics: Dict[str, float] = {}
        err = None
        
        # 处理成功的预测
        if ok and parsed is not None:
            try:
                # 解析预测的幻觉片段
                hallucination_list = parsed.hallucination_list or []
                predicted_spans = parse_spans_from_text(hallucination_list, response)
                
            except Exception as e:
                logger.warning(f"解析预测 spans 失败: {e}")
                ok = False
                err = str(e)
        
        # 解析真实的幻觉标注
        try:
            hallucination_labels_gt = item.hallucination_labels
            if hallucination_labels_gt  and hallucination_labels_gt != "[]":
                parsed_labels = json.loads(hallucination_labels_gt)
                ground_truth_spans = parse_spans_from_text(parsed_labels, response)
        except Exception as e:
            logger.warning(
                f"解析真实 spans 失败 (idx={item.idx}): {e}"
            )
        
        # 计算指标
        if ok:
            metrics = compute_hallucination_metrics(
                predicted_spans,
                ground_truth_spans
            )
        
        record = {
            "idx": item.idx,
            "task_type": task_type,
            "context_length": len(item.context),
            "output_length": len(response),
            "hallucination_labels": item.hallucination_labels,
            "predicted_hallucinations": json.dumps(
                parsed.hallucination_list if parsed else [], ensure_ascii=False
            ),
            "raw_response": raw,
            "ok": ok,
            "latency_sec": latency,
            "error": err,
            "predicted_spans": json.dumps(
                [(s[0], s[1]) for s in predicted_spans],
                ensure_ascii=False
            ),
            "ground_truth_spans": json.dumps(
                [(s[0], s[1]) for s in ground_truth_spans],
                ensure_ascii=False
            ),
        }
        
        # 添加指标
        record.update(metrics)
        
        return record
    
    def compute_metrics(self, results: List[Dict]) -> Dict:
        """计算评估指标
        
        Args:
            results: 所有结果记录列表
        
        Returns:
            指标字典
        """
        df = pd.DataFrame(results)
        
        total = len(df)
        ok = df["ok"].sum()
        parse_success_rate = ok / total if total > 0 else 0.0
        
        # 计算整体指标
        overall_metrics = compute_overall_metrics(results)
        
        # 计算分任务指标
        by_task_metrics = aggregate_metrics_by_task(results)
        
        metrics = {
            "total": int(total),
            "ok": int(ok),
            "parse_success_rate": float(parse_success_rate),
            "overall": overall_metrics,
            "by_task": by_task_metrics,
        }
        
        logger.info(f"指标计算完成: {metrics}")
        return metrics
    
    def save_results(self, results: List[Dict], metrics: Dict) -> None:
        """保存结果和指标
        
        Args:
            results: 所有结果记录列表
            metrics: 指标字典
        """
        df = pd.DataFrame(results)
        
        # 保存详细结果到 CSV
        csv_path = self.output_dir / "results.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"结果已保存: {csv_path}")
        
        # 保存指标到 JSON
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"指标已保存: {metrics_path}")
        
        # 生成分任务报告
        self._save_task_reports(df, metrics)
        
        # 生成综合摘要
        self._save_summary(metrics)
    
    def _save_task_reports(self, df: pd.DataFrame, metrics: Dict) -> None:
        """保存分任务报告
        
        Args:
            df: 结果数据框
            metrics: 指标字典
        """
        by_task = metrics.get("by_task", {})
        
        for task_type in self.task_types:
            task_df = df[df["task_type"] == task_type]
            
            if len(task_df) == 0:
                continue
            
            # 保存任务特定的结果
            task_csv_path = self.output_dir / f"results_{task_type}.csv"
            task_df.to_csv(task_csv_path, index=False, encoding='utf-8')
            logger.info(f"任务 {task_type} 结果已保存: {task_csv_path}")
            
            # 生成任务报告文本
            task_report_path = self.output_dir / f"report_{task_type}.txt"
            with open(task_report_path, "w", encoding="utf-8") as f:
                f.write(f"=== {task_type} 任务评估报告 ===\n\n")
                f.write(f"样本数: {len(task_df)}\n")
                f.write(f"成功率: {task_df['ok'].sum() / len(task_df):.1%}\n\n")
                
                if task_type in by_task:
                    task_metrics = by_task[task_type]
                    f.write("评估指标:\n")
                    f.write(f"  Precision: {task_metrics.get('precision', 0):.4f}\n")
                    f.write(f"  Recall:    {task_metrics.get('recall', 0):.4f}\n")
                    f.write(f"  F1:        {task_metrics.get('f1', 0):.4f}\n")
            
            logger.info(f"任务 {task_type} 报告已保存: {task_report_path}")
    
    def _save_summary(self, metrics: Dict) -> None:
        """保存综合摘要
        
        Args:
            metrics: 指标字典
        """
        summary_path = self.output_dir / "summary.txt"
        
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=== RAGTruth 幻觉检测评估总结 ===\n\n")
            
            f.write("配置信息:\n")
            f.write(f"  任务类型: {', '.join(self.task_types)}\n")
            f.write(f"  数据分割: {self.split}\n")
            f.write(f"  是否使用 CoT: {self.use_cot}\n")
            f.write("\n")
            
            f.write("总体结果:\n")
            f.write(f"  总样本数: {metrics['total']}\n")
            f.write(f"  成功样本: {metrics['ok']}\n")
            f.write(f"  成功率: {metrics['parse_success_rate']:.1%}\n")
            f.write("\n")
            
            overall = metrics.get("overall", {})
            f.write("总体指标:\n")
            f.write(f"  Precision: {overall.get('precision', 0):.4f}\n")
            f.write(f"  Recall:    {overall.get('recall', 0):.4f}\n")
            f.write(f"  F1:        {overall.get('f1', 0):.4f}\n")
            f.write("\n")
            
            by_task = metrics.get("by_task", {})
            if by_task:
                f.write("分任务指标:\n")
                for task_type, task_metrics in by_task.items():
                    count = task_metrics.get("count", 0)
                    f.write(f"\n  {task_type} (n={count}):\n")
                    f.write(
                        f"    Precision: {task_metrics.get('precision', 0):.4f}\n"
                    )
                    f.write(f"    Recall:    {task_metrics.get('recall', 0):.4f}\n")
                    f.write(f"    F1:        {task_metrics.get('f1', 0):.4f}\n")
        
        logger.info(f"摘要已保存: {summary_path}")
