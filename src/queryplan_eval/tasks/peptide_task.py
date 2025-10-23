"""肽段专利任务"""

import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import Dataset

from ..core.base_task import BaseTask
from ..core.prompt_manager import PatentPromptManager
from ..datasets.peptide_patent import PeptideDataset
from ..schemas import PeptidePatentRepresentation

logger = logging.getLogger(__name__)


class PeptideTask(BaseTask):
    """肽段专利任务
    
    从肽段序列数据中调用 LLM 进行专利风格线性表示转换
    """
    
    def __init__(
        self,
        sample_n: Optional[int] = None,
        **kwargs
    ):
        """初始化肽段任务
        
        Args:
            sample_n: 如果指定，从数据集中随机采样 n 个样本
            **kwargs: 传递给 BaseTask 的参数
        """
        self.sample_n = sample_n
        super().__init__(**kwargs)
    
    def load_dataset(self, path: str) -> Dataset:
        """加载肽段数据集
        
        Args:
            path: CSV 文件路径
            
        Returns:
            加载后的肽段数据集
        """
        dataset = PeptideDataset(path, n=self.sample_n)
        logger.info(f"加载数据集: {len(dataset)} 个肽段样本")
        return dataset  # type: ignore[return-value]
    
    def build_chat(self, item: Any) -> list[dict[str, str]]:
        """构建 chat 消息
        
        Args:
            item: 肽段数据项 (PeptideItem)
            
        Returns:
            chat 消息列表
        """
        # 加载专利 prompt
        prompt_manager = PatentPromptManager(version="v1")
        system_prompt = prompt_manager.load()
        
        # 构建用户查询 payload - 包含肽段序列信息
        payload = json.dumps(
            {
                "sequence": item.sequence.get("sequence", ""),
                "features": item.sequence.get("features", [])
            },
            ensure_ascii=False
        )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": payload}
        ]
    
    def get_output_schema(self) -> type:
        """返回输出类型"""
        return PeptidePatentRepresentation
    
    def process_single_result(
        self,
        item: Any,
        parsed: Optional[PeptidePatentRepresentation],
        raw: Optional[str],
        latency: float
    ) -> Dict:
        """处理单个样本的结果
        
        Args:
            item: 原始样本
            parsed: 解析后的 PeptidePatentRepresentation
            raw: 原始输出
            latency: 耗时
            
        Returns:
            处理后的结果记录
        """
        ok = parsed is not None
        representation = None
        err = None
        
        if ok and parsed is not None:
            representation = parsed.representation
        
        record = {
            "idx": item.idx,
            "sequence": item.sequence.get("sequence", ""),
            "raw_response": raw,
            "ok": ok,
            "representation": representation,
            "latency_sec": latency,
            "error": err,
        }
        
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
        
        lat_mean_value = df["latency_sec"].dropna().mean()
        lat_series = df["latency_sec"].dropna()
        lat_p95_value = lat_series.quantile(0.95) if len(lat_series) > 0 else None
        
        # 安全地转换为 Python float
        lat_mean: Optional[float] = None
        if isinstance(lat_mean_value, (int, float)) and pd.notna(lat_mean_value):
            lat_mean = float(lat_mean_value)
        
        lat_p95: Optional[float] = None
        if lat_p95_value is not None and isinstance(lat_p95_value, (int, float)):
            lat_p95 = float(lat_p95_value)
        
        metrics = {
            "total": int(total),
            "ok": int(ok),
            "ok_rate": float(ok / total) if total > 0 else 0.0,
            "latency_mean": lat_mean,
            "latency_p95": lat_p95,
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
        csv_path = self.output_dir / "eval_results.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"结果已保存: {csv_path}")
        
        # 保存摘要
        summary_path = self.output_dir / "summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"总样本数: {metrics['total']}\n")
            f.write(f"成功: {metrics['ok']} ({metrics['ok_rate']:.1%})\n")
            if metrics["latency_mean"] is not None:
                f.write(f"平均延迟: {metrics['latency_mean']:.3f}s\n")
            if metrics["latency_p95"] is not None:
                f.write(f"P95 延迟: {metrics['latency_p95']:.3f}s\n")
        logger.info(f"摘要已保存: {summary_path}")
