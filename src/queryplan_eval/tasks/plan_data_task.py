"""计划数据任务 - 支持多模式评估

支持四种模式：
- single: 单轮查询，输出 type 字段
- multi: 多轮对话，输出 time_frame 字段
- single_think: 单轮查询，包含推理过程 think
- multi_think: 多轮对话，包含推理过程 think
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import Dataset

from ..core.base_task import BaseTask
from ..core.prompt_manager import PromptManager
from ..datasets.plan_data import PlanDataset, PlanDataItem
from ..schemas_plan_data import (
    PlanDataSingleResult,
    PlanDataMultiResult,
    PlanDataSingleThinkResult,
    PlanDataMultiThinkResult,
    normalize_plan_data_result,
)

logger = logging.getLogger(__name__)


class PlanDataTask(BaseTask):
    """计划数据评估任务
    
    从 plan_data.xlsx 加载四个工作表，支持单/多轮查询、含/不含推理过程。
    动态注入 Excel 中的权威 domain/sub 枚举到提示词中。
    """
    
    def __init__(
        self,
        mode: str = "single",
        enable_cot: bool = True,
        sample_n: Optional[int] = None,
        **kwargs
    ):
        """初始化计划数据任务
        
        Args:
            mode: 模式 ('single', 'multi', 'single_think', 'multi_think')
            enable_cot: 是否在提示词中启用 CoT 提示
            sample_n: 采样数量
            **kwargs: 传递给 BaseTask 的参数
        """
        if mode not in ("single", "multi", "single_think", "multi_think"):
            raise ValueError(f"mode 必须是 single/multi/single_think/multi_think 之一，得到: {mode}")
        
        self.mode = mode
        self.enable_cot = enable_cot
        self.sample_n = sample_n
        super().__init__(**kwargs)
    
    def load_dataset(self, path: str) -> Dataset:
        """加载计划数据集
        
        Args:
            path: Excel 文件路径
        
        Returns:
            加载后的 HuggingFace Dataset
        """
        dataset = PlanDataset(path, mode=self.mode, n=self.sample_n)
        logger.info(f"加载数据集 (mode={self.mode}): {len(dataset)} 个样本")
        return dataset
    
    def build_chat(self, item: PlanDataItem) -> list[dict[str, str]]:
        """构造用于 LLM 的对话格式
        
        Args:
            item: 数据样本
        
        Returns:
            [{"role": "user", "content": "..."}] 格式
        """
        # 获取权威枚举
        dataset: PlanDataset = self.dataset
        domains = dataset.get_domains()
        subs_map = dataset.get_subs_map()
        
        # 加载并渲染提示词
        prompt_manager = PromptManager(
            task_name="query_plan/plan_data",
            version=f"{self.mode}_v1"
        )
        
        # 根据模式构造渲染变量
        render_vars = {
            "today": item.date,
            "domains": domains,
            "subs_map": subs_map,
        }
        
        # 对于 multi/multi_think 模式，添加 history
        if self.mode in ("multi", "multi_think"):
            render_vars["history"] = item.history or ""
        
        # 添加当前问题
        render_vars["question"] = item.query
        
        # 渲染提示词
        system_prompt = prompt_manager.load(**render_vars)
        
        # 构造对话
        return [
            {"role": "user", "content": system_prompt}
        ]
    
    def get_output_schema(self) -> type:
        """返回对应模式的输出类型"""
        mode_to_schema = {
            "single": PlanDataSingleResult,
            "multi": PlanDataMultiResult,
            "single_think": PlanDataSingleThinkResult,
            "multi_think": PlanDataMultiThinkResult,
        }
        return mode_to_schema[self.mode]
    
    def process_single_result(
        self,
        item: PlanDataItem,
        parsed: Optional[Any],
        raw: Optional[str],
        latency: float
    ) -> Dict:
        """处理单个样本的结果
        
        Args:
            item: 原始样本
            parsed: 解析后的结果对象（某个 PlanDataXxxResult）
            raw: 原始 LLM 输出
            latency: 请求耗时
        
        Returns:
            处理后的结果记录
        """
        ok = parsed is not None
        err = None
        norm = None
        
        # 基本结果记录
        record = {
            "idx": item.idx,
            "mode": self.mode,
            "query": str(item.query),
            "date": str(item.date),
            "raw_response": raw,
            "ok": ok,
            "latency_sec": latency,
            "error": err,
        }
        
        # 添加金标签字段（来自 Excel）
        record.update({
            "gold_domain": item.domain,
            "gold_sub": item.sub,
            "gold_is_personal": item.is_personal,
            "gold_time": item.time,
            "gold_food": item.food,
        })
        
        # 根据模式添加额外的金标签字段
        if self.mode == "single":
            record["gold_type"] = item.type
        elif self.mode == "multi":
            record["gold_time_frame"] = item.time_frame
        elif self.mode in ("single_think", "multi_think"):
            record["gold_think"] = item.think
        
        # 解析结果
        if ok and parsed is not None:
            # 规范化结果
            norm = normalize_plan_data_result(parsed, include_reasoning=False)
            
            # 添加预测字段
            record.update({
                "pred_domain": norm.get("domain"),
                "pred_sub": norm.get("sub"),
                "pred_is_personal": norm.get("is_personal"),
                "pred_time": norm.get("time"),
                "pred_food": norm.get("food"),
            })
            
            # 根据模式添加额外的预测字段
            if self.mode == "single":
                record["pred_type"] = norm.get("type")
            elif self.mode == "multi":
                record["pred_time_frame"] = norm.get("time_frame")
            elif self.mode in ("single_think", "multi_think"):
                record["pred_think"] = norm.get("think")
            
            # 计算字段级准确率
            record.update(self._compute_field_accuracy(item, norm))
        
        return record
    
    def _compute_field_accuracy(self, gold: PlanDataItem, pred: Dict) -> Dict:
        """计算字段级准确率
        
        Args:
            gold: 金标签样本
            pred: 预测结果字典
        
        Returns:
            字段准确率字典
        """
        accuracy = {}
        
        # 核心字段准确率
        accuracy["acc_domain"] = int(gold.domain == pred.get("domain"))
        accuracy["acc_sub"] = int(gold.sub == pred.get("sub"))
        accuracy["acc_is_personal"] = int(gold.is_personal == pred.get("is_personal"))
        accuracy["acc_time"] = int(gold.time == pred.get("time"))
        accuracy["acc_food"] = int(gold.food == pred.get("food"))
        
        # 模式特定字段准确率
        if self.mode == "single":
            accuracy["acc_type"] = int(gold.type == pred.get("type"))
        elif self.mode == "multi":
            accuracy["acc_time_frame"] = int(gold.time_frame == pred.get("time_frame"))
        elif self.mode in ("single_think", "multi_think"):
            # think 字段通常需要语义相似度，这里简单用精确匹配
            accuracy["acc_think"] = int(gold.think == pred.get("think"))
        
        return accuracy
    
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
        
        metrics = {
            "total": int(total),
            "ok": int(ok),
            "ok_rate": float(ok / total) if total > 0 else 0.0,
        }
        
        # 计算各字段的准确率
        for field in ["domain", "sub", "is_personal", "time", "food"]:
            acc_col = f"acc_{field}"
            if acc_col in df.columns:
                acc_vals = df[acc_col].dropna()
                if len(acc_vals) > 0:
                    metrics[f"acc_{field}"] = float(acc_vals.mean())
        
        # 模式特定指标
        if self.mode == "single":
            if "acc_type" in df.columns:
                acc_vals = df["acc_type"].dropna()
                if len(acc_vals) > 0:
                    metrics["acc_type"] = float(acc_vals.mean())
        elif self.mode == "multi":
            if "acc_time_frame" in df.columns:
                acc_vals = df["acc_time_frame"].dropna()
                if len(acc_vals) > 0:
                    metrics["acc_time_frame"] = float(acc_vals.mean())
        elif self.mode in ("single_think", "multi_think"):
            if "acc_think" in df.columns:
                acc_vals = df["acc_think"].dropna()
                if len(acc_vals) > 0:
                    metrics["acc_think"] = float(acc_vals.mean())
        
        # 延迟统计
        lat_vals = df["latency_sec"].dropna()
        if len(lat_vals) > 0:
            metrics["latency_mean"] = float(lat_vals.mean())
            metrics["latency_p95"] = float(lat_vals.quantile(0.95))
        
        logger.info(f"指标计算完成: {metrics}")
        return metrics
    
    def save_results(self, results: List[Dict], metrics: Dict) -> None:
        """保存结果到文件
        
        Args:
            results: 结果列表
            metrics: 指标字典
        """
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存结果 CSV
        df = pd.DataFrame(results)
        results_file = output_dir / "results.csv"
        df.to_csv(results_file, index=False, encoding="utf-8-sig")
        logger.info(f"结果已保存: {results_file}")
        
        # 保存指标 JSON
        metrics_file = output_dir / "metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"指标已保存: {metrics_file}")
