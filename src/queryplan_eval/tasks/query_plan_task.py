"""查询计划抽取任务"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import Dataset

from ..core.base_task import BaseTask
from ..datasets import QueryPlanDataset, SplitConfig, split_dataset, take_samples
from ..schemas import QueryResult, normalize_result

logger = logging.getLogger(__name__)


class QueryPlanTask(BaseTask):
    """查询计划抽取任务
    
    从用户查询中抽取结构化的查询计划参数，支持 CoT 推理
    """
    
    def __init__(
        self,
        enable_cot: bool = False,
        sample_n: Optional[int] = None,
        **kwargs
    ):
        """初始化查询计划任务
        
        Args:
            enable_cot: 是否启用 Chain-of-Thought 推理
            sample_n: 如果指定，从数据集中随机采样 n 个样本
            **kwargs: 传递给 BaseTask 的参数
        """
        self.enable_cot = enable_cot
        self.sample_n = sample_n
        super().__init__(**kwargs)
    
    def load_dataset(self, path: str) -> Dataset:
        """加载查询计划数据集
        
        Args:
            path: Excel 文件路径
            
        Returns:
            加载后的 HuggingFace Dataset
        """
        dataset = QueryPlanDataset(path, n=self.sample_n)
        logger.info(f"加载数据集: {len(dataset)} 个样本")
        return dataset
    
    def split_dataset(self, **split_config) -> Dict[str, Dataset]:
        """分割数据集
        
        Args:
            **split_config: 分割配置（split_type, train_ratio 等）
            
        Returns:
            分割后的数据集字典
        """
        config = SplitConfig(**split_config)
        splits = split_dataset(self.dataset, config)
        logger.info(f"数据集分割完成: {splits}")
        return splits
    
    def build_chat(self, item: Any) -> list[dict[str, str]]:
        """构建 chat 消息
        
        Args:
            item: 查询计划数据项
            
        Returns:
            chat 消息列表
        """
        # 确定使用的 prompt 版本
        if self.enable_cot:
            prompt_version = "v6_cot"
        else:
            prompt_version = "latest"
        
        # 加载并渲染 prompt
        today = datetime.now().strftime("%Y年%m月%d日")
        system_prompt = self.prompt_manager.load(today=today)
        
        # 构建用户查询 payload
        payload = json.dumps(
            {
                "history": {"user": None, "assistant": None},
                "question": str(item.query)
            },
            ensure_ascii=False
        )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": payload}
        ]
    
    def get_output_schema(self) -> type:
        """返回输出类型"""
        return QueryResult
    
    def process_single_result(
        self,
        item: Any,
        parsed: Optional[QueryResult],
        raw: Optional[str],
        latency: float
    ) -> Dict:
        """处理单个样本的结果
        
        Args:
            item: 原始样本
            parsed: 解析后的 QueryResult
            raw: 原始输出
            latency: 耗时
            
        Returns:
            处理后的结果记录
        """
        ok = parsed is not None
        n_plans = 0
        out_type = None
        err = None
        reasoning_fields = {}
        norm = None
        
        if ok and parsed is not None:
            if parsed.refused:
                out_type = "refuse"
                n_plans = 0
            else:
                out_type = "plans"
                n_plans = len(parsed.plans)
            
            norm = normalize_result(parsed, include_reasoning=False)
            
            # 如果启用 CoT 且存在 reasoning，提取各个步骤
            if self.enable_cot and parsed.reasoning is not None:
                reasoning_fields = {
                    "reasoning_query_analysis": parsed.reasoning.query_analysis,
                    "reasoning_domain_identification": parsed.reasoning.domain_identification,
                    "reasoning_time_calculation": parsed.reasoning.time_calculation,
                    "reasoning_refuse_check": parsed.reasoning.refuse_check,
                    "reasoning_final_decision": parsed.reasoning.final_decision
                }
        else:
            out_type = "parse_error" if raw else "exception"
        
        record = {
            "idx": item.idx,
            "query": str(item.query),
            "raw_response": raw,
            "ok": ok,
            "type": out_type,
            "n_plans": n_plans,
            "latency_sec": latency,
            "parsed": json.dumps(norm, ensure_ascii=False) if norm else None,
            "gold_label": str(item.plan) if item.plan else None,
            "error": err,
        }
        
        # 添加 CoT reasoning 字段
        if self.enable_cot:
            record.update(reasoning_fields)
        
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
        refuse = (df["type"] == "refuse").sum()
        parse_err = df["type"].isin(["parse_error", "exception"]).sum()
        
        lat_mean = df["latency_sec"].dropna().mean()
        lat_p95 = df["latency_sec"].dropna().quantile(0.95) if len(df["latency_sec"].dropna()) > 0 else None
        
        metrics = {
            "total": int(total),
            "ok": int(ok),
            "ok_rate": float(ok / total) if total > 0 else 0.0,
            "refuse": int(refuse),
            "refuse_rate": float(refuse / total) if total > 0 else 0.0,
            "parse_error": int(parse_err),
            "parse_error_rate": float(parse_err / total) if total > 0 else 0.0,
            "latency_mean": float(lat_mean) if pd.notna(lat_mean) else None,
            "latency_p95": float(lat_p95) if lat_p95 is not None else None,
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
            f.write(f"拒答: {metrics['refuse']} ({metrics['refuse_rate']:.1%})\n")
            f.write(f"解析错误: {metrics['parse_error']} ({metrics['parse_error_rate']:.1%})\n")
            if metrics["latency_mean"]:
                f.write(f"平均延迟: {metrics['latency_mean']:.3f}s\n")
            if metrics["latency_p95"]:
                f.write(f"P95 延迟: {metrics['latency_p95']:.3f}s\n")
        logger.info(f"摘要已保存: {summary_path}")
        
        # 计算差异（new vs old）
        df_pivot = df.pivot_table(
            index=["idx", "query"], 
            columns="variant" if "variant" in df.columns else None,
            values="parsed",
            aggfunc="first"
        ).reset_index()
        
        if "new" in df_pivot.columns and "old" in df_pivot.columns:
            diffs = df_pivot[
                (df_pivot["new"] != df_pivot["old"]) &
                df_pivot["new"].notna() &
                df_pivot["old"].notna()
            ]
            diffs_path = self.output_dir / "diffs.csv"
            diffs.to_csv(diffs_path, index=False, encoding='utf-8')
            logger.info(f"差异已保存: {diffs_path} ({len(diffs)} 条记录)")
