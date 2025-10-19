"""成对比较工具"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from ..core.base_llm import BaseLLM
from ..core.prompt_manager import PromptManager
from ..datasets import EvalResultsDataset
from ..schemas import JudgementResult

logger = logging.getLogger(__name__)


class PairwiseJudge:
    """成对比较工具
    
    用于比较两组输出（如同一模型的不同 prompt 或同一 prompt 的不同模型）
    设计风格与 BaseTask 保持一致，使用相同的 LLM 接口
    """
    
    def __init__(
        self,
        csv_path_a: str,
        csv_path_b: str,
        llm: BaseLLM,
        prompt_manager: PromptManager,
        output_dir: str,
    ):
        """初始化 PairwiseJudge
        
        Args:
            csv_path_a: 评估结果 A 的 CSV 路径
            csv_path_b: 评估结果 B 的 CSV 路径
            llm: LLM 实例
            prompt_manager: Prompt 管理器实例
            output_dir: 输出目录
        """
        self.dataset_a = EvalResultsDataset(csv_path_a)
        self.dataset_b = EvalResultsDataset(csv_path_b)
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"PairwiseJudge 初始化完成")
        logger.info(f"  - 数据集 A: {len(self.dataset_a)} 行")
        logger.info(f"  - 数据集 B: {len(self.dataset_b)} 行")
        logger.info(f"  - 输出目录: {self.output_dir}")
    
    def build_judge_chat(
        self,
        query: str,
        gold_standard: str,
        candidate_a: str,
        candidate_b: str
    ) -> list[dict[str, str]]:
        """构建判别 chat
        
        Args:
            query: 用户原始查询
            gold_standard: 金标准答案
            candidate_a: 候选A 的输出
            candidate_b: 候选B 的输出
            
        Returns:
            chat 消息列表
        """
        # 加载判别 prompt
        system_prompt = self.prompt_manager.load()
        
        user_content = f"""请评估以下两个候选输出，判断哪一个更接近金标准答案。

## 用户查询
```
{query}
```

## 金标准答案
```json
{gold_standard}
```

## 候选A 输出
```json
{candidate_a}
```

## 候选B 输出
```json
{candidate_b}
```

请根据系统提示中的评估标准，给出你的判断结果。
        """
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    
    def prepare_comparison_pairs(self) -> List[Dict[str, Any]]:
        """按 idx 配对两个数据集
        
        Returns:
            配对的样本列表
        """
        pairs = []
        
        # 获取两个数据集中都存在的 idx
        indices_a = set(self.dataset_a._df['idx'].unique())
        indices_b = set(self.dataset_b._df['idx'].unique())
        common_indices = sorted(indices_a & indices_b)
        
        logger.info(f"发现共同 idx: {len(common_indices)} 个")
        
        for idx in common_indices:
            try:
                row_a = self.dataset_a.get_row_by_idx(idx)
                row_b = self.dataset_b.get_row_by_idx(idx)
            except IndexError:
                logger.debug(f"跳过 idx={idx}，两个数据集中不匹配")
                continue
            
            # 跳过两者都失败的情况
            if not row_a.get("ok", False) and not row_b.get("ok", False):
                logger.debug(f"跳过 idx={idx}，两个候选都失败")
                continue
            
            pair = {
                'idx': int(idx),
                'query': str(row_a.get('query', '')),
                'gold_label': str(row_a.get('gold_label', '')) if pd.notna(row_a.get('gold_label')) else "{}",
                'response_a': str(row_a.get('raw_response', '')) if pd.notna(row_a.get('raw_response')) else "{}",
                'response_b': str(row_b.get('raw_response', '')) if pd.notna(row_b.get('raw_response')) else "{}",
                'ok_a': bool(row_a.get('ok', False)),
                'ok_b': bool(row_b.get('ok', False))
            }
            pairs.append(pair)
        
        logger.info(f"准备了 {len(pairs)} 个比较样本对")
        return pairs
    
    def run_judgement(self, temperature: float = 0.0) -> Dict[str, Any]:
        """运行判别流程
        
        Args:
            temperature: 采样温度
            
        Returns:
            分析结果字典
        """
        logger.info("开始准备比较样本对")
        pairs = self.prepare_comparison_pairs()
        
        if not pairs:
            logger.error("没有可比较的样本对")
            return {}
        
        logger.info(f"开始 LLM 判别，共 {len(pairs)} 对")
        results = []
        
        for pair in tqdm(pairs, desc="LLM 判别中"):
            # 构建判别 chat
            chat = self.build_judge_chat(
                pair["query"],
                pair["gold_label"],
                pair["response_a"],
                pair["response_b"]
            )
            
            # 调用判别模型
            parsed, raw, dt = self.llm.generate_structured(
                chat,
                JudgementResult,
                temperature=temperature
            )
            
            # 记录结果
            result = {
                "idx": pair["idx"],
                "query": pair["query"],
                "winner": parsed.winner if parsed else "error",
                "confidence": parsed.confidence if parsed else 0.0,
                "reason": parsed.reason if parsed else "判别失败",
                "latency_sec": dt,
                "raw_judgement": raw
            }
            
            # 添加维度得分
            if parsed and parsed.dimensions:
                dims = parsed.dimensions
                result.update({
                    "structure_a": dims.structure_a,
                    "structure_b": dims.structure_b,
                    "semantic_a": dims.semantic_a,
                    "semantic_b": dims.semantic_b,
                    "completeness_a": dims.completeness_a,
                    "completeness_b": dims.completeness_b,
                    "format_a": dims.format_a,
                    "format_b": dims.format_b
                })
            else:
                # 填充默认值
                result.update({
                    "structure_a": 0.0,
                    "structure_b": 0.0,
                    "semantic_a": 0.0,
                    "semantic_b": 0.0,
                    "completeness_a": 0.0,
                    "completeness_b": 0.0,
                    "format_a": 0.0,
                    "format_b": 0.0
                })
            
            results.append(result)
            
            # 延迟避免过载
            time.sleep(0.1)
        
        # 分析结果
        analysis = self.analyze_results(results)
        
        # 生成报告
        results_df = pd.DataFrame(results)
        self.generate_report(analysis, results_df)
        
        return analysis
    
    def analyze_results(self, results: List[Dict]) -> Dict[str, Any]:
        """分析判别结果
        
        Args:
            results: 判别结果列表
            
        Returns:
            分析统计字典
        """
        df = pd.DataFrame(results)
        
        # 统计胜负
        a_wins = (df["winner"] == "candidate_a").sum()
        b_wins = (df["winner"] == "candidate_b").sum()
        ties = (df["winner"] == "tie").sum()
        errors = (df["winner"] == "error").sum()
        
        total_valid = a_wins + b_wins + ties
        
        # 平均置信度
        avg_confidence = df["confidence"].mean()
        
        # 高置信度判别（confidence >= 0.7）
        high_conf = df[df["confidence"] >= 0.7]
        high_conf_a_wins = (high_conf["winner"] == "candidate_a").sum()
        high_conf_b_wins = (high_conf["winner"] == "candidate_b").sum()
        
        # 各维度平均得分
        dim_scores = {
            "structure_a": df["structure_a"].mean(),
            "structure_b": df["structure_b"].mean(),
            "semantic_a": df["semantic_a"].mean(),
            "semantic_b": df["semantic_b"].mean(),
            "completeness_a": df["completeness_a"].mean(),
            "completeness_b": df["completeness_b"].mean(),
            "format_a": df["format_a"].mean(),
            "format_b": df["format_b"].mean()
        }
        
        # 综合得分（加权）
        weights = {
            "structure": 0.3,
            "semantic": 0.4,
            "completeness": 0.2,
            "format": 0.1
        }
        
        a_overall = (
            dim_scores["structure_a"] * weights["structure"] +
            dim_scores["semantic_a"] * weights["semantic"] +
            dim_scores["completeness_a"] * weights["completeness"] +
            dim_scores["format_a"] * weights["format"]
        )
        
        b_overall = (
            dim_scores["structure_b"] * weights["structure"] +
            dim_scores["semantic_b"] * weights["semantic"] +
            dim_scores["completeness_b"] * weights["completeness"] +
            dim_scores["format_b"] * weights["format"]
        )
        
        analysis = {
            "total_judgements": len(df),
            "a_wins": int(a_wins),
            "b_wins": int(b_wins),
            "ties": int(ties),
            "errors": int(errors),
            "a_win_rate": a_wins / total_valid if total_valid > 0 else 0,
            "b_win_rate": b_wins / total_valid if total_valid > 0 else 0,
            "tie_rate": ties / total_valid if total_valid > 0 else 0,
            "avg_confidence": float(avg_confidence),
            "high_conf_judgements": len(high_conf),
            "high_conf_a_wins": int(high_conf_a_wins),
            "high_conf_b_wins": int(high_conf_b_wins),
            "dimension_scores": dim_scores,
            "a_overall_score": float(a_overall),
            "b_overall_score": float(b_overall)
        }
        
        logger.info(f"分析完成: A 胜率 {analysis['a_win_rate']:.1%}, B 胜率 {analysis['b_win_rate']:.1%}")
        return analysis
    
    def generate_report(self, analysis: Dict[str, Any], results_df: pd.DataFrame) -> None:
        """生成判别报告
        
        Args:
            analysis: 分析统计
            results_df: 判别结果 DataFrame
        """
        # 保存详细结果
        results_path = self.output_dir / "judgement_results.csv"
        results_df.to_csv(results_path, index=False, encoding='utf-8')
        logger.info(f"详细判别结果已保存: {results_path}")
        
        # 保存统计分析
        analysis_path = self.output_dir / "judgement_analysis.json"
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        logger.info(f"统计分析已保存: {analysis_path}")
        
        # 生成 Markdown 报告
        md_lines = []
        md_lines.append("# 成对比较判别报告")
        md_lines.append("")
        md_lines.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md_lines.append("")
        
        md_lines.append("## 📊 胜负统计")
        md_lines.append("")
        md_lines.append(f"- 总判别数: {analysis['total_judgements']}")
        md_lines.append(f"- **方案A 胜出**: {analysis['a_wins']} ({analysis['a_win_rate']*100:.1f}%)")
        md_lines.append(f"- **方案B 胜出**: {analysis['b_wins']} ({analysis['b_win_rate']*100:.1f}%)")
        md_lines.append(f"- **平局**: {analysis['ties']} ({analysis['tie_rate']*100:.1f}%)")
        md_lines.append(f"- 判别失败: {analysis['errors']}")
        md_lines.append("")
        
        md_lines.append("## 🎯 高置信度判别（confidence ≥ 0.7）")
        md_lines.append("")
        md_lines.append(f"- 高置信判别数: {analysis['high_conf_judgements']}")
        md_lines.append(f"- 方案A 胜出: {analysis['high_conf_a_wins']}")
        md_lines.append(f"- 方案B 胜出: {analysis['high_conf_b_wins']}")
        md_lines.append("")
        
        md_lines.append("## 📈 维度得分对比（满分10分）")
        md_lines.append("")
        md_lines.append("| 维度 | 方案A | 方案B | 差异 | 优势方 |")
        md_lines.append("|------|-------|-------|------|--------|")
        
        dims = analysis["dimension_scores"]
        dim_pairs = [
            ("结构完整性", "structure_a", "structure_b"),
            ("语义准确性", "semantic_a", "semantic_b"),
            ("信息完整度", "completeness_a", "completeness_b"),
            ("格式规范性", "format_a", "format_b")
        ]
        
        for label, a_key, b_key in dim_pairs:
            a_val = dims[a_key]
            b_val = dims[b_key]
            diff = a_val - b_val
            winner = "✅ A" if diff > 0.5 else "✅ B" if diff < -0.5 else "➖ 相当"
            md_lines.append(f"| {label} | {a_val:.2f} | {b_val:.2f} | {diff:+.2f} | {winner} |")
        
        md_lines.append("")
        
        md_lines.append("## 🏆 综合得分")
        md_lines.append("")
        md_lines.append(f"- **方案A**: {analysis['a_overall_score']:.2f} 分")
        md_lines.append(f"- **方案B**: {analysis['b_overall_score']:.2f} 分")
        md_lines.append(f"- **差异**: {analysis['a_overall_score'] - analysis['b_overall_score']:+.2f} 分")
        md_lines.append("")
        
        md_lines.append("## 💡 结论")
        md_lines.append("")
        if analysis['a_win_rate'] > 0.6:
            md_lines.append(f"✅ **方案A 显著优于方案B**")
            md_lines.append(f"   - 胜率: {analysis['a_win_rate']*100:.1f}%")
            md_lines.append(f"   - 综合得分: {analysis['a_overall_score']:.2f} vs {analysis['b_overall_score']:.2f}")
        elif analysis['b_win_rate'] > 0.6:
            md_lines.append(f"✅ **方案B 显著优于方案A**")
            md_lines.append(f"   - 胜率: {analysis['b_win_rate']*100:.1f}%")
            md_lines.append(f"   - 综合得分: {analysis['b_overall_score']:.2f} vs {analysis['a_overall_score']:.2f}")
        else:
            md_lines.append(f"➖ **两个方案表现相当**")
            md_lines.append(f"   - 方案A 胜率: {analysis['a_win_rate']*100:.1f}%")
            md_lines.append(f"   - 方案B 胜率: {analysis['b_win_rate']*100:.1f}%")
        
        md_lines.append("")
        md_lines.append(f"平均置信度: {analysis['avg_confidence']:.2f}")
        md_lines.append("")
        
        # 保存 Markdown 报告
        md_path = self.output_dir / "judgement_report.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(md_lines))
        logger.info(f"Markdown 报告已保存: {md_path}")
