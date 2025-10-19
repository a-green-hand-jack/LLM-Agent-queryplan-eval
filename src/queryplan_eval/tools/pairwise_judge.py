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
        eval_results_path: str,
        llm: BaseLLM,
        prompt_manager: PromptManager,
        output_dir: str,
    ):
        """初始化 PairwiseJudge
        
        Args:
            eval_results_path: eval_results.csv 文件路径
            llm: LLM 实例
            prompt_manager: Prompt 管理器实例
            output_dir: 输出目录
        """
        self.dataset = EvalResultsDataset(eval_results_path)
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"PairwiseJudge 初始化完成，输出目录: {self.output_dir}")
    
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
            candidate_a: 候选A 的输出（通常是 new）
            candidate_b: 候选B 的输出（通常是 old）
            
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
    
    def run_judgement(self, temperature: float = 0.0) -> Dict[str, Any]:
        """运行判别流程
        
        Args:
            temperature: 采样温度
            
        Returns:
            分析结果字典
        """
        logger.info("开始准备比较样本对")
        pairs = self.dataset.prepare_comparison_pairs()
        
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
                pair["new_response"],
                pair["old_response"]
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
                    "structure_a_new": dims.structure_a,
                    "structure_b_old": dims.structure_b,
                    "semantic_a_new": dims.semantic_a,
                    "semantic_b_old": dims.semantic_b,
                    "completeness_a_new": dims.completeness_a,
                    "completeness_b_old": dims.completeness_b,
                    "format_a_new": dims.format_a,
                    "format_b_old": dims.format_b
                })
            else:
                # 填充默认值
                result.update({
                    "structure_a_new": 0.0,
                    "structure_b_old": 0.0,
                    "semantic_a_new": 0.0,
                    "semantic_b_old": 0.0,
                    "completeness_a_new": 0.0,
                    "completeness_b_old": 0.0,
                    "format_a_new": 0.0,
                    "format_b_old": 0.0
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
        new_wins = (df["winner"] == "candidate_a").sum()
        old_wins = (df["winner"] == "candidate_b").sum()
        ties = (df["winner"] == "tie").sum()
        errors = (df["winner"] == "error").sum()
        
        total_valid = new_wins + old_wins + ties
        
        # 平均置信度
        avg_confidence = df["confidence"].mean()
        
        # 高置信度判别（confidence >= 0.7）
        high_conf = df[df["confidence"] >= 0.7]
        high_conf_new_wins = (high_conf["winner"] == "candidate_a").sum()
        high_conf_old_wins = (high_conf["winner"] == "candidate_b").sum()
        
        # 各维度平均得分
        dim_scores = {
            "new_structure": df["structure_a_new"].mean(),
            "old_structure": df["structure_b_old"].mean(),
            "new_semantic": df["semantic_a_new"].mean(),
            "old_semantic": df["semantic_b_old"].mean(),
            "new_completeness": df["completeness_a_new"].mean(),
            "old_completeness": df["completeness_b_old"].mean(),
            "new_format": df["format_a_new"].mean(),
            "old_format": df["format_b_old"].mean()
        }
        
        # 综合得分（加权）
        weights = {
            "structure": 0.3,
            "semantic": 0.4,
            "completeness": 0.2,
            "format": 0.1
        }
        
        new_overall = (
            dim_scores["new_structure"] * weights["structure"] +
            dim_scores["new_semantic"] * weights["semantic"] +
            dim_scores["new_completeness"] * weights["completeness"] +
            dim_scores["new_format"] * weights["format"]
        )
        
        old_overall = (
            dim_scores["old_structure"] * weights["structure"] +
            dim_scores["old_semantic"] * weights["semantic"] +
            dim_scores["old_completeness"] * weights["completeness"] +
            dim_scores["old_format"] * weights["format"]
        )
        
        analysis = {
            "total_judgements": len(df),
            "new_wins": int(new_wins),
            "old_wins": int(old_wins),
            "ties": int(ties),
            "errors": int(errors),
            "new_win_rate": new_wins / total_valid if total_valid > 0 else 0,
            "old_win_rate": old_wins / total_valid if total_valid > 0 else 0,
            "tie_rate": ties / total_valid if total_valid > 0 else 0,
            "avg_confidence": float(avg_confidence),
            "high_conf_judgements": len(high_conf),
            "high_conf_new_wins": int(high_conf_new_wins),
            "high_conf_old_wins": int(high_conf_old_wins),
            "dimension_scores": dim_scores,
            "new_overall_score": float(new_overall),
            "old_overall_score": float(old_overall)
        }
        
        logger.info(f"分析完成: NEW 胜率 {analysis['new_win_rate']:.1%}, OLD 胜率 {analysis['old_win_rate']:.1%}")
        return analysis
    
    def generate_report(self, analysis: Dict[str, Any], results_df: pd.DataFrame) -> None:
        """生成判别报告
        
        Args:
            analysis: 分析统计
            results_df: 判别结果 DataFrame
        """
        # 保存详细结果
        results_path = self.output_dir / "llm_judgement_results.csv"
        results_df.to_csv(results_path, index=False, encoding='utf-8')
        logger.info(f"详细判别结果已保存: {results_path}")
        
        # 保存统计分析
        analysis_path = self.output_dir / "llm_judgement_analysis.json"
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        logger.info(f"统计分析已保存: {analysis_path}")
        
        # 生成 Markdown 报告
        md_lines = []
        md_lines.append("# LLM 判别报告")
        md_lines.append("")
        md_lines.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md_lines.append("")
        
        md_lines.append("## 📊 胜负统计")
        md_lines.append("")
        md_lines.append(f"- 总判别数: {analysis['total_judgements']}")
        md_lines.append(f"- **NEW 胜出**: {analysis['new_wins']} ({analysis['new_win_rate']*100:.1f}%)")
        md_lines.append(f"- **OLD 胜出**: {analysis['old_wins']} ({analysis['old_win_rate']*100:.1f}%)")
        md_lines.append(f"- **平局**: {analysis['ties']} ({analysis['tie_rate']*100:.1f}%)")
        md_lines.append(f"- 判别失败: {analysis['errors']}")
        md_lines.append("")
        
        md_lines.append("## 🎯 高置信度判别（confidence ≥ 0.7）")
        md_lines.append("")
        md_lines.append(f"- 高置信判别数: {analysis['high_conf_judgements']}")
        md_lines.append(f"- NEW 胜出: {analysis['high_conf_new_wins']}")
        md_lines.append(f"- OLD 胜出: {analysis['high_conf_old_wins']}")
        md_lines.append("")
        
        md_lines.append("## 📈 维度得分对比（满分10分）")
        md_lines.append("")
        md_lines.append("| 维度 | NEW | OLD | 差异 | 优势方 |")
        md_lines.append("|------|-----|-----|------|--------|")
        
        dims = analysis["dimension_scores"]
        dim_pairs = [
            ("结构完整性", "new_structure", "old_structure"),
            ("语义准确性", "new_semantic", "old_semantic"),
            ("信息完整度", "new_completeness", "old_completeness"),
            ("格式规范性", "new_format", "old_format")
        ]
        
        for label, new_key, old_key in dim_pairs:
            new_val = dims[new_key]
            old_val = dims[old_key]
            diff = new_val - old_val
            winner = "✅ NEW" if diff > 0.5 else "✅ OLD" if diff < -0.5 else "➖ 相当"
            md_lines.append(f"| {label} | {new_val:.2f} | {old_val:.2f} | {diff:+.2f} | {winner} |")
        
        md_lines.append("")
        
        md_lines.append("## 🏆 综合得分")
        md_lines.append("")
        md_lines.append(f"- **NEW Prompt**: {analysis['new_overall_score']:.2f} 分")
        md_lines.append(f"- **OLD Prompt**: {analysis['old_overall_score']:.2f} 分")
        md_lines.append(f"- **差异**: {analysis['new_overall_score'] - analysis['old_overall_score']:+.2f} 分")
        md_lines.append("")
        
        md_lines.append("## 💡 结论")
        md_lines.append("")
        if analysis['new_win_rate'] > 0.6:
            md_lines.append(f"✅ **NEW Prompt 显著优于 OLD Prompt**")
            md_lines.append(f"   - 胜率: {analysis['new_win_rate']*100:.1f}%")
            md_lines.append(f"   - 综合得分: {analysis['new_overall_score']:.2f} vs {analysis['old_overall_score']:.2f}")
        elif analysis['old_win_rate'] > 0.6:
            md_lines.append(f"❌ **OLD Prompt 优于 NEW Prompt**")
            md_lines.append(f"   - 胜率: {analysis['old_win_rate']*100:.1f}%")
            md_lines.append(f"   - 综合得分: {analysis['old_overall_score']:.2f} vs {analysis['new_overall_score']:.2f}")
        else:
            md_lines.append(f"➖ **两者表现相当**")
            md_lines.append(f"   - NEW 胜率: {analysis['new_win_rate']*100:.1f}%")
            md_lines.append(f"   - OLD 胜率: {analysis['old_win_rate']*100:.1f}%")
        
        md_lines.append("")
        md_lines.append(f"平均置信度: {analysis['avg_confidence']:.2f}")
        md_lines.append("")
        
        # 保存 Markdown 报告
        md_path = self.output_dir / "llm_judgement_report.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(md_lines))
        logger.info(f"Markdown 报告已保存: {md_path}")
