#!/usr/bin/env python3
"""基于 LLM 的 Prompt 效果判别脚本

使用 qwen3-max 模型作为评判器，对比 new 和 old prompt 的输出质量。

使用方式:
    uv run python scripts/llm_judge.py outputs/v4/eval_results.csv
    uv run python scripts/llm_judge.py outputs/v4/eval_results.csv --model qwen-max
"""

import sys
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import os
import json
import time
import logging
from typing import Any, Optional, Tuple
from argparse import ArgumentParser

import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

import openai
import outlines

from queryplan_eval.schemas import JudgementResult
from queryplan_eval.renderer import read_raw_prompt

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def build_judge_chat(
    system_prompt: str,
    query: str,
    gold_standard: str,
    candidate_a: str,
    candidate_b: str
) -> list[dict[str, Any]]:
    """构建判别请求的聊天消息
    
    Args:
        system_prompt: 系统提示
        query: 用户查询
        gold_standard: 金标准答案
        candidate_a: 候选A的输出
        candidate_b: 候选B的输出
        
    Returns:
        聊天消息列表
    """
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


def call_judge_model(
    model: Any,
    chat: list[dict[str, Any]],
    temperature: float = 0.0
) -> Tuple[Optional[JudgementResult], Optional[str], float]:
    """调用判别模型
    
    Args:
        model: Outlines 包装的模型
        chat: 聊天消息
        temperature: 采样温度
        
    Returns:
        (解析后的判别结果, 原始响应, 延迟)
    """
    t0 = time.time()
    try:
        result = model(
            outlines.inputs.Chat(chat),
            JudgementResult,
            temperature=temperature
        )
        dt = time.time() - t0
        
        # 处理结果
        parsed: Optional[JudgementResult] = None
        if isinstance(result, str):
            raw = result
            try:
                if hasattr(JudgementResult, "model_validate_json"):
                    parsed = JudgementResult.model_validate_json(raw)
            except Exception as e:
                logging.debug(f"解析判别结果失败: {str(e)}")
                parsed = None
        else:
            raw = json.dumps(result, ensure_ascii=False)
            parsed = result
        
        return parsed, raw, dt
    except Exception as e:
        dt = time.time() - t0
        logging.error(f"调用判别模型失败: {str(e)}")
        return None, None, dt


def load_eval_results(csv_path: str) -> pd.DataFrame:
    """加载评估结果
    
    Args:
        csv_path: CSV 文件路径
        
    Returns:
        评估结果 DataFrame
    """
    df = pd.read_csv(csv_path)
    logging.info(f"📊 加载评估结果: {len(df)} 行")
    return df


def prepare_judgement_pairs(df: pd.DataFrame) -> list[dict[str, Any]]:
    """准备需要判别的样本对
    
    Args:
        df: 评估结果 DataFrame
        
    Returns:
        样本对列表，每个包含 query, gold_label, new_response, old_response
    """
    # 将数据按 idx 分组
    grouped = df.groupby("idx")
    
    pairs = []
    for idx, group in grouped:
        # 确保有 new 和 old 两个版本
        variants = group["variant"].unique()
        if len(variants) < 2:
            logging.warning(f"跳过 idx={idx}，缺少完整的变体数据")
            continue
        
        # 提取数据
        new_row = group[group["variant"] == "new"].iloc[0]
        old_row = group[group["variant"] == "old"].iloc[0]
        
        # 跳过两者都失败的情况
        if not new_row["ok"] and not old_row["ok"]:
            logging.debug(f"跳过 idx={idx}，两个候选都失败")
            continue
        
        pairs.append({
            "idx": int(idx),
            "query": str(new_row["query"]),
            "gold_label": str(new_row["gold_label"]) if pd.notna(new_row["gold_label"]) else "REFUSE",
            "new_response": str(new_row["raw_response"]) if pd.notna(new_row["raw_response"]) else "{}",
            "old_response": str(old_row["raw_response"]) if pd.notna(old_row["raw_response"]) else "{}",
            "new_ok": bool(new_row["ok"]),
            "old_ok": bool(old_row["ok"])
        })
    
    logging.info(f"🔍 准备判别样本对: {len(pairs)} 对")
    return pairs


def run_judgements(
    pairs: list[dict[str, Any]],
    model: Any,
    system_prompt: str,
    temperature: float = 0.0
) -> list[dict[str, Any]]:
    """运行 LLM 判别
    
    Args:
        pairs: 样本对列表
        model: 判别模型
        system_prompt: 系统提示
        temperature: 采样温度
        
    Returns:
        判别结果列表
    """
    results = []
    
    for pair in tqdm(pairs, desc="LLM 判别中"):
        # 构建聊天消息 (candidate_a = new, candidate_b = old)
        chat = build_judge_chat(
            system_prompt,
            pair["query"],
            pair["gold_label"],
            pair["new_response"],
            pair["old_response"]
        )
        
        # 调用判别模型
        parsed, raw, dt = call_judge_model(model, chat, temperature)
        
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
    
    return results


def analyze_judgement_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """分析判别结果
    
    Args:
        results: 判别结果列表
        
    Returns:
        分析统计
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
    
    # 综合得分 (加权)
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
    
    return {
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


def generate_report(
    analysis: dict[str, Any],
    results_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """生成判别报告
    
    Args:
        analysis: 分析统计
        results_df: 判别结果 DataFrame
        output_dir: 输出目录
    """
    # 保存详细结果
    results_path = output_dir / "llm_judgement_results.csv"
    results_df.to_csv(results_path, index=False, encoding='utf-8')
    logging.info(f"💾 详细判别结果已保存: {results_path}")
    
    # 保存统计分析
    analysis_path = output_dir / "llm_judgement_analysis.json"
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    logging.info(f"💾 统计分析已保存: {analysis_path}")
    
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
    md_path = output_dir / "llm_judgement_report.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines))
    logging.info(f"💾 Markdown 报告已保存: {md_path}")


def print_summary(analysis: dict[str, Any]) -> None:
    """打印摘要
    
    Args:
        analysis: 分析统计
    """
    print("\n" + "="*60)
    print("📊 LLM 判别摘要")
    print("="*60)
    print(f"总判别数: {analysis['total_judgements']}")
    print(f"NEW 胜出: {analysis['new_wins']} ({analysis['new_win_rate']*100:.1f}%)")
    print(f"OLD 胜出: {analysis['old_wins']} ({analysis['old_win_rate']*100:.1f}%)")
    print(f"平局: {analysis['ties']} ({analysis['tie_rate']*100:.1f}%)")
    print(f"平均置信度: {analysis['avg_confidence']:.2f}")
    print(f"\n综合得分:")
    print(f"  NEW: {analysis['new_overall_score']:.2f}")
    print(f"  OLD: {analysis['old_overall_score']:.2f}")
    print("="*60)


def main():
    """主程序入口"""
    parser = ArgumentParser(description="基于 LLM 的 Prompt 效果判别")
    parser.add_argument(
        "csv_path",
        help="评估结果 CSV 文件路径"
    )
    parser.add_argument(
        "--model",
        default="qwen3-max",
        help="判别模型名称（默认: qwen-max）"
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get(
            "QWEN_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        help="模型 API base URL"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="采样温度（默认: 0.0）"
    )
    parser.add_argument(
        "--outdir",
        help="输出目录（默认为 CSV 文件所在目录）"
    )
    parser.add_argument(
        "--judge-prompt",
        default=str(
            Path(__file__).resolve().parents[1]
            / "src"
            / "queryplan_eval"
            / "prompts"
            / "judge_system_prompt.j2"
        ),
        help="判别系统提示文件路径"
    )
    
    args = parser.parse_args()
    
    # 加载环境变量
    load_dotenv()
    api_key = os.environ.get("qwen_key") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("❌ 缺少 API key。请在 .env 中设置 qwen_key 或 OPENAI_API_KEY")
    
    # 检查文件
    if not Path(args.csv_path).exists():
        logging.error(f"❌ 文件不存在: {args.csv_path}")
        sys.exit(1)
    
    # 确定输出目录
    output_dir = Path(args.outdir) if args.outdir else Path(args.csv_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"🚀 开始 LLM 判别")
    logging.info(f"📁 数据文件: {args.csv_path}")
    logging.info(f"🤖 判别模型: {args.model}")
    logging.info(f"📂 输出目录: {output_dir}")
    
    try:
        # 初始化模型
        client = openai.OpenAI(base_url=args.base_url, api_key=api_key)
        model = outlines.from_openai(client, args.model)
        logging.info(f"✅ 模型初始化成功")
        
        # 加载判别提示
        judge_prompt = read_raw_prompt(args.judge_prompt)
        logging.info(f"✅ 判别提示加载成功")
        
        # 加载评估结果
        df = load_eval_results(args.csv_path)
        
        # 准备判别样本对
        pairs = prepare_judgement_pairs(df)
        
        if not pairs:
            logging.error("❌ 没有可判别的样本对")
            sys.exit(1)
        
        # 运行判别
        results = run_judgements(pairs, model, judge_prompt, args.temperature)
        
        # 分析结果
        results_df = pd.DataFrame(results)
        analysis = analyze_judgement_results(results)
        
        # 生成报告
        generate_report(analysis, results_df, output_dir)
        
        # 打印摘要
        print_summary(analysis)
        
        logging.info("\n✅ LLM 判别完成！")
        logging.info("📝 请查看生成的报告文件:")
        logging.info(f"   - {output_dir}/llm_judgement_results.csv")
        logging.info(f"   - {output_dir}/llm_judgement_analysis.json")
        logging.info(f"   - {output_dir}/llm_judgement_report.md")
        
    except Exception as e:
        logging.error(f"❌ 判别过程中出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

