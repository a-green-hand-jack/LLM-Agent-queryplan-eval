#!/usr/bin/env python3
"""Prompt 效果分析脚本

基于 docs/ai/02_analsys_prompt.md 中定义的分析方法，对比新旧 prompt 的效果。

使用方式:
    uv run python scripts/analyze_prompts.py outputs/v4/eval_results.csv
    uv run python scripts/analyze_prompts.py outputs/v4/eval_results.csv --outdir outputs/v4
"""

import sys
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import logging
from typing import Any
from difflib import SequenceMatcher

import pandas as pd
import numpy as np
from argparse import ArgumentParser

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ============================================================================
# 1. 结构化程度分析
# ============================================================================

def check_json_validity(raw_response: str) -> dict[str, Any]:
    """检查 raw_response 的 JSON 有效性
    
    Args:
        raw_response: 原始响应字符串
        
    Returns:
        包含有效性信息的字典
    """
    try:
        obj = json.loads(raw_response.strip())
        
        # 检查是否是拒答
        if isinstance(obj, dict) and obj.get("refuse"):
            return {
                "valid": True,
                "type": "refuse",
                "fields_complete": "reason" in obj or "refuse_reason" in obj
            }
        
        # 检查是否是计划对象（新格式）
        if isinstance(obj, dict) and "plans" in obj:
            plans = obj["plans"]
            if not plans:
                return {"valid": True, "type": "empty_plan", "completeness": 1.0}
            
            required_fields = {"domain", "sub", "is_personal", "time", "food"}
            complete_items = sum(
                1 for item in plans 
                if isinstance(item, dict) and required_fields.issubset(item.keys())
            )
            completeness = complete_items / len(plans)
            
            return {
                "valid": True,
                "type": "plans",
                "completeness": completeness,
                "n_plans": len(plans)
            }
        
        # 检查是否是计划数组（旧格式）
        if isinstance(obj, list):
            if not obj:
                return {"valid": True, "type": "empty_plan", "completeness": 1.0}
            
            required_fields = {"domain", "sub", "is_personal", "time", "food", "query"}
            complete_items = sum(
                1 for item in obj 
                if isinstance(item, dict) and required_fields.issubset(item.keys())
            )
            completeness = complete_items / len(obj)
            
            return {
                "valid": True,
                "type": "plans",
                "completeness": completeness,
                "n_plans": len(obj)
            }
        
        return {"valid": False, "reason": "unexpected_structure"}
    except json.JSONDecodeError as e:
        return {"valid": False, "reason": f"json_decode_error: {str(e)[:50]}"}
    except Exception as e:
        return {"valid": False, "reason": f"unexpected_error: {str(e)[:50]}"}


def analyze_structure(df: pd.DataFrame, variant: str) -> dict[str, Any]:
    """分析结构化程度
    
    Args:
        df: 评估结果 DataFrame
        variant: prompt 变体 ('new' 或 'old')
        
    Returns:
        结构化程度分析结果
    """
    dv = df[df["variant"] == variant]
    
    # 检查每个响应的 JSON 有效性
    validity_results = []
    for raw_resp in dv["raw_response"]:
        validity_results.append(check_json_validity(str(raw_resp)))
    
    # 统计指标
    valid_count = sum(1 for r in validity_results if r.get("valid", False))
    total_count = len(validity_results)
    
    # 字段完整性
    completeness_scores = [
        r.get("completeness", 0) 
        for r in validity_results 
        if r.get("valid", False) and r.get("type") == "plans"
    ]
    avg_completeness = np.mean(completeness_scores) if completeness_scores else 0
    
    return {
        "variant": variant,
        "json_validity": valid_count / total_count if total_count > 0 else 0,
        "avg_completeness": avg_completeness,
        "total_responses": total_count,
        "valid_responses": valid_count
    }


# ============================================================================
# 2. 金标签匹配度分析
# ============================================================================

def normalize_json(obj: Any) -> Any:
    """规范化 JSON 对象用于比较"""
    if isinstance(obj, dict):
        # 如果是新格式的包装对象，提取 plans
        if "plans" in obj:
            return normalize_json(obj["plans"])
        # 递归规范化字典
        return {k: normalize_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [normalize_json(item) for item in obj]
    else:
        return obj


def compare_with_gold(raw_response: str, gold_label: str) -> dict[str, Any]:
    """对比 raw_response 与 gold_label
    
    Args:
        raw_response: 原始响应
        gold_label: 金标签
        
    Returns:
        匹配度分析结果
    """
    try:
        # 解析预测结果
        try:
            pred = json.loads(raw_response.strip())
            pred = normalize_json(pred)
        except Exception:
            return {"exact_match": False, "key_field_match": False, "valid": False, "similarity": 0.0}
        
        # 解析金标签
        try:
            if str(gold_label).strip() == "REFUSE" or pd.isna(gold_label):
                gold = {"refuse": True}
            else:
                gold = json.loads(str(gold_label).strip())
                gold = normalize_json(gold)
        except Exception:
            return {"exact_match": False, "key_field_match": False, "valid": False, "similarity": 0.0}
        
        # 1. 完全匹配
        exact_match = (pred == gold)
        
        # 2. 关键字段匹配
        key_field_match = False
        if isinstance(pred, list) and isinstance(gold, list):
            if len(pred) == len(gold):
                key_field_match = all(
                    p.get("domain") == g.get("domain") and 
                    p.get("is_personal") == g.get("is_personal")
                    for p, g in zip(pred, gold)
                )
        elif isinstance(pred, dict) and isinstance(gold, dict):
            # 都是拒答
            if pred.get("refuse") and gold.get("refuse"):
                key_field_match = True
        
        # 3. 相似度
        similarity = SequenceMatcher(
            None,
            json.dumps(pred, ensure_ascii=False, sort_keys=True),
            json.dumps(gold, ensure_ascii=False, sort_keys=True)
        ).ratio()
        
        return {
            "exact_match": exact_match,
            "key_field_match": key_field_match,
            "similarity": similarity,
            "valid": True
        }
    except Exception as e:
        logging.debug(f"比较失败: {str(e)[:50]}")
        return {
            "exact_match": False,
            "key_field_match": False,
            "similarity": 0.0,
            "valid": False,
            "error": str(e)[:50]
        }


def analyze_gold_match(df: pd.DataFrame, variant: str) -> dict[str, Any]:
    """分析金标签匹配度
    
    Args:
        df: 评估结果 DataFrame
        variant: prompt 变体
        
    Returns:
        匹配度分析结果
    """
    dv = df[df["variant"] == variant]
    
    # 对每行进行对比
    match_results = []
    for _, row in dv.iterrows():
        result = compare_with_gold(str(row["raw_response"]), str(row["gold_label"]))
        match_results.append(result)
    
    # 统计指标
    valid_comparisons = [r for r in match_results if r.get("valid", False)]
    exact_matches = sum(1 for r in valid_comparisons if r.get("exact_match", False))
    key_field_matches = sum(1 for r in valid_comparisons if r.get("key_field_match", False))
    similarities = [r.get("similarity", 0) for r in valid_comparisons]
    
    return {
        "variant": variant,
        "exact_match_rate": exact_matches / len(valid_comparisons) if valid_comparisons else 0,
        "key_field_match_rate": key_field_matches / len(valid_comparisons) if valid_comparisons else 0,
        "avg_similarity": np.mean(similarities) if similarities else 0,
        "total_comparisons": len(dv),
        "valid_comparisons": len(valid_comparisons)
    }


# ============================================================================
# 3. 多样性与覆盖率分析
# ============================================================================

def analyze_diversity(df: pd.DataFrame, variant: str) -> dict[str, Any]:
    """分析输出多样性
    
    Args:
        df: 评估结果 DataFrame
        variant: prompt 变体
        
    Returns:
        多样性分析结果
    """
    dv = df[df["variant"] == variant]
    
    # 总查询数
    n_queries = dv["idx"].nunique()
    
    # 不同的 raw_response 种类
    unique_raw = dv["raw_response"].nunique()
    diversity_rate = unique_raw / len(dv) if len(dv) > 0 else 0
    
    # 拒答比例
    refuse_count = (dv["type"] == "refuse").sum()
    refuse_rate = refuse_count / len(dv) if len(dv) > 0 else 0
    
    # 计划平均长度（仅统计非拒答的）
    plans_dv = dv[dv["type"] == "plans"]
    avg_plan_length = (
        plans_dv["n_plans"].mean() 
        if len(plans_dv) > 0 else 0
    )
    
    return {
        "variant": variant,
        "n_queries": n_queries,
        "unique_raw_responses": unique_raw,
        "diversity_rate": diversity_rate,
        "refuse_rate": refuse_rate,
        "avg_plan_length": avg_plan_length
    }


# ============================================================================
# 4. 鲁棒性分析
# ============================================================================

def analyze_robustness(df: pd.DataFrame, variant: str, 
                       valid_domains: set[str] | None = None) -> dict[str, Any]:
    """分析鲁棒性
    
    Args:
        df: 评估结果 DataFrame
        variant: prompt 变体
        valid_domains: 有效的 domain 集合
        
    Returns:
        鲁棒性分析结果
    """
    dv = df[df["variant"] == variant]
    
    # 1. 失败率
    failure_mask = dv["ok"] == False
    failure_rate = failure_mask.sum() / len(dv) if len(dv) > 0 else 0
    
    # 2. 幻觉率（输出不存在的 domain）
    if valid_domains is None:
        valid_domains = {
            "体温", "减脂", "心脏健康", "情绪健康", "生理健康",
            "血压", "血氧饱和度", "血糖", "睡眠", "午睡", "步数",
            "活力三环", "微体检", "饮食", "跑步", "骑行", "步行徒步",
            "游泳", "登山", "跳绳", "瑜伽", "普拉提", "划船机", "其他"
        }
    
    hallucination_count = 0
    total_plans = 0
    for _, row in dv.iterrows():
        if row["type"] == "plans" and pd.notna(row["parsed"]):
            try:
                parsed = json.loads(str(row["parsed"]))
                # 处理新格式
                if isinstance(parsed, dict) and "plans" in parsed:
                    parsed = parsed["plans"]
                if isinstance(parsed, list):
                    for plan in parsed:
                        total_plans += 1
                        if plan.get("domain") not in valid_domains:
                            hallucination_count += 1
            except Exception:
                pass
    
    hallucination_rate = hallucination_count / total_plans if total_plans > 0 else 0
    
    # 3. 超长输出（> 500 字符）
    long_output_count = (dv["raw_response"].str.len() > 500).sum()
    long_output_rate = long_output_count / len(dv) if len(dv) > 0 else 0
    
    return {
        "variant": variant,
        "failure_rate": failure_rate,
        "hallucination_rate": hallucination_rate,
        "long_output_rate": long_output_rate,
        "total_plans_checked": total_plans
    }


# ============================================================================
# 5. 性能分析
# ============================================================================

def analyze_performance(df: pd.DataFrame, variant: str) -> dict[str, Any]:
    """分析性能指标
    
    Args:
        df: 评估结果 DataFrame
        variant: prompt 变体
        
    Returns:
        性能分析结果
    """
    dv = df[df["variant"] == variant]
    valid_latency = dv["latency_sec"].dropna()
    
    return {
        "variant": variant,
        "mean_latency": valid_latency.mean() if len(valid_latency) > 0 else None,
        "median_latency": valid_latency.median() if len(valid_latency) > 0 else None,
        "p95_latency": valid_latency.quantile(0.95) if len(valid_latency) > 0 else None,
        "p99_latency": valid_latency.quantile(0.99) if len(valid_latency) > 0 else None,
        "timeout_rate": (dv["latency_sec"] > 30).sum() / len(dv) if len(dv) > 0 else 0,
        "n_samples": len(valid_latency)
    }


# ============================================================================
# 6. 综合评分
# ============================================================================

def calculate_quality_score(metrics: dict[str, Any]) -> float:
    """计算综合质量评分 (0-100)
    
    Args:
        metrics: 包含各项指标的字典
        
    Returns:
        质量评分
    """
    # 权重分配
    weights = {
        "json_validity": 0.25,
        "exact_match_rate": 0.25,
        "key_field_match_rate": 0.15,
        "failure_rate": 0.15,
        "hallucination_rate": 0.10,
        "diversity_rate": 0.05,
        "refuse_rate_accuracy": 0.05
    }
    
    score = 0.0
    score += metrics.get("json_validity", 0) * weights["json_validity"] * 100
    score += metrics.get("exact_match_rate", 0) * weights["exact_match_rate"] * 100
    score += metrics.get("key_field_match_rate", 0) * weights["key_field_match_rate"] * 100
    score += (1 - metrics.get("failure_rate", 0)) * weights["failure_rate"] * 100
    score += (1 - metrics.get("hallucination_rate", 0)) * weights["hallucination_rate"] * 100
    score += min(metrics.get("diversity_rate", 0), 0.3) / 0.3 * weights["diversity_rate"] * 100
    
    # 拒答率准确度（接近 8.4% 为最优）
    target_refuse_rate = 0.084
    actual_refuse_rate = metrics.get("refuse_rate", 0)
    refuse_accuracy = 1 - abs(actual_refuse_rate - target_refuse_rate) / target_refuse_rate
    refuse_accuracy = max(0, refuse_accuracy)
    score += refuse_accuracy * weights["refuse_rate_accuracy"] * 100
    
    return score


# ============================================================================
# 7. 完整分析流程
# ============================================================================

def full_analysis(csv_path: str, output_dir: str | None = None) -> dict[str, Any]:
    """完整的分析流程
    
    Args:
        csv_path: 评估结果 CSV 文件路径
        output_dir: 输出目录（可选）
        
    Returns:
        完整的分析结果
    """
    logging.info(f"📊 开始分析 prompt 效果")
    logging.info(f"📁 数据文件: {csv_path}")
    
    # 加载数据
    df = pd.read_csv(csv_path)
    logging.info(f"📈 数据集大小: {len(df)} 行")
    
    # 确定输出目录
    if output_dir is None:
        output_dir = str(Path(csv_path).parent)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 分析结果容器
    results = {}
    
    # 对每个 variant 进行分析
    variants = df["variant"].unique()
    logging.info(f"🔍 发现 {len(variants)} 个变体: {', '.join(variants)}")
    
    for variant in variants:
        logging.info(f"\n{'='*60}")
        logging.info(f"📊 分析 {variant.upper()} Prompt")
        logging.info(f"{'='*60}")
        
        # 1. 结构化程度
        structure_metrics = analyze_structure(df, variant)
        logging.info(f"✅ 结构化分析完成")
        
        # 2. 金标签匹配度
        gold_metrics = analyze_gold_match(df, variant)
        logging.info(f"✅ 金标签匹配分析完成")
        
        # 3. 多样性
        diversity_metrics = analyze_diversity(df, variant)
        logging.info(f"✅ 多样性分析完成")
        
        # 4. 鲁棒性
        robustness_metrics = analyze_robustness(df, variant)
        logging.info(f"✅ 鲁棒性分析完成")
        
        # 5. 性能
        performance_metrics = analyze_performance(df, variant)
        logging.info(f"✅ 性能分析完成")
        
        # 合并所有指标
        all_metrics = {
            **structure_metrics,
            **gold_metrics,
            **diversity_metrics,
            **robustness_metrics,
            **performance_metrics
        }
        
        # 计算综合评分
        quality_score = calculate_quality_score(all_metrics)
        all_metrics["quality_score"] = quality_score
        
        results[variant] = all_metrics
        
        logging.info(f"🎯 综合质量评分: {quality_score:.2f}")
    
    # 生成对比报告
    logging.info(f"\n{'='*60}")
    logging.info(f"📊 生成对比报告")
    logging.info(f"{'='*60}")
    
    # 保存 JSON 报告
    json_path = output_path / "analysis_report.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info(f"💾 JSON 报告已保存: {json_path}")
    
    # 生成 Markdown 报告
    md_report = generate_markdown_report(results)
    md_path = output_path / "analysis_report.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    logging.info(f"💾 Markdown 报告已保存: {md_path}")
    
    # 打印简要对比
    print_comparison_summary(results)
    
    return results


def generate_markdown_report(results: dict[str, Any]) -> str:
    """生成 Markdown 格式的对比报告
    
    Args:
        results: 分析结果
        
    Returns:
        Markdown 格式的报告
    """
    lines = []
    lines.append("# Prompt 效果分析报告")
    lines.append("")
    lines.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # 质量评分对比
    lines.append("## 📊 综合质量评分")
    lines.append("")
    for variant, metrics in results.items():
        score = metrics.get("quality_score", 0)
        emoji = "✅" if score >= 80 else "⚠️" if score >= 70 else "❌"
        lines.append(f"- **{variant.upper()} Prompt**: {score:.2f} 分 {emoji}")
    lines.append("")
    
    # 详细指标对比
    lines.append("## 📈 详细指标对比")
    lines.append("")
    lines.append("| 指标 | " + " | ".join([v.upper() for v in results.keys()]) + " | 最优 |")
    lines.append("|------|" + "------|" * len(results) + "------|")
    
    # 定义要对比的指标
    metrics_to_compare = [
        ("JSON 有效性", "json_validity", "percent", "higher"),
        ("字段完整性", "avg_completeness", "percent", "higher"),
        ("精确匹配率", "exact_match_rate", "percent", "higher"),
        ("关键字段匹配率", "key_field_match_rate", "percent", "higher"),
        ("平均相似度", "avg_similarity", "percent", "higher"),
        ("失败率", "failure_rate", "percent", "lower"),
        ("幻觉率", "hallucination_rate", "percent", "lower"),
        ("超长输出率", "long_output_rate", "percent", "lower"),
        ("拒答率", "refuse_rate", "percent", "target_8.4"),
        ("多样性率", "diversity_rate", "percent", "moderate"),
        ("平均计划长度", "avg_plan_length", "number", "neutral"),
        ("平均延迟(秒)", "mean_latency", "number", "lower"),
        ("P95 延迟(秒)", "p95_latency", "number", "lower"),
    ]
    
    for label, key, fmt, direction in metrics_to_compare:
        row = [label]
        values = []
        for variant in results.keys():
            val = results[variant].get(key)
            if val is None:
                row.append("N/A")
                values.append(None)
            elif fmt == "percent":
                row.append(f"{val*100:.1f}%")
                values.append(val)
            else:
                row.append(f"{val:.2f}")
                values.append(val)
        
        # 确定最优
        valid_values = [(i, v) for i, v in enumerate(values) if v is not None]
        if valid_values:
            if direction == "higher":
                best_idx = max(valid_values, key=lambda x: x[1])[0]
            elif direction == "lower":
                best_idx = min(valid_values, key=lambda x: x[1])[0]
            elif direction == "target_8.4":
                best_idx = min(valid_values, key=lambda x: abs(x[1] - 0.084))[0]
            else:
                best_idx = -1
            
            if best_idx >= 0:
                best_variant = list(results.keys())[best_idx]
                row.append(f"✅ {best_variant.upper()}")
            else:
                row.append("➖")
        else:
            row.append("N/A")
        
        lines.append("| " + " | ".join(row) + " |")
    
    lines.append("")
    
    # 性能统计
    lines.append("## ⚡ 性能统计")
    lines.append("")
    for variant, metrics in results.items():
        lines.append(f"### {variant.upper()} Prompt")
        lines.append("")
        lines.append(f"- 平均延迟: {metrics.get('mean_latency', 0):.2f}s")
        lines.append(f"- 中位数延迟: {metrics.get('median_latency', 0):.2f}s")
        lines.append(f"- P95 延迟: {metrics.get('p95_latency', 0):.2f}s")
        lines.append(f"- P99 延迟: {metrics.get('p99_latency', 0):.2f}s")
        lines.append(f"- 超时率: {metrics.get('timeout_rate', 0)*100:.1f}%")
        lines.append("")
    
    # 结论
    lines.append("## 🎯 结论")
    lines.append("")
    
    # 找出得分最高的
    best_variant = max(results.items(), key=lambda x: x[1].get("quality_score", 0))
    lines.append(f"**最优 Prompt**: {best_variant[0].upper()}")
    lines.append(f"**综合评分**: {best_variant[1].get('quality_score', 0):.2f}")
    lines.append("")
    
    # 关键优势
    lines.append("### 关键优势")
    lines.append("")
    best_metrics = best_variant[1]
    if best_metrics.get("json_validity", 0) > 0.95:
        lines.append("- ✅ JSON 有效性优秀 (>95%)")
    if best_metrics.get("exact_match_rate", 0) > 0.70:
        lines.append("- ✅ 精确匹配率优秀 (>70%)")
    if best_metrics.get("failure_rate", 0) < 0.05:
        lines.append("- ✅ 失败率低 (<5%)")
    if best_metrics.get("hallucination_rate", 0) < 0.01:
        lines.append("- ✅ 几乎无幻觉输出")
    
    lines.append("")
    
    return "\n".join(lines)


def print_comparison_summary(results: dict[str, Any]) -> None:
    """打印简要对比摘要
    
    Args:
        results: 分析结果
    """
    print("\n" + "="*60)
    print("📊 Prompt 效果对比摘要")
    print("="*60)
    
    for variant, metrics in results.items():
        print(f"\n🔷 {variant.upper()} Prompt:")
        print(f"   综合评分: {metrics.get('quality_score', 0):.2f} 分")
        print(f"   JSON有效性: {metrics.get('json_validity', 0)*100:.1f}%")
        print(f"   精确匹配率: {metrics.get('exact_match_rate', 0)*100:.1f}%")
        print(f"   失败率: {metrics.get('failure_rate', 0)*100:.1f}%")
        print(f"   平均延迟: {metrics.get('mean_latency', 0):.2f}s")
    
    print("\n" + "="*60)


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序入口"""
    parser = ArgumentParser(description="Prompt 效果分析脚本")
    parser.add_argument(
        "csv_path",
        help="评估结果 CSV 文件路径"
    )
    parser.add_argument(
        "--outdir",
        help="输出目录（默认为 CSV 文件所在目录）",
        default=None
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.csv_path).exists():
        logging.error(f"❌ 文件不存在: {args.csv_path}")
        sys.exit(1)
    
    try:
        # 运行完整分析
        results = full_analysis(args.csv_path, args.outdir)
        
        logging.info("\n✅ 分析完成！")
        logging.info("📝 请查看生成的报告文件:")
        output_dir = args.outdir or str(Path(args.csv_path).parent)
        logging.info(f"   - {output_dir}/analysis_report.json")
        logging.info(f"   - {output_dir}/analysis_report.md")
        
    except Exception as e:
        logging.error(f"❌ 分析过程中出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

