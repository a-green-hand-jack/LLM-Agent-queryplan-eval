#!/usr/bin/env python3
"""
CoT 功能基础测试脚本

快速验证 CoT prompt 和 schema 是否正常工作
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from queryplan_eval.schemas import QueryResult, ReasoningSteps
from queryplan_eval.renderer import render_system_prompt
import json


def test_schema():
    """测试 Schema 定义"""
    print("=" * 60)
    print("测试 1: Schema 定义")
    print("=" * 60)
    
    # 测试创建带 reasoning 的 QueryResult
    reasoning = ReasoningSteps(
        query_analysis="测试问题分析",
        domain_identification="测试领域识别",
        time_calculation="测试时间计算",
        refuse_check="测试拒答检查",
        final_decision="测试最终决策"
    )
    
    result = QueryResult(
        reasoning=reasoning,
        plans=[],
        refused=False,
        refuse_reason=""
    )
    
    # 转换为字典
    result_dict = result.model_dump()
    print("✅ QueryResult with reasoning 创建成功")
    print(json.dumps(result_dict, ensure_ascii=False, indent=2))
    print()
    
    # 测试从 JSON 解析
    json_str = json.dumps(result_dict, ensure_ascii=False)
    parsed = QueryResult.model_validate_json(json_str)
    print("✅ 从 JSON 解析成功")
    print(f"   reasoning.query_analysis: {parsed.reasoning.query_analysis if parsed.reasoning else 'None'}")
    print()


def test_prompt_rendering():
    """测试 CoT Prompt 渲染"""
    print("=" * 60)
    print("测试 2: CoT Prompt 渲染")
    print("=" * 60)
    
    cot_prompt_path = str(
        Path(__file__).resolve().parents[1]
        / "src"
        / "queryplan_eval"
        / "prompts"
        / "queryplan_system_prompt_v6_cot.j2"
    )
    
    today = "2025年10月19日"
    rendered = render_system_prompt(cot_prompt_path, today=today)
    
    print("✅ CoT Prompt 渲染成功")
    print(f"   Prompt 长度: {len(rendered)} 字符")
    print(f"   包含 'reasoning': {'reasoning' in rendered}")
    print(f"   包含 'query_analysis': {'query_analysis' in rendered}")
    print(f"   包含 'Chain-of-Thought': {'Chain-of-Thought' in rendered}")
    print()
    
    # 显示部分内容
    print("   前 500 字符:")
    print("   " + "-" * 50)
    print("   " + rendered[:500].replace("\n", "\n   "))
    print()


def test_json_example():
    """测试 Prompt 中的 JSON 示例是否有效"""
    print("=" * 60)
    print("测试 3: JSON 示例有效性")
    print("=" * 60)
    
    # 模拟一个 Prompt 中的 JSON 示例
    example_json = """
    {
      "reasoning": {
        "query_analysis": "用户询问今天的睡眠情况",
        "domain_identification": "属于健康领域-睡眠场景",
        "time_calculation": "今天是2025年10月19日，查询今天的睡眠",
        "refuse_check": "不属于拒答情形，是正常的个人数据查询",
        "final_decision": "提取睡眠领域的个人查询计划"
      },
      "plans": [
        {
          "domain": "睡眠",
          "sub": null,
          "is_personal": true,
          "time": "2025年10月19日",
          "food": null
        }
      ],
      "refused": false,
      "refuse_reason": ""
    }
    """
    
    try:
        result = QueryResult.model_validate_json(example_json)
        print("✅ JSON 示例验证成功")
        print(f"   Plans 数量: {len(result.plans)}")
        print(f"   Reasoning 存在: {result.reasoning is not None}")
        if result.reasoning:
            print(f"   Query Analysis: {result.reasoning.query_analysis}")
        print()
    except Exception as e:
        print(f"❌ JSON 示例验证失败: {e}")
        print()


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("CoT 功能基础测试")
    print("=" * 60 + "\n")
    
    try:
        test_schema()
        test_prompt_rendering()
        test_json_example()
        
        print("=" * 60)
        print("✅ 所有基础测试通过！")
        print("=" * 60)
        print()
        print("下一步：运行实际评估测试")
        print("  python scripts/run_eval.py --data data/summary_train_v3.xlsx --n 5 --enable-cot --outdir outputs/cot_test")
        print()
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ 测试失败: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

