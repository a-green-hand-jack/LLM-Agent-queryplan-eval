"""测试 UnifiedPromptManager 的功能

该脚本测试所有四个任务的 prompt 渲染功能。

Usage:
    uv run python tests/test_unified_prompt_manager.py
"""

import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from queryplan_eval.core.unified_prompt_manager import UnifiedPromptManager


def print_section(title: str):
    """打印章节标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_messages(messages: list):
    """打印消息列表"""
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]
        # 截断过长的内容
        if len(content) > 500:
            content = content[:500] + "\n... (截断)"
        print(f"\n[{role}]")
        print(content)


def test_query_plan(manager: UnifiedPromptManager):
    """测试 Query Plan 任务"""
    print_section("测试 Query Plan 任务")

    try:
        # 测试 v6_cot 版本
        today = datetime.now().strftime("%Y年%m月%d日")
        messages = manager.render_query_plan(
            version="v6_cot",
            today=today
        )
        print(f"\n✓ 成功渲染 Query Plan (v6_cot) prompt")
        print(f"  消息数量: {len(messages)}")
        print_messages(messages)

        # 获取 LLM 配置
        llm_config = manager.get_query_plan_llm_config()
        print(f"\n✓ LLM 配置: {llm_config}")

        # 列出所有版本
        versions = manager.list_versions("query_plan", "system")
        print(f"\n✓ 可用版本: {versions}")

        return True
    except Exception as e:
        print(f"\n✗ Query Plan 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ragtruth(manager: UnifiedPromptManager):
    """测试 RAGTruth 任务"""
    print_section("测试 RAGTruth 任务")

    try:
        # 测试 Summary 任务 with CoT
        messages = manager.render_ragtruth(
            task_type="Summary",
            use_cot=True,
            reference="这是一篇关于人工智能的文章，讨论了深度学习的最新进展...",
            response="文章摘要：深度学习在近年来取得了显著进展..."
        )
        print(f"\n✓ 成功渲染 RAGTruth Summary (with CoT) prompt")
        print(f"  消息数量: {len(messages)}")
        print_messages(messages)

        # 测试 QA 任务 without CoT
        messages = manager.render_ragtruth(
            task_type="QA",
            use_cot=False,
            question="什么是深度学习？",
            reference="深度学习是机器学习的一个子领域，使用神经网络...",
            response="深度学习是一种使用多层神经网络的机器学习方法..."
        )
        print(f"\n✓ 成功渲染 RAGTruth QA (without CoT) prompt")
        print(f"  消息数量: {len(messages)}")

        # 获取 LLM 配置
        llm_config = manager.get_ragtruth_llm_config()
        print(f"\n✓ LLM 配置: {llm_config}")

        return True
    except Exception as e:
        print(f"\n✗ RAGTruth 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_patent(manager: UnifiedPromptManager):
    """测试 Patent 任务"""
    print_section("测试 Patent 任务")

    try:
        # 测试 v2 版本
        messages = manager.render_patent(
            version="v2",
            sequence="Ala Ser Lys Gly",
            features=[
                {
                    "name_key": "MOD_RES",
                    "location": "(3).. (3)",
                    "other_information": "D-Lys"
                }
            ]
        )
        print(f"\n✓ 成功渲染 Patent (v2) prompt")
        print(f"  消息数量: {len(messages)}")
        print_messages(messages)

        # 获取 LLM 配置
        llm_config = manager.get_patent_llm_config()
        print(f"\n✓ LLM 配置: {llm_config}")

        return True
    except Exception as e:
        print(f"\n✗ Patent 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_judgement(manager: UnifiedPromptManager):
    """测试 Judgement 任务"""
    print_section("测试 Judgement 任务")

    try:
        # 测试 v1 版本
        messages = manager.render_judgement(
            version="v1",
            question="今天我跑了5公里",
            gold_standard={
                "plans": [
                    {
                        "domain": "运动",
                        "sub": "跑步",
                        "is_personal": True,
                        "time": "2025-10-30",
                        "food": None
                    }
                ],
                "refused": False,
                "refuse_reason": ""
            },
            candidate_a={
                "plans": [
                    {
                        "domain": "运动",
                        "sub": "跑步",
                        "is_personal": True,
                        "time": "2025-10-30",
                        "food": None
                    }
                ],
                "refused": False,
                "refuse_reason": ""
            },
            candidate_b={
                "plans": [
                    {
                        "domain": "健康",
                        "sub": None,
                        "is_personal": True,
                        "time": None,
                        "food": None
                    }
                ],
                "refused": False,
                "refuse_reason": ""
            }
        )
        print(f"\n✓ 成功渲染 Judgement (v1) prompt")
        print(f"  消息数量: {len(messages)}")
        print_messages(messages)

        # 获取 LLM 配置
        llm_config = manager.get_judgement_llm_config()
        print(f"\n✓ LLM 配置: {llm_config}")

        return True
    except Exception as e:
        print(f"\n✗ Judgement 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_general_features(manager: UnifiedPromptManager):
    """测试通用功能"""
    print_section("测试通用功能")

    try:
        # 列出所有 prompts
        prompts = manager.list_prompts()
        print(f"\n✓ 可用的 prompts: {prompts}")

        # 获取缓存统计
        cache_stats = manager.cache_stats()
        print(f"\n✓ 缓存统计: {cache_stats}")

        # 测试清除缓存
        manager.clear_cache()
        print("\n✓ 缓存已清除")

        return True
    except Exception as e:
        print(f"\n✗ 通用功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print_section("UnifiedPromptManager 功能测试")

    try:
        # 初始化 manager
        print("\n正在初始化 UnifiedPromptManager...")
        manager = UnifiedPromptManager(dev_mode=True, enable_cache=True)
        print("✓ UnifiedPromptManager 初始化成功")

        # 运行所有测试
        results = {
            "Query Plan": test_query_plan(manager),
            "RAGTruth": test_ragtruth(manager),
            "Patent": test_patent(manager),
            "Judgement": test_judgement(manager),
            "General Features": test_general_features(manager),
        }

        # 打印测试结果总结
        print_section("测试结果总结")
        passed = sum(results.values())
        total = len(results)

        for task, result in results.items():
            status = "✓ 通过" if result else "✗ 失败"
            print(f"  {task:20s}: {status}")

        print(f"\n总计: {passed}/{total} 个测试通过")

        if passed == total:
            print("\n🎉 所有测试通过！")
            return 0
        else:
            print(f"\n⚠️  有 {total - passed} 个测试失败")
            return 1

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
