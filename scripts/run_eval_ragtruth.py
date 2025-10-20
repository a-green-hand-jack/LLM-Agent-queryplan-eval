#!/usr/bin/env python3
"""RAGTruth 幻觉检测评估脚本

用于评估 LLM 的幻觉检测能力，支持多任务类型（Summary/QA/Data2txt）。

使用方式:
    # 评估 3 个任务类型，采样 50 个样本
    uv run python scripts/run_eval_ragtruth.py

    # 只评估 Summary 任务
    uv run python scripts/run_eval_ragtruth.py --task-types Summary

    # 评估 Summary 和 QA，采样 100 个样本
    uv run python scripts/run_eval_ragtruth.py --task-types Summary QA -n 100

    # 使用 CoT 推理
    uv run python scripts/run_eval_ragtruth.py --enable-cot

    # 自定义输出目录
    uv run python scripts/run_eval_ragtruth.py --outdir outputs/ragtruth_custom
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from argparse import ArgumentParser
from dotenv import load_dotenv

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from queryplan_eval.llms import OpenAILLM
from queryplan_eval.tasks import RAGTruthTask

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主程序入口"""
    parser = ArgumentParser(description="RAGTruth 幻觉检测评估")
    
    parser.add_argument(
        "--task-types",
        nargs="+",
        default=["Summary", "QA", "Data2txt"],
        help="任务类型列表，默认: Summary QA Data2txt"
    )
    
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test"],
        help="数据分割，默认: test"
    )
    
    parser.add_argument(
        "-n",
        "--sample",
        type=int,
        default=50,
        help="采样样本数，默认: 50"
    )
    
    parser.add_argument(
        "--model",
        default="qwen-flash",
        help="模型名称，默认: qwen-flash"
    )
    
    parser.add_argument(
        "--base-url",
        default=None,
        help="API 基础 URL"
    )
    
    parser.add_argument(
        "--enable-cot",
        action="store_true",
        help="启用 Chain-of-Thought 推理"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="采样温度，默认: 0.0"
    )
    
    parser.add_argument(
        "--outdir",
        default="outputs/ragtruth_latest",
        help="输出目录，默认: outputs/ragtruth_latest"
    )
    
    args = parser.parse_args()
    
    # 验证任务类型
    valid_tasks = ["Summary", "QA", "Data2txt"]
    for task_type in args.task_types:
        if task_type not in valid_tasks:
            logger.error(f"无效的任务类型: {task_type}，必须是 {valid_tasks} 之一")
            sys.exit(1)
    
    # 加载环境变量
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
    api_key = os.environ.get("qwen_key") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("缺少 API 密钥。请在 .env 中设置 qwen_key 或 OPENAI_API_KEY")
        sys.exit(1)
    
    # 确定 base_url
    if args.base_url is None:
        args.base_url = os.environ.get(
            "QWEN_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    
    logger.info("=" * 70)
    logger.info("RAGTruth 幻觉检测评估")
    logger.info("=" * 70)
    logger.info(f"任务类型: {', '.join(args.task_types)}")
    logger.info(f"数据分割: {args.split}")
    logger.info(f"采样样本数: {args.sample}")
    logger.info(f"模型: {args.model}")
    logger.info(f"启用 CoT: {args.enable_cot}")
    logger.info(f"采样温度: {args.temperature}")
    logger.info(f"输出目录: {args.outdir}")
    logger.info("=" * 70)
    
    try:
        # 初始化 LLM
        logger.info("初始化 LLM...")
        llm = OpenAILLM(
            model_name=args.model,
            base_url=args.base_url,
            api_key=api_key
        )
        logger.info("✓ LLM 初始化完成")
        
        # 初始化任务
        logger.info("初始化 RAGTruthTask...")
        task = RAGTruthTask(
            task_types=args.task_types,
            split=args.split,
            sample_n=args.sample,
            use_cot=args.enable_cot,
            llm=llm,
            output_dir=args.outdir
        )
        logger.info(f"✓ 任务初始化完成，数据集大小: {len(task.dataset)}")
        
        # 运行评估
        logger.info("开始评估...")
        metrics = task.run_evaluation(temperature=args.temperature)
        logger.info("✓ 评估完成")
        
        # 打印摘要
        _print_summary(metrics, args.outdir)
        
    except Exception as e:
        logger.error(f"评估过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


def _print_summary(metrics: dict, outdir: str) -> None:
    """打印评估摘要
    
    Args:
        metrics: 评估指标字典
        outdir: 输出目录
    """
    print("\n" + "=" * 70)
    print("评估结果摘要")
    print("=" * 70)
    
    # 总体结果
    print("\n总体结果:")
    print(f"  总样本数: {metrics['total']}")
    print(f"  成功样本: {metrics['ok']}")
    print(f"  成功率: {metrics['ok_rate']:.1%}")
    
    # 性能统计
    if metrics.get("latency_mean") is not None:
        print("\n性能统计:")
        print(f"  平均延迟: {metrics['latency_mean']:.2f}s")
        if metrics.get("latency_p95") is not None:
            print(f"  P95延迟: {metrics['latency_p95']:.2f}s")
    
    # 分任务统计
    by_task = metrics.get("by_task", {})
    if by_task:
        print("\n分任务统计:")
        for task_type, task_metrics in by_task.items():
            total = task_metrics.get("total", 0)
            ok_rate = task_metrics.get("ok_rate", 0)
            print(f"\n  {task_type} (n={total}):")
            print(f"    成功率: {ok_rate:.1%}")
    
    print("\n" + "=" * 70)
    print(f"详细结果已保存到: {outdir}")
    print("  - results.csv: 详细结果")
    print("  - metrics.json: 指标数据")
    print("  - summary.txt: 文本摘要")
    print("  - report_*.txt: 分任务报告")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
