#!/usr/bin/env python3
"""基于 LLM 的成对比较判别脚本 - 重构版本

使用方式:
    # 比较两个 CSV 文件
    uv run python scripts/llm_judge.py outputs/original/eval_results.csv outputs/test/eval_results.csv
    
    # 指定输出目录
    uv run python scripts/llm_judge.py outputs/original/eval_results.csv outputs/test/eval_results.csv --outdir outputs/comparison
    
    # 指定判别模型
    uv run python scripts/llm_judge.py outputs/original/eval_results.csv outputs/test/eval_results.csv --model qwen3-max
"""

import sys
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import os
import logging
from argparse import ArgumentParser
from dotenv import load_dotenv

from queryplan_eval.llms import OpenAILLM
from queryplan_eval.core.prompt_manager import PromptManager
from queryplan_eval.tools import PairwiseJudge

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主程序入口"""
    parser = ArgumentParser(description="基于 LLM 的成对比较判别")
    parser.add_argument(
        "csv_path_a",
        help="方案 A 的评估结果 CSV 文件路径"
    )
    parser.add_argument(
        "csv_path_b",
        help="方案 B 的评估结果 CSV 文件路径"
    )
    parser.add_argument(
        "--model",
        default="qwen3-max",
        help="判别模型名称（默认: qwen3-max）"
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="API 基础 URL"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="采样温度（默认: 0.0）"
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="输出目录（默认为方案 A 的目录）"
    )
    
    args = parser.parse_args()
    
    # 加载环境变量
    load_dotenv()
    api_key = os.environ.get("qwen_key") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("缺少 API 密钥。请在 .env 中设置 qwen_key 或 OPENAI_API_KEY")
        sys.exit(1)
    
    # 检查文件
    for csv_path in [args.csv_path_a, args.csv_path_b]:
        if not Path(csv_path).exists():
            logger.error(f"文件不存在: {csv_path}")
        sys.exit(1)
    
    # 确定输出目录
    if args.outdir is None:
        output_dir = str(Path(args.csv_path_a).parent / "comparison")
    else:
        output_dir = args.outdir
    
    # 确定 base_url
    if args.base_url is None:
        args.base_url = os.environ.get(
            "QWEN_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    
    logger.info("开始 LLM 成对比较判别")
    logger.info(f"方案 A 文件: {args.csv_path_a}")
    logger.info(f"方案 B 文件: {args.csv_path_b}")
    logger.info(f"判别模型: {args.model}")
    logger.info(f"输出目录: {output_dir}")
    
    try:
        # 初始化 LLM
        llm = OpenAILLM(
            model_name=args.model,
            base_url=args.base_url,
            api_key=api_key
        )
        
        # 初始化 Prompt Manager
        prompt_manager = PromptManager(
            task_name="judgement",
            version="latest"
        )
        
        # 初始化判别工具
        judge = PairwiseJudge(
            csv_path_a=args.csv_path_a,
            csv_path_b=args.csv_path_b,
            llm=llm,
            prompt_manager=prompt_manager,
            output_dir=output_dir
        )
        
        # 运行判别
        analysis = judge.run_judgement(temperature=args.temperature)
        
        # 打印摘要
        print("\n" + "="*60)
        print("LLM 成对比较判别完成")
        print("="*60)
        print(f"总判别数: {analysis['total_judgements']}")
        print(f"方案A 胜出: {analysis['a_wins']} ({analysis['a_win_rate']*100:.1f}%)")
        print(f"方案B 胜出: {analysis['b_wins']} ({analysis['b_win_rate']*100:.1f}%)")
        print(f"平局: {analysis['ties']} ({analysis['tie_rate']*100:.1f}%)")
        print(f"平均置信度: {analysis['avg_confidence']:.2f}")
        print(f"综合得分: 方案A {analysis['a_overall_score']:.2f} vs 方案B {analysis['b_overall_score']:.2f}")
        print("="*60)
        print(f"\n结果已保存到: {output_dir}")
        
    except Exception as e:
        logger.error(f"判别过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

