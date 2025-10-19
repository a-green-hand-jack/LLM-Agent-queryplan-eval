#!/usr/bin/env python3
"""查询计划抽取评估脚本 - 重构版本

使用方式:
    uv run python scripts/run_eval.py --data data/summary_train_v3.xlsx
    uv run python scripts/run_eval.py --data data/summary_train_v3.xlsx -n 50 --outdir outputs/test
    uv run python scripts/run_eval.py --data data/summary_train_v3.xlsx --enable-cot --outdir outputs/cot_test
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
from queryplan_eval.core.prompt_manager import PromptManager
from queryplan_eval.tasks import QueryPlanTask

# 配置日志
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主程序入口"""
    parser = ArgumentParser(description="查询计划抽取评估")
    parser.add_argument(
        "--data",
        required=True,
        help="数据文件路径（Excel 格式）"
    )
    parser.add_argument(
        "-n",
        "--sample",
        type=int,
        default=None,
        help="随机采样 n 个样本（如果不指定则使用全部数据）"
    )
    parser.add_argument(
        "--model",
        default="qwen-flash",
        help="模型名称（默认: qwen-flash）"
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="API 基础 URL"
    )
    parser.add_argument(
        "--prompt-version",
        default="latest",
        help="Prompt 版本（默认: latest 自动选择最新版本）"
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
        help="采样温度（默认: 0.0）"
    )
    parser.add_argument(
        "--outdir",
        default="outputs/latest",
        help="输出目录（默认: outputs/latest）"
    )
    
    args = parser.parse_args()
    
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
    
    logger.info(f"开始查询计划评估")
    logger.info(f"数据文件: {args.data}")
    logger.info(f"样本数: {args.sample if args.sample else '全部'}")
    logger.info(f"模型: {args.model}")
    logger.info(f"Prompt 版本: {args.prompt_version}")
    logger.info(f"启用 CoT: {args.enable_cot}")
    logger.info(f"输出目录: {args.outdir}")
    
    try:
        # 初始化 LLM
        llm = OpenAILLM(
            model_name=args.model,
            base_url=args.base_url,
            api_key=api_key
        )
        
        # 初始化 Prompt Manager
        prompt_manager = PromptManager(
            task_name="query_plan",
            version=args.prompt_version
        )
        
        # 初始化任务
        task = QueryPlanTask(
            data_path=args.data,
            llm=llm,
            prompt_manager=prompt_manager,
            output_dir=args.outdir,
            enable_cot=args.enable_cot,
            sample_n=args.sample
        )
        
        # 运行评估
        metrics = task.run_evaluation(temperature=args.temperature)
        
        # 打印摘要
        print("\n" + "="*60)
        print("评估完成")
        print("="*60)
        print(f"总样本数: {metrics['total']}")
        print(f"成功: {metrics['ok']} ({metrics['ok_rate']:.1%})")
        print(f"拒答: {metrics['refuse']} ({metrics['refuse_rate']:.1%})")
        print(f"解析错误: {metrics['parse_error']} ({metrics['parse_error_rate']:.1%})")
        if metrics.get('latency_mean'):
            print(f"平均延迟: {metrics['latency_mean']:.3f}s")
        if metrics.get('latency_p95'):
            print(f"P95 延迟: {metrics['latency_p95']:.3f}s")
        print("="*60)
        print(f"\n结果已保存到: {args.outdir}")
        
    except Exception as e:
        logger.error(f"评估过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
