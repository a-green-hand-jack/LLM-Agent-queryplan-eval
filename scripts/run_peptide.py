#!/usr/bin/env python3
"""肽段专利任务评估脚本

使用方式:
    uv run python scripts/run_peptide.py --data data/patents/US11111272B2_SIF_SGF_RawData_with_sequence.csv
    uv run python scripts/run_peptide.py --data data/patents/US11111272B2_SIF_SGF_RawData_with_sequence.csv -n 10 --outdir outputs/peptide_test
    uv run python scripts/run_peptide.py --data data/patents/US11111272B2_SIF_SGF_RawData_with_sequence.csv --llm-type openai --outdir outputs/peptide_openai
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from argparse import ArgumentParser
from dotenv import load_dotenv
import os

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from queryplan_eval.llms import OpenAILLM #, HuggingFaceLLM
from queryplan_eval.core.prompt_manager import PatentPromptManager
from queryplan_eval.tasks import PeptideTask

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主程序入口"""
    parser = ArgumentParser(description="肽段专利任务评估")
    parser.add_argument(
        "--data",
        required=True,
        help="数据文件路径（CSV 格式）"
    )
    parser.add_argument(
        "-n",
        "--sample",
        type=int,
        default=None,
        help="随机采样 n 个样本（如果不指定则使用全部数据）"
    )
    parser.add_argument(
        "--llm-type",
        choices=["openai", "huggingface"],
        default="openai",
        help="LLM 类型（默认: openai）"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="qwen3-max",
        help="LLM 模型名称（如 gpt-4o, qwen3-max 等）"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM 采样温度（默认: 0.0）"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/peptide_eval",
        help="输出目录（默认: outputs/peptide_eval）"
    )
    
    args = parser.parse_args()
    
    # 加载环境变量
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
    
    # 初始化 LLM
    logger.info(f"初始化 LLM: {args.llm_type}")
    if args.llm_type == "openai":
        api_key = os.getenv("qwen_key") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 环境变量未设置")
        llm = OpenAILLM(
            model_name=args.model_name or "gpt-4o",
            base_url=base_url,
            api_key=api_key
        )
    else:
        # llm = HuggingFaceLLM(model_name=args.model_name)
        raise ValueError(f"不支持的 LLM 类型: {args.llm_type}")
    
    # 初始化 Prompt Manager（切换到 v2）
    prompt_manager = PatentPromptManager(version="v2")
    
    # 初始化任务
    logger.info(f"初始化肽段任务，数据文件: {args.data}")
    task = PeptideTask(
        data_path=args.data,
        llm=llm,
        prompt_manager=prompt_manager,
        output_dir=args.outdir,
        sample_n=args.sample
    )
    
    # 运行评估
    logger.info(f"开始评估，温度: {args.temperature}")
    metrics = task.run_evaluation(temperature=args.temperature)
    
    # 输出结果
    logger.info("===== 评估完成 =====")
    logger.info(f"总样本数: {metrics['total']}")
    logger.info(f"成功数: {metrics['ok']} ({metrics['ok_rate']:.1%})")
    logger.info(f"平均延迟: {metrics['latency_mean']:.3f}s" if metrics['latency_mean'] else "延迟: N/A")
    logger.info(f"P95 延迟: {metrics['latency_p95']:.3f}s" if metrics['latency_p95'] else "P95: N/A")
    logger.info(f"结果已保存到: {args.outdir}")


if __name__ == "__main__":
    main()
