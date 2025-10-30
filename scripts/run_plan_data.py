#!/usr/bin/env python3
"""计划数据评估运行脚本

支持对 plan_data.xlsx 进行多模式评估：
- single: 单轮查询，输出 type 字段
- multi: 多轮对话，输出 time_frame 字段
- single_think: 单轮查询，包含推理过程
- multi_think: 多轮对话，包含推理过程

用法示例：
    python scripts/run_plan_data.py --mode single
    python scripts/run_plan_data.py --mode multi --sample-n 50
    python scripts/run_plan_data.py --mode single_think --use-cot
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加 src 到路径以支持导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from queryplan_eval.tasks import PlanDataTask
from queryplan_eval.core.llm_factory import create_llm

logger = logging.getLogger(__name__)


def setup_logging(output_dir: str):
    """设置日志"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    log_file = output_path / "run.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"日志已配置，输出到: {log_file}")


def main():
    parser = argparse.ArgumentParser(
        description="计划数据评估脚本 - 评估 plan_data.xlsx 数据集"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "multi", "single_think", "multi_think"],
        help="评估模式"
    )
    
    parser.add_argument(
        "--path",
        type=str,
        default="data/plan_data.xlsx",
        help="plan_data.xlsx 文件路径"
    )
    
    parser.add_argument(
        "--sample-n",
        type=int,
        default=None,
        help="采样数量（默认使用全部数据）"
    )
    
    parser.add_argument(
        "--use-cot",
        action="store_true",
        default=True,
        help="是否在提示词中启用 Chain-of-Thought（默认启用）"
    )
    
    parser.add_argument(
        "--no-cot",
        action="store_false",
        dest="use_cot",
        help="禁用 Chain-of-Thought"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认为 outputs/plan_data_<mode>/）"
    )
    
    parser.add_argument(
        "--llm-model",
        type=str,
        default="qwen-turbo",
        help="LLM 模型名称（默认 qwen-turbo）"
    )
    
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default=None,
        help="LLM API 密钥（默认从环境变量读取）"
    )
    
    args = parser.parse_args()
    
    # 设置输出目录
    output_dir = args.output_dir or f"outputs/plan_data_{args.mode}"
    setup_logging(output_dir)
    
    logger.info(f"开始计划数据评估")
    logger.info(f"模式: {args.mode}")
    logger.info(f"数据文件: {args.path}")
    logger.info(f"采样数量: {args.sample_n}")
    logger.info(f"使用 CoT: {args.use_cot}")
    logger.info(f"输出目录: {output_dir}")
    
    # 检查数据文件
    data_path = Path(args.path)
    if not data_path.exists():
        logger.error(f"数据文件不存在: {args.path}")
        return 1
    
    try:
        # 初始化 LLM
        logger.info(f"初始化 LLM (model={args.llm_model})")
        llm = create_llm(
            model=args.llm_model,
            api_key=args.llm_api_key
        )
        
        # 初始化任务
        logger.info(f"初始化 PlanDataTask")
        task = PlanDataTask(
            mode=args.mode,
            enable_cot=args.use_cot,
            sample_n=args.sample_n,
            llm=llm,
            output_dir=output_dir
        )
        
        # 加载数据集
        logger.info(f"加载数据集")
        dataset = task.load_dataset(str(data_path))
        task.dataset = dataset
        
        # 运行评估
        logger.info(f"开始运行评估")
        metrics = task.run_evaluation()
        
        # 输出结果摘要
        logger.info("=" * 60)
        logger.info("评估完成 - 结果摘要")
        logger.info("=" * 60)
        logger.info(f"总样本数: {metrics.get('total', 0)}")
        logger.info(f"成功解析: {metrics.get('ok', 0)} ({metrics.get('ok_rate', 0):.1%})")
        
        # 输出字段准确率
        for field in ["domain", "sub", "is_personal", "time", "food"]:
            acc_key = f"acc_{field}"
            if acc_key in metrics:
                logger.info(f"  {field} 准确率: {metrics[acc_key]:.1%}")
        
        # 模式特定指标
        if args.mode == "single" and "acc_type" in metrics:
            logger.info(f"  type 准确率: {metrics['acc_type']:.1%}")
        elif args.mode == "multi" and "acc_time_frame" in metrics:
            logger.info(f"  time_frame 准确率: {metrics['acc_time_frame']:.1%}")
        elif args.mode in ("single_think", "multi_think") and "acc_think" in metrics:
            logger.info(f"  think 准确率: {metrics['acc_think']:.1%}")
        
        # 延迟统计
        if "latency_mean" in metrics:
            logger.info(f"平均延迟: {metrics['latency_mean']:.2f}s")
            logger.info(f"P95 延迟: {metrics['latency_p95']:.2f}s")
        
        logger.info("=" * 60)
        logger.info(f"结果已保存到: {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"评估过程中出错: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
