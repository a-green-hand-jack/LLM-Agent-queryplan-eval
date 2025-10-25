#!/usr/bin/env python3
"""肽段专利任务评估脚本

支持单文件或批处理模式。

使用方式（单文件）:
    uv run python scripts/run_peptide.py --data data/patents/US11111272B2_SIF_SGF_RawData_with_sequence.csv
    uv run python scripts/run_peptide.py --data data/patents/US11111272B2_SIF_SGF_RawData_with_sequence.csv -n 10 --outdir outputs/peptide_test
    uv run python scripts/run_peptide.py --data data/patents/US11111272B2_SIF_SGF_RawData_with_sequence.csv --llm-type openai --outdir outputs/peptide_openai

使用方式（批处理）:
    uv run python scripts/run_peptide.py --data data/patents --outdir outputs/peptide_batch --workers 4
    uv run python scripts/run_peptide.py --data data/patents -n 10 --outdir outputs/peptide_batch --workers 8
"""

from __future__ import annotations

import sys
import logging
import os
from pathlib import Path
from argparse import ArgumentParser
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Tuple

import pandas as pd

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from queryplan_eval.llms import OpenAILLM
from queryplan_eval.core.prompt_manager import PatentPromptManager
from queryplan_eval.tasks import PeptideTask

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_csv(csv_path: Path) -> Tuple[bool, Optional[str]]:
    """校验 CSV 文件的合法性
    
    校验项：
    - 文件存在且可读
    - 能被 pandas 读取
    - 包含 SEQUENCE 列
    - 行数 >= 1
    
    Args:
        csv_path: CSV 文件路径
        
    Returns:
        (是否合法, 错误信息)。若合法返回 (True, None)，否则返回 (False, 错误信息)
    """
    if not csv_path.exists():
        return False, f"文件不存在: {csv_path}"
    
    if not csv_path.is_file():
        return False, f"路径不是文件: {csv_path}"
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return False, f"无法读取 CSV: {e}"
    
    if len(df) == 0:
        return False, "CSV 文件为空（0 行）"
    
    # 规范化列名并检查必需列
    cols = [c.strip().upper() for c in df.columns]
    if "SEQUENCE" not in cols:
        return False, f"CSV 缺少 'SEQUENCE' 列，可用列: {df.columns.tolist()}"
    
    return True, None


def scan_and_validate_csv_files(directory: Path) -> Tuple[List[Path], List[Tuple[Path, str]]]:
    """扫描目录中的 CSV 文件并校验合法性
    
    Args:
        directory: 目录路径
        
    Returns:
        (合法文件列表, 非法文件列表及其错误信息)
    """
    if not directory.is_dir():
        raise ValueError(f"路径不是目录: {directory}")
    
    valid_files: List[Path] = []
    invalid_files: List[Tuple[Path, str]] = []
    
    # 扫描目录中的 CSV 文件（非递归）
    csv_candidates = [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() == ".csv"
    ]
    
    logger.info(f"扫描到 {len(csv_candidates)} 个 CSV 候选文件")
    
    for csv_file in sorted(csv_candidates):
        is_valid, error_msg = validate_csv(csv_file)
        if is_valid:
            valid_files.append(csv_file)
            logger.info(f"✓ 合法: {csv_file.name}")
        else:
            if error_msg is not None:
                invalid_files.append((csv_file, error_msg))
                logger.warning(f"✗ 非法 {csv_file.name}: {error_msg}")
    
    return valid_files, invalid_files


def process_single_csv(
    csv_path: Path,
    api_key: str,
    base_url: str,
    model_name: str,
    output_dir: Path,
    sample_n: Optional[int],
    temperature: float
) -> Dict:
    """处理单个 CSV 文件
    
    Args:
        csv_path: CSV 文件路径
        api_key: OpenAI API 密钥
        base_url: API 基础 URL
        model_name: 模型名称
        output_dir: 输出目录
        sample_n: 采样数量
        temperature: 采样温度
        
    Returns:
        处理结果字典
    """
    csv_stem = csv_path.stem
    file_output_dir = output_dir / csv_stem
    
    try:
        logger.info(f"[{csv_stem}] 开始处理")
        
        # 创建输出目录
        file_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化 LLM（每个线程独立实例）
        llm = OpenAILLM(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key
        )
        
        # 初始化 Prompt Manager
        prompt_manager = PatentPromptManager(version="v2")
        
        # 初始化任务
        task = PeptideTask(
            data_path=str(csv_path),
            llm=llm,
            prompt_manager=prompt_manager,
            output_dir=str(file_output_dir),
            sample_n=sample_n
        )
        
        # 运行评估
        metrics = task.run_evaluation(temperature=temperature)
        
        logger.info(
            f"[{csv_stem}] 完成: "
            f"total={metrics['total']}, ok={metrics['ok']}, "
            f"ok_rate={metrics['ok_rate']:.1%}, "
            f"latency_mean={metrics['latency_mean']:.3f}s" if metrics['latency_mean'] else "N/A"
        )
        
        return {
            "file": csv_stem,
            "path": str(csv_path),
            "output_dir": str(file_output_dir),
            "status": "success",
            "metrics": metrics,
            "error": None
        }
    
    except Exception as e:
        logger.error(f"[{csv_stem}] 处理失败: {e}")
        return {
            "file": csv_stem,
            "path": str(csv_path),
            "output_dir": str(file_output_dir),
            "status": "failed",
            "metrics": None,
            "error": str(e)
        }


def process_single_file(
    csv_path: Path,
    api_key: str,
    base_url: str,
    model_name: str,
    output_dir: Path,
    sample_n: Optional[int],
    temperature: float
) -> None:
    """处理单个 CSV 文件（单文件模式）
    
    Args:
        csv_path: CSV 文件路径
        api_key: OpenAI API 密钥
        base_url: API 基础 URL
        model_name: 模型名称
        output_dir: 输出目录
        sample_n: 采样数量
        temperature: 采样温度
    """
    logger.info(f"单文件模式: {csv_path}")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化 LLM
    llm = OpenAILLM(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key
    )
    
    # 初始化 Prompt Manager
    prompt_manager = PatentPromptManager(version="v2")
    
    # 初始化任务
    task = PeptideTask(
        data_path=str(csv_path),
        llm=llm,
        prompt_manager=prompt_manager,
        output_dir=str(output_dir),
        sample_n=sample_n
    )
    
    # 运行评估
    metrics = task.run_evaluation(temperature=temperature)
    
    # 输出结果
    logger.info("===== 评估完成 =====")
    logger.info(f"总样本数: {metrics['total']}")
    logger.info(f"成功数: {metrics['ok']} ({metrics['ok_rate']:.1%})")
    logger.info(f"平均延迟: {metrics['latency_mean']:.3f}s" if metrics['latency_mean'] else "延迟: N/A")
    logger.info(f"P95 延迟: {metrics['latency_p95']:.3f}s" if metrics['latency_p95'] else "P95: N/A")
    logger.info(f"结果已保存到: {output_dir}")


def process_batch(
    directory: Path,
    api_key: str,
    base_url: str,
    model_name: str,
    output_dir: Path,
    sample_n: Optional[int],
    temperature: float,
    num_workers: int
) -> None:
    """批处理模式：扫描目录、校验、并行处理 CSV
    
    Args:
        directory: 包含 CSV 的目录
        api_key: OpenAI API 密钥
        base_url: API 基础 URL
        model_name: 模型名称
        output_dir: 批处理输出根目录
        sample_n: 采样数量
        temperature: 采样温度
        num_workers: 并行线程数
    """
    logger.info(f"批处理模式: {directory}")
    
    # 创建输出根目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 扫描和校验 CSV 文件
    valid_files, invalid_files = scan_and_validate_csv_files(directory)
    
    logger.info(f"扫描结果: {len(valid_files)} 个合法文件, {len(invalid_files)} 个非法文件")
    
    if invalid_files:
        logger.warning("非法文件列表:")
        for path, error in invalid_files:
            logger.warning(f"  - {path.name}: {error}")
    
    if not valid_files:
        logger.warning("未找到合法的 CSV 文件，退出批处理")
        return
    
    # 并行处理
    results: List[Dict] = []
    logger.info(f"开始并行处理 {len(valid_files)} 个文件，使用 {num_workers} 个工作线程")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(
                process_single_csv,
                csv_file,
                api_key,
                base_url,
                model_name,
                output_dir,
                sample_n,
                temperature
            ): csv_file
            for csv_file in valid_files
        }
        
        # 收集结果
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    # 生成汇总报告
    logger.info("===== 批处理完成 =====")
    logger.info(f"总处理文件数: {len(valid_files)}")
    
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")
    
    logger.info(f"成功: {success_count}/{len(results)}")
    logger.info(f"失败: {failed_count}/{len(results)}")
    
    if success_count > 0:
        logger.info("\n===== 成功处理的文件 =====")
        for result in results:
            if result["status"] == "success":
                metrics = result["metrics"]
                logger.info(
                    f"{result['file']}: "
                    f"total={metrics['total']}, ok={metrics['ok']}, "
                    f"ok_rate={metrics['ok_rate']:.1%}, "
                    f"avg_latency={metrics['latency_mean']:.3f}s" if metrics['latency_mean'] else "N/A"
                )
    
    if failed_count > 0:
        logger.info("\n===== 失败的文件 =====")
        for result in results:
            if result["status"] == "failed":
                logger.error(f"{result['file']}: {result['error']}")
    
    logger.info(f"\n结果已保存到: {output_dir}")


def main():
    """主程序入口"""
    parser = ArgumentParser(description="肽段专利任务评估（支持单文件或批处理模式）")
    parser.add_argument(
        "--data",
        required=True,
        help="数据文件路径（CSV 格式）或包含 CSV 的目录路径"
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
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="批处理模式下的并行工作线程数（默认: 8）"
    )
    
    args = parser.parse_args()
    
    # 加载环境变量
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
    
    # 初始化 LLM 配置
    logger.info(f"初始化 LLM: {args.llm_type}")
    if args.llm_type == "openai":
        api_key = os.getenv("qwen_key") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 环境变量未设置")
    else:
        raise ValueError(f"不支持的 LLM 类型: {args.llm_type}")
    
    # 判定输入类型
    data_path = Path(args.data)
    
    if not data_path.exists():
        raise FileNotFoundError(f"数据路径不存在: {args.data}")
    
    if data_path.is_file():
        # 单文件模式
        logger.info("检测到单文件模式")
        process_single_file(
            csv_path=data_path,
            api_key=api_key,
            base_url=base_url,
            model_name=args.model_name or "gpt-4o",
            output_dir=Path(args.outdir),
            sample_n=args.sample,
            temperature=args.temperature
        )
    elif data_path.is_dir():
        # 批处理模式
        logger.info("检测到批处理模式")
        process_batch(
            directory=data_path,
            api_key=api_key,
            base_url=base_url,
            model_name=args.model_name or "gpt-4o",
            output_dir=Path(args.outdir),
            sample_n=args.sample,
            temperature=args.temperature,
            num_workers=args.workers
        )
    else:
        raise ValueError(f"数据路径既不是文件也不是目录: {args.data}")


if __name__ == "__main__":
    main()
