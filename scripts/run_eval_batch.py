from __future__ import annotations
import os
import time
import json
import argparse
import warnings
import csv
import logging
from pathlib import Path
from typing import Any, Tuple, Optional, Type, TypeVar
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

import openai

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from queryplan_eval.schemas import QueryResult, normalize_result
from queryplan_eval.renderer import render_system_prompt, read_raw_prompt
from queryplan_eval.datasets import QueryPlanDataset, SplitConfig, split_dataset, take_samples
from queryplan_eval.batch_handler import (
    BatchExecutor, BatchRequest, BatchResponseProcessor, batch_split
)

T = TypeVar('T')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_chat(system_prompt: str, user_query: str) -> list[dict[str, Any]]:
    """构建聊天消息格式
    
    Args:
        system_prompt: 系统提示
        user_query: 用户查询
        
    Returns:
        聊天消息列表
    """
    payload = json.dumps(
        {"history": {"user": None, "assistant": None}, "question": user_query},
        ensure_ascii=False,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": payload},
    ]


def build_batch_requests(
    items: list[Any],
    system_new: str,
    system_old: str,
    req_id_offset: int = 0,
) -> list[BatchRequest]:
    """为一个批次构建请求列表
    
    Args:
        items: 数据项列表
        system_new: 新系统提示
        system_old: 旧系统提示
        req_id_offset: 请求 ID 偏移量
        
    Returns:
        BatchRequest 对象列表
    """
    requests = []
    req_id = req_id_offset
    
    for item in items:
        q = str(item.query).strip()
        
        for variant, system_prompt in [("new", system_new), ("old", system_old)]:
            chat = build_chat(system_prompt, q)
            custom_id = f"{item.idx}_{variant}_{req_id}"
            
            request = BatchRequest(
                custom_id=custom_id,
                body={
                    "model": "qwen-plus",  # 这会被实际模型名称覆盖
                    "messages": chat,
                    "temperature": 0.0,
                }
            )
            requests.append(request)
            req_id += 1
    
    return requests


def process_batch_results(
    batch_results: list,
    demo_set: list[Any],
    fieldnames: list[str],
    csv_writer: csv.DictWriter,
    csv_file,
) -> list[dict]:
    """处理批处理结果并写入 CSV
    
    Args:
        batch_results: 批处理结果列表
        demo_set: 原始数据集
        fieldnames: CSV 字段名
        csv_writer: CSV 写入器
        csv_file: CSV 文件对象
        
    Returns:
        所有记录的列表
    """
    rows = []
    
    # 建立查找表
    item_map = {item.idx: item for item in demo_set}
    
    for result in batch_results:
        # 解析 custom_id: idx_variant_req_id
        parts = result.custom_id.rsplit("_", 1)
        if len(parts) != 2:
            logger.warning(f"无效的 custom_id 格式: {result.custom_id}")
            continue
        
        idx_variant_part, _ = parts
        parts2 = idx_variant_part.rsplit("_", 1)
        if len(parts2) != 2:
            logger.warning(f"无法解析 custom_id: {result.custom_id}")
            continue
        
        idx_str, variant = parts2
        
        try:
            idx = int(idx_str)
        except ValueError:
            logger.warning(f"无效的索引值: {idx_str}")
            continue
        
        item = item_map.get(idx)
        if not item:
            logger.warning(f"找不到索引为 {idx} 的项目")
            continue
        
        q = str(item.query).strip()
        gold_label = str(item.plan).strip() if not pd.isna(item.plan) else None
        
        # 提取结构化输出
        parsed, raw, error = BatchResponseProcessor.extract_structured_output(
            result, QueryResult
        )
        
        ok = parsed is not None
        norm: Optional[dict] = None
        n_plans: Optional[int] = None
        out_type: Optional[str] = None
        err: Optional[str] = error
        
        if ok and parsed is not None:
            if parsed.refused:
                out_type = "refuse"
                n_plans = 0
            else:
                out_type = "plans"
                n_plans = len(parsed.plans)
            norm = normalize_result(parsed)
        else:
            out_type = "parse_error" if err is None else "exception"
        
        # 计算延迟（批处理中不可用，设为 None）
        dt = None
        
        record = {
            "idx": idx,
            "variant": variant,
            "query": q,
            "raw_response": raw,
            "ok": ok,
            "type": out_type,
            "n_plans": n_plans,
            "latency_sec": dt,
            "parsed": json.dumps(norm, ensure_ascii=False)
            if norm is not None
            else None,
            "gold_label": gold_label,
            "error": err,
        }
        
        csv_writer.writerow(record)
        csv_file.flush()
        rows.append(record)
    
    return rows


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to 12w_query_label.xlsx")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--today", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--model", type=str, default=os.environ.get("QWEN_MODEL", "qwen-flash")
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get(
            "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "outputs/tmp"),
    )
    parser.add_argument(
        "--new-prompt",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1]
            / "src"
            / "queryplan_eval"
            / "prompts"
            / "queryplan_system_prompt_v5.j2"
        ),
    )
    parser.add_argument(
        "--old-prompt",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1]
            / "src"
            / "queryplan_eval"
            / "prompts"
            / "original_system_prompt.txt"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="每个批次的大小"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=10.0,
        help="轮询间隔（秒）"
    )
    parser.add_argument(
        "--max-wait-time",
        type=float,
        default=1800.0,
        help="最大等待时间（秒）"
    )

    return parser.parse_args()


def main():
    args = parser_args()

    load_dotenv()
    api_key = os.environ.get("qwen_key") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key. Set qwen_key in .env or OPENAI_API_KEY.")

    # Init OpenAI-compatible client (DashScope by default)
    client = openai.OpenAI(base_url=args.base_url, api_key=api_key)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Prepare prompts
    today = args.today or time.strftime("%Y年%m月%d日")
    system_new = render_system_prompt(args.new_prompt, today=today)
    if not Path(args.old_prompt).exists():
        warnings.warn(
            f"Old prompt not found at {args.old_prompt}. Please paste your baseline prompt there."
        )
        system_old = system_new  # fallback to new prompt to avoid crash
    else:
        system_old = read_raw_prompt(args.old_prompt)

    dataset = QueryPlanDataset(args.data)
    config = SplitConfig(split_type="train_eval_test", train_ratio=0.7, eval_ratio=0.2, test_ratio=0.1)
    splits = split_dataset(dataset, config)
    train_set = splits["train"]
    demo_set = take_samples(train_set, n=args.n)

    # 准备 CSV 写入器
    csv_path = outdir / "eval_results.csv"
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    
    # 定义 CSV 列
    fieldnames = [
        "idx", "variant", "query", "raw_response", "ok", "type", 
        "n_plans", "latency_sec", "parsed", "gold_label", "error"
    ]
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()  # 写入表头
    csv_file.flush()
    
    # 初始化批处理执行器
    executor = BatchExecutor(
        client=client,
        model=args.model,
        batch_size=args.batch_size,
        poll_interval=args.poll_interval,
        max_wait_time=args.max_wait_time,
    )
    
    # 将演示集分割成多个批次
    all_rows = []
    demo_batches = batch_split(list(demo_set), args.batch_size)
    
    logger.info(f"将 {len(demo_set)} 条数据分成 {len(demo_batches)} 个批次，每批大小: {args.batch_size}")
    
    for batch_idx, batch_items in enumerate(tqdm(demo_batches, desc="Processing batches")):
        logger.info(f"处理批次 {batch_idx + 1}/{len(demo_batches)}")
        
        # 构建批处理请求
        requests = build_batch_requests(
            batch_items,
            system_new,
            system_old,
            req_id_offset=batch_idx * args.batch_size * 2,  # 每项生成 2 个请求
        )
        
        logger.info(f"批次 {batch_idx + 1}: 提交 {len(requests)} 条请求")
        
        # 执行批处理
        batch_results = executor.execute_batch(requests)
        
        if batch_results is None:
            logger.error(f"批次 {batch_idx + 1} 执行失败")
            continue
        
        logger.info(f"批次 {batch_idx + 1}: 收到 {len(batch_results)} 条结果")
        
        # 处理结果并写入 CSV
        rows = process_batch_results(
            batch_results,
            batch_items,
            fieldnames,
            csv_writer,
            csv_file,
        )
        
        all_rows.extend(rows)
        logger.info(f"批次 {batch_idx + 1} 完成，已处理 {len(rows)} 条结果")
    
    csv_file.close()

    logger.info(f"共处理 {len(all_rows)} 条结果，保存到 {csv_path}")

    # 将结果读取回内存用于后续统计分析
    res = pd.read_csv(csv_path)

    # Summary
    def summarise(dfv):
        total = len(dfv)
        ok = dfv["ok"].sum()
        refuse = (dfv["type"] == "refuse").sum()
        parse_err = dfv["type"].isin(["parse_error", "exception"]).sum()
        lat_mean = dfv["latency_sec"].dropna().mean()
        lat_p95 = (
            dfv["latency_sec"].dropna().quantile(0.95)
            if dfv["latency_sec"].notna().any()
            else None
        )
        return total, ok, refuse, parse_err, lat_mean, lat_p95

    with open(outdir / "summary.txt", "w", encoding="utf-8") as f:
        for variant in ["new", "old"]:
            dv = res[res["variant"] == variant]
            total, okc, refc, perr, latm, latp95 = summarise(dv)
            # 处理 NaN 和 None 值，使用安全的格式化
            try:
                latm_str = f"{float(latm):.3f}" if latm is not None and not (isinstance(latm, float) and latm != latm) else "N/A"
            except (ValueError, TypeError):
                latm_str = "N/A"
            latp95_str = f"{latp95:.3f}" if latp95 is not None else "N/A"
            f.write(
                f"[{variant}] total={total}, ok={okc} ({okc / total:.1%}), refuse={refc}, parse_err/except={perr}, latency_mean={latm_str}s, p95={latp95_str}s\n"
            )

    # Diffs: where normalized string differs between variants
    pivot = res.pivot_table(
        index=["idx", "query"], columns="variant", values="parsed", aggfunc="first"
    ).reset_index()
    
    # 安全地处理可能不存在的列
    if "new" in pivot.columns and "old" in pivot.columns:
        diffs = pivot[
            (pivot["new"] != pivot["old"])
            & pivot["new"].notna()
            & pivot["old"].notna()
        ]
    else:
        # 如果任意一个变体没有数据，创建空的 diffs 表
        diffs = pivot.iloc[:0]  # 创建空的同结构 DataFrame
    
    diffs.to_csv(outdir / "diffs.csv", index=False)

    logger.info("评估完成！")


if __name__ == "__main__":
    main()
