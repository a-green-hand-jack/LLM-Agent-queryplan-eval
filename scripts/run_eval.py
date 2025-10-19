from __future__ import annotations
import os
import time
import json
import argparse
import warnings
import csv
from pathlib import Path
from typing import Any, Tuple, Optional, Type, TypeVar
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

import openai
import outlines

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from queryplan_eval.schemas import QueryResult, normalize_result
from queryplan_eval.renderer import render_system_prompt, read_raw_prompt
from queryplan_eval.datasets import QueryPlanDataset, SplitConfig, split_dataset, take_samples

T = TypeVar('T')


def build_chat(system_prompt: str, user_query: str) -> list[dict[str, Any]]:
    # Pack history as empty for this simple eval; can be extended to include actual last turn.
    payload = json.dumps(
        {"history": {"user": None, "assistant": None}, "question": user_query},
        ensure_ascii=False,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": payload},
    ]


def call_outlines(model: Any, prompt_or_chat: Any, *, temperature: float, output_type: Type[T]) -> Tuple[Optional[T], Optional[str], float]:
    """使用 Outlines 调用模型进行结构化输出
    
    Args:
        model: Outlines 包装的模型对象
        prompt_or_chat: 提示或聊天消息
        temperature: 采样温度
        output_type: 期望的输出类型（Pydantic 模型）
        
    Returns:
        包含以下内容的元组：
        - parsed: 解析后的对象实例，如果解析失败则为 None
        - raw: 原始返回字符串
        - dt: 执行耗时（秒）
    """
    t0 = time.time()
    result = model(prompt_or_chat, output_type, temperature=temperature)
    dt = time.time() - t0
    # Outlines 通常返回 JSON 字符串；保持健壮性
    parsed: Optional[T] = None
    if isinstance(result, str):
        raw = result
        try:
            # 尝试使用 Pydantic 的 model_validate_json 方法进行解析
            if hasattr(output_type, "model_validate_json"):
                parsed = output_type.model_validate_json(raw)  # type: ignore
        except Exception:
            parsed = None
    else:
        raw = json.dumps(result, ensure_ascii=False)
        parsed = result
    return parsed, raw, dt

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
        default=str(Path(__file__).resolve().parents[1] / "outputs/full"),
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

    return parser.parse_args()


def main():
    
    args = parser_args()

    load_dotenv()
    api_key = os.environ.get("qwen_key") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key. Set qwen_key in .env or OPENAI_API_KEY.")

    # Init OpenAI-compatible client (DashScope by default)
    client = openai.OpenAI(base_url=args.base_url, api_key=api_key)

    # Wrap with Outlines
    model = outlines.from_openai(client, args.model)

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
    # eval_set = splits["eval"]
    # test_set = splits["test"]
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
    
    rows = []
    for item in tqdm(demo_set, total=len(demo_set), desc="Evaluating demo set"):
        q = str(item.query).strip()
        gold_label = str(item.plan).strip() if not pd.isna(item.plan) else None # type: ignore
        for variant, system_prompt in [("new", system_new), ("old", system_old)]:
            chat = build_chat(system_prompt, q)
            try:
                parsed, raw, dt = call_outlines(
                    model,
                    outlines.inputs.Chat(chat),
                    temperature=args.temperature,
                    output_type=QueryResult,
                )
                ok = parsed is not None
                norm: Optional[dict] = None
                n_plans: Optional[int] = None
                out_type: Optional[str] = None
                err: Optional[str] = None
                if ok and parsed is not None:
                    # parsed 现在是 QueryResult 对象
                    if parsed.refused:
                        out_type = "refuse"
                        n_plans = 0
                    else:
                        out_type = "plans"
                        n_plans = len(parsed.plans)
                    norm = normalize_result(parsed)
                else:
                    out_type = "parse_error"
            except Exception as e:
                ok = False
                raw = None
                dt = None
                out_type = "exception"
                n_plans = None
                err = str(e)
                norm = None

            record = {
                "idx": item.idx,
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
            
            # 立即写入 CSV
            csv_writer.writerow(record)
            csv_file.flush()
            rows.append(record)

    csv_file.close()

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


if __name__ == "__main__":
    main()
