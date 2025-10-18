from __future__ import annotations
import os, time, json, argparse, warnings
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

import openai
import outlines

from .schemas import ResultType, Plan, Refuse, normalize_result
from .renderer import render_system_prompt, read_raw_prompt
from .data_utils import load_queries

def build_chat(system_prompt: str, user_query: str) -> list[dict[str, Any]]:
    # Pack history as empty for this simple eval; can be extended to include actual last turn.
    payload = json.dumps({
        "history": {"user": None, "assistant": None},
        "question": user_query
    }, ensure_ascii=False)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": payload},
    ]

def call_outlines(model, prompt_or_chat, *, temperature: float, output_type):
    """Call outlines model with structured output. Returns (json_like, raw_str)."""
    t0 = time.time()
    result = model(prompt_or_chat, output_type, temperature=temperature)
    dt = time.time() - t0
    # Outlines usually returns a JSON string; be robust:
    if isinstance(result, str):
        raw = result
        try:
            parsed = output_type.model_validate_json(raw) if hasattr(output_type, 'model_validate_json') else None
        except Exception:
            parsed = None
    else:
        raw = json.dumps(result, ensure_ascii=False)
        parsed = result
    return parsed, raw, dt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to 12w_query_label.xlsx")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--today", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--model", type=str, default=os.environ.get("QWEN_MODEL", "qwen3-7b-instruct"))
    parser.add_argument("--base-url", type=str, default=os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    parser.add_argument("--outdir", type=str, default=str(Path(__file__).resolve().parents[1] / "outputs"))
    parser.add_argument("--new-prompt", type=str, default=str(Path(__file__).resolve().parents[1] / "prompts" / "queryplan_system_prompt.j2"))
    parser.add_argument("--old-prompt", type=str, default=str(Path(__file__).resolve().parents[1] / "prompts" / "original_system_prompt.txt"))
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("qwen_key") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key. Set qwen_key in .env or OPENAI_API_KEY.")

    # Init OpenAI-compatible client (DashScope by default)
    client = openai.OpenAI(base_url=args.base_url, api_key=api_key)

    # Wrap with Outlines
    model = outlines.from_openai(client, args.model)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Prepare prompts
    today = args.today or time.strftime("%Y年%m月%d日")
    system_new = render_system_prompt(args.new_prompt, today=today)
    if not Path(args.old_prompt).exists():
        warnings.warn(f"Old prompt not found at {args.old_prompt}. Please paste your baseline prompt there.")
        system_old = system_new  # fallback to new prompt to avoid crash
    else:
        system_old = read_raw_prompt(args.old_prompt)

    df = load_queries(args.data, n=args.n)

    rows = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        q = str(row["query"]).strip()
        for variant, system_prompt in [("new", system_new), ("old", system_old)]:
            chat = build_chat(system_prompt, q)
            try:
                parsed, raw, dt = call_outlines(model, outlines.inputs.Chat(chat), temperature=args.temperature, output_type=ResultType)
                ok = parsed is not None
                norm = None
                n_plans = None
                out_type = None
                err = None
                if ok:
                    # parsed is either list[Plan] or Refuse
                    if isinstance(parsed, list):
                        out_type = "plans"
                        n_plans = len(parsed)
                        norm = {"plans": [p.model_dump() for p in parsed]}
                    else:
                        out_type = "refuse"
                        n_plans = 0
                        norm = {"refuse": True, "reason": parsed.reason}
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

            rows.append({
                "idx": i,
                "variant": variant,
                "query": q,
                "ok": ok,
                "type": out_type,
                "n_plans": n_plans,
                "latency_sec": dt,
                "raw": raw,
                "normalized": json.dumps(norm, ensure_ascii=False) if norm is not None else None,
                "error": err
            })

    res = pd.DataFrame(rows)
    res.to_csv(outdir / "eval_results.csv", index=False)

    # Summary
    def summarise(dfv):
        total = len(dfv)
        ok = dfv["ok"].sum()
        refuse = (dfv["type"] == "refuse").sum()
        parse_err = (dfv["type"].isin(["parse_error","exception"]).sum())
        lat_mean = dfv["latency_sec"].dropna().mean()
        lat_p95 = dfv["latency_sec"].dropna().quantile(0.95) if dfv["latency_sec"].notna().any() else None
        return total, ok, refuse, parse_err, lat_mean, lat_p95

    with open(outdir / "summary.txt", "w", encoding="utf-8") as f:
        for variant in ["new","old"]:
            dv = res[res["variant"]==variant]
            total, okc, refc, perr, latm, latp95 = summarise(dv)
            f.write(f"[{variant}] total={total}, ok={okc} ({okc/total:.1%}), refuse={refc}, parse_err/except={perr}, latency_mean={latm:.3f}s, p95={latp95:.3f}s\n")

    # Diffs: where normalized string differs between variants
    pivot = res.pivot_table(index=["idx","query"], columns="variant", values="normalized", aggfunc="first").reset_index()
    diffs = pivot[(pivot.get("new") != pivot.get("old")) & pivot["new"].notna() & pivot["old"].notna()]
    diffs.to_csv(outdir / "diffs.csv", index=False)

if __name__ == "__main__":
    main()
