# QueryPlan-LLM Prompt A/B Eval (Outlines + Qwen)

This mini project lets you **compare two prompt variants** against your dataset
(`/Users/jieke/Projects/KAUST/LLM-Agent/data/12w_query_label.xlsx`) using **Qwen3-7B**
through an **OpenAI-compatible API** and **Outlines** for structured generation.

## 1) Install

```bash
cd /Users/jieke/Projects/KAUST/LLM-Agent/code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Configure API
Create `.env` in this folder with your Qwen API key:

```
qwen_key=REPLACE_WITH_YOUR_KEY
# Optional: override the default DashScope OpenAI-compatible URL or model name
# QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# QWEN_MODEL=qwen3-7b-instruct
```

> Uses OpenAI-compatible client via `openai.OpenAI(base_url=..., api_key=...)`.
> DashScope's base URL for OpenAI-compat is documented by Alibaba Cloud Model Studio.
> (You can change provider if you serve Qwen yourself; just update `QWEN_BASE_URL`.)

## 3) Prompts
- `prompts/queryplan_system_prompt.j2` : **new Jinja2 system prompt** (recommended).
- `prompts/original_system_prompt.txt` : paste your **old prompt** here to A/B test.

## 4) Run
```bash
python -m src.run_eval   --data /Users/jieke/Projects/KAUST/LLM-Agent/data/12w_query_label.xlsx   --n 50   --today "$(date +'%Y年%m月%d日')"   --model "${QWEN_MODEL:-qwen3-7b-instruct}"
```

Outputs:
- `outputs/eval_results.csv` : per-sample results for both prompts
- `outputs/summary.txt`      : schema success rate, latency stats, refusal ratio
- `outputs/diffs.csv`        : cases where normalized outputs differ

## Notes
- We use **Pydantic schemas** with **Outlines** to guarantee structure.
- The result type is `Union[List[Plan], Refuse]`. We normalize to a unified dict.
- Temperature defaults to 0 for stability. Tune with `--temperature` if needed.
