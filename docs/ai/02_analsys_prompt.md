# Prompt 质量分析指南

## 概述

本文档阐述如何系统地衡量和对比不同版本 prompt 的 `raw_response` 质量。通过多个维度的指标分析，可以科学地评估哪个 prompt 表现更好。

---

## 一、核心分析维度

### 1. 结构化程度 ⭐⭐⭐ (最核心)

**定义**: 评估 `raw_response` 是否符合 JSON 结构规范

**子指标**:
- **JSON 有效性**: % 能被完全解析的有效 JSON 响应
- **字段完整性**: % 包含所有必需字段 (domain, sub, is_personal, time, food, query)
- **格式规范性**: % 严格遵循 JSON 格式，无多余文本或 markdown 标记

**计算方法**:
```python
def check_json_validity(raw_response: str) -> dict:
    """检查 raw_response 的 JSON 有效性"""
    try:
        obj = json.loads(raw_response.strip())
        
        # 检查是否是拒答
        if isinstance(obj, dict) and obj.get("refuse"):
            return {
                "valid": True,
                "type": "refuse",
                "fields_complete": "reason" in obj
            }
        
        # 检查是否是计划数组
        if isinstance(obj, list):
            if not obj:
                return {"valid": True, "type": "empty_plan", "completeness": 1.0}
            
            required_fields = {"domain", "sub", "is_personal", "time", "food", "query"}
            complete_items = sum(
                1 for item in obj 
                if isinstance(item, dict) and required_fields.issubset(item.keys())
            )
            completeness = complete_items / len(obj)
            
            return {
                "valid": True,
                "type": "plans",
                "completeness": completeness,
                "n_plans": len(obj)
            }
        
        return {"valid": False, "reason": "unexpected_structure"}
    except json.JSONDecodeError as e:
        return {"valid": False, "reason": f"json_decode_error: {str(e)[:50]}"}
    except Exception as e:
        return {"valid": False, "reason": f"unexpected_error: {str(e)[:50]}"}
```

**期望结果**: new prompt 应该有 > 95% 的 JSON 有效性

---

### 2. 金标签匹配度 ⭐⭐⭐

**定义**: 对比 `raw_response` 与数据集中的 `gold_label`

**子指标**:
- **精确匹配**: % 与 gold_label 完全相同（去除空格后比对）
- **关键字段匹配**: % 在 domain 和 is_personal 上与 gold_label 一致
- **部分匹配**: 编辑距离（Levenshtein distance）相似度

**计算方法**:
```python
def compare_with_gold(raw_response: str, gold_label: str) -> dict:
    """对比 raw_response 与 gold_label"""
    try:
        # 解析两边
        try:
            pred = json.loads(raw_response.strip())
        except:
            return {"exact_match": False, "key_field_match": False, "valid": False}
        
        try:
            if str(gold_label).strip() == "REFUSE":
                gold = {"refuse": True}
            else:
                gold = json.loads(gold_label.strip())
        except:
            return {"exact_match": False, "key_field_match": False, "valid": False}
        
        # 1. 完全匹配（规范化后）
        exact_match = (pred == gold)
        
        # 2. 关键字段匹配
        key_field_match = False
        if isinstance(pred, list) and isinstance(gold, list):
            if len(pred) == len(gold):
                key_field_match = all(
                    p.get("domain") == g.get("domain") and 
                    p.get("is_personal") == g.get("is_personal")
                    for p, g in zip(pred, gold)
                )
        elif isinstance(pred, dict) and isinstance(gold, dict):
            # 都是拒答或都有特定结构
            if pred.get("refuse") and gold.get("refuse"):
                key_field_match = True
        
        # 3. 相似度（可选）
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(
            None,
            json.dumps(pred, ensure_ascii=False),
            json.dumps(gold, ensure_ascii=False)
        ).ratio()
        
        return {
            "exact_match": exact_match,
            "key_field_match": key_field_match,
            "similarity": similarity,
            "valid": True
        }
    except Exception as e:
        return {
            "exact_match": False,
            "key_field_match": False,
            "similarity": 0.0,
            "valid": False,
            "error": str(e)[:50]
        }
```

**期望结果**: new prompt 应该有更高的 gold_label 匹配度

---

### 3. 多样性与覆盖率 ⭐⭐

**定义**: 评估输出的多样性和适应能力

**子指标**:
- **输出多样性**: 不同 `raw_response` 的种类数 / 总查询数
- **拒答比例**: % 返回拒答的查询
- **计划平均长度**: 每个查询的 plan 数量平均值

**计算方法**:
```python
def analyze_diversity(df: pd.DataFrame, variant: str) -> dict:
    """分析输出多样性"""
    dv = df[df["variant"] == variant]
    
    # 总查询数（去重 idx）
    n_queries = dv["idx"].nunique()
    
    # 不同的 raw_response 种类
    unique_raw = dv["raw_response"].nunique()
    diversity_rate = unique_raw / len(dv) if len(dv) > 0 else 0
    
    # 拒答比例
    refuse_count = (dv["type"] == "refuse").sum()
    refuse_rate = refuse_count / len(dv) if len(dv) > 0 else 0
    
    # 计划平均长度（仅统计非拒答的）
    plans_dv = dv[dv["type"] == "plans"]
    avg_plan_length = (
        plans_dv["n_plans"].mean() 
        if len(plans_dv) > 0 else 0
    )
    
    return {
        "n_queries": n_queries,
        "unique_raw_responses": unique_raw,
        "diversity_rate": diversity_rate,
        "refuse_rate": refuse_rate,
        "avg_plan_length": avg_plan_length,
        "variant": variant
    }
```

**期望结果**: 
- 多样性适中（过高表示不稳定，过低表示过度重复）
- 拒答比例应接近金标签的拒答比例（≈ 8.4%）

---

### 4. 鲁棒性检查 ⭐⭐⭐

**定义**: 评估输出的错误率和质量稳定性

**子指标**:
- **失败率**: % 无法解析或异常的响应
- **幻觉率**: % 输出不存在的 domain 或 sub 的情况
- **超长输出**: % raw_response 超过预期长度的情况

**计算方法**:
```python
def analyze_robustness(df: pd.DataFrame, variant: str, 
                       valid_domains: set = None) -> dict:
    """分析鲁棒性"""
    dv = df[df["variant"] == variant]
    
    # 1. 失败率（包括解析错误和异常）
    failure_mask = dv["ok"] == False
    failure_rate = failure_mask.sum() / len(dv) if len(dv) > 0 else 0
    
    # 2. 幻觉率（输出不存在的 domain）
    if valid_domains is None:
        valid_domains = {
            "体温", "减脂", "心脏健康", "情绪健康", "生理健康",
            "血压", "血氧饱和度", "血糖", "睡眠", "午睡", "步数",
            "活力三环", "微体检", "饮食", "跑步", "骑行", "步行徒步",
            "游泳", "登山", "跳绳", "瑜伽", "普拉提", "划船机", "其他"
        }
    
    hallucination_count = 0
    for idx, row in dv.iterrows():
        if row["type"] == "plans" and pd.notna(row["parsed"]):
            try:
                parsed = json.loads(row["parsed"])
                if isinstance(parsed, list):
                    for plan in parsed:
                        if plan.get("domain") not in valid_domains:
                            hallucination_count += 1
            except:
                pass
    
    hallucination_rate = hallucination_count / len(dv) if len(dv) > 0 else 0
    
    # 3. 超长输出（定义为 > 500 字符）
    long_output_count = (dv["raw_response"].str.len() > 500).sum()
    long_output_rate = long_output_count / len(dv) if len(dv) > 0 else 0
    
    return {
        "failure_rate": failure_rate,
        "hallucination_rate": hallucination_rate,
        "long_output_rate": long_output_rate,
        "variant": variant
    }
```

**期望结果**:
- 失败率 < 5%
- 幻觉率 ≈ 0%
- 超长输出率 < 1%

---

### 5. 性能指标 ⭐

**定义**: 响应时间等性能表现

**子指标**:
- **平均延迟**: mean(latency_sec)
- **P95 延迟**: 95 分位数延迟
- **超时率**: % 响应时间 > 阈值的情况

**计算方法**:
```python
def analyze_performance(df: pd.DataFrame, variant: str) -> dict:
    """分析性能指标"""
    dv = df[df["variant"] == variant]
    valid_latency = dv["latency_sec"].dropna()
    
    return {
        "mean_latency": valid_latency.mean() if len(valid_latency) > 0 else None,
        "p95_latency": valid_latency.quantile(0.95) if len(valid_latency) > 0 else None,
        "p99_latency": valid_latency.quantile(0.99) if len(valid_latency) > 0 else None,
        "timeout_rate": (dv["latency_sec"] > 30).sum() / len(dv) if len(dv) > 0 else 0,
        "variant": variant
    }
```

**期望结果**: new prompt 延迟应该与 old prompt 相当或更短

---

## 二、综合评分方法

### 总体质量评分 (Quality Score)

根据各维度加权计算：

```python
def calculate_quality_score(metrics: dict) -> float:
    """计算综合质量评分 (0-100)"""
    
    # 权重分配
    weights = {
        "json_validity": 0.25,      # 结构化程度
        "exact_match_rate": 0.25,   # 金标签匹配度
        "key_field_match_rate": 0.15,  # 关键字段匹配
        "failure_rate": 0.15,       # 鲁棒性（负向指标）
        "hallucination_rate": 0.10, # 鲁棒性（负向指标）
        "diversity_rate": 0.05,     # 多样性
        "refuse_rate_accuracy": 0.05 # 拒答率准确度
    }
    
    score = 0.0
    score += metrics.get("json_validity", 0) * weights["json_validity"] * 100
    score += metrics.get("exact_match_rate", 0) * weights["exact_match_rate"] * 100
    score += metrics.get("key_field_match_rate", 0) * weights["key_field_match_rate"] * 100
    score += (1 - metrics.get("failure_rate", 0)) * weights["failure_rate"] * 100
    score += (1 - metrics.get("hallucination_rate", 0)) * weights["hallucination_rate"] * 100
    score += min(metrics.get("diversity_rate", 0), 0.3) / 0.3 * weights["diversity_rate"] * 100
    
    # 拒答率准确度（接近 8.4% 为最优）
    target_refuse_rate = 0.084
    actual_refuse_rate = metrics.get("refuse_rate", 0)
    refuse_accuracy = 1 - abs(actual_refuse_rate - target_refuse_rate) / target_refuse_rate
    refuse_accuracy = max(0, refuse_accuracy)
    score += refuse_accuracy * weights["refuse_rate_accuracy"] * 100
    
    return score
```

**评分范围**:
- 80-100: 优秀 (Excellent)
- 70-79: 良好 (Good)
- 60-69: 中等 (Fair)
- < 60: 需要改进 (Need Improvement)

---

## 三、操作流程

### Step 1: 运行评估

```bash
# 比较 v3 和 original prompt
uv run python scripts/run_eval.py \
  --data data/summary_train_v3.xlsx \
  --n 50 \
  --outdir outputs/analysis
```

结果输出到: `outputs/analysis/eval_results.csv`

### Step 2: 运行分析脚本

创建 `scripts/analyze_prompts.py`:

```python
#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import json

def full_analysis(csv_path: str):
    """完整的分析流程"""
    df = pd.read_csv(csv_path)
    
    # 为每个 variant 计算所有指标
    results = {}
    for variant in ['new', 'old']:
        print(f"\n{'='*50}")
        print(f"分析 {variant.upper()} Prompt")
        print(f"{'='*50}")
        
        # 1. 结构化程度
        # 2. 金标签匹配度
        # 3. 多样性
        # 4. 鲁棒性
        # 5. 性能
        
        results[variant] = {
            # 分别调用各个函数
        }
    
    # 对比两个 prompt
    print("\n" + "="*50)
    print("对比结果")
    print("="*50)
    # 生成对比表格
    
    return results
```

### Step 3: 生成对比报告

报告应包含以下内容：

```
## 对比总结

### 质量评分
- new prompt: 82.5 分 ✅
- old prompt: 75.3 分

### 关键指标对比
| 指标 | new | old | 优劣 |
|------|-----|-----|------|
| JSON 有效性 | 97% | 89% | ✅ new 更优 |
| 精确匹配 | 68% | 62% | ✅ new 更优 |
| 失败率 | 2% | 4% | ✅ new 更优 |
| 拒答率 | 8.6% | 8.2% | ➖ 接近 |
| 平均延迟 | 2.3s | 2.1s | ❌ old 更快 |

### 结论
new prompt 在结构化程度、准确性和鲁棒性上表现更好，
虽然延迟稍长，但整体质量提升明显。
```

---

## 四、注意事项

### 4.1 数据质量注意

- ⚠️ 确保 `gold_label` 是正确的基准真值
- ⚠️ 对于拒答项，应有清晰的拒答原因
- ⚠️ 注意 NaN 和 None 值的处理

### 4.2 分析的局限性

- 当前数据集较小（427 行），统计结论可能有偏差
- 建议在更大数据集上验证结论
- 不同类型的查询（拒答 vs 正常）应分别分析

### 4.3 迭代改进建议

1. **提高 JSON 有效性**: 在 prompt 中强调 JSON 格式要求
2. **提高匹配度**: 
   - 调整 domain 枚举的清晰度
   - 提供更多的示例
3. **减少幻觉**: 严格限制 domain 和 sub 的枚举值
4. **加快响应**: 简化 prompt，减少不必要的说明

---

## 五、参考数据

### 基线指标（original_system_prompt）

根据 v3 运行结果：
- JSON 有效性: ~89%
- 平均延迟: ~2.1s
- 拒答率: ~8.2%

### 目标指标

- JSON 有效性: ≥ 95%
- 精确匹配率: ≥ 70%
- 失败率: ≤ 3%
- 拒答率: 7-9%（接近金标签的 8.4%）
- 综合评分: ≥ 80

---

## 六、快速开始

### 最简单的对比方式

```bash
# 1. 运行评估
uv run python scripts/run_eval.py --data data/summary_train_v3.xlsx --n 50

# 2. 在 outputs/v4 中查看结果
cat outputs/v4/summary.txt        # 快速统计
# 在 Excel 中打开 eval_results.csv 手动分析
```

### 自动化分析

```bash
# 创建分析脚本后运行
uv run python scripts/analyze_prompts.py outputs/v4/eval_results.csv
```
