#!/bin/bash
#SBATCH --job-name=1021-prompt-compare
#SBATCH --output=logs/1021_compare_%j.out
#SBATCH --error=logs/1021_compare_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# 设置环境变量
export HF_HOME=/ibex/user/wuj0c/cache/HF
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "Prompt 版本对比评估 (1021 实验)"
echo "=========================================="
echo "开始时间: $(date)"
echo "HF_HOME: $HF_HOME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# 进入项目目录
cd /ibex/user/wuj0c/Projects/LLM/Safety/LLM-Agent-queryplan-eval

# 创建必要的目录
mkdir -p logs
mkdir -p outputs/1021/original
mkdir -p outputs/1021/v6_cot
mkdir -p outputs/1021/compare

# 设置数据文件和参数
DATA_FILE="data/summary_train_v3.xlsx"  # 根据实际情况调整
SAMPLE_N=100
DEVICE="cuda"
# MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
# LLM_TYPE="local"
MODEL_NAME="qwen-flash"
LLM_TYPE="openai"

echo ""
echo "步骤 1: 运行 original 版本评估..."
echo "=========================================="
uv run python scripts/run_eval.py \
  --data "$DATA_FILE" \
  --llm-type "$LLM_TYPE" \
  --model-name "$MODEL_NAME" \
  --device "$DEVICE" \
  -n $SAMPLE_N \
  --prompt-version original \
  --outdir outputs/1021/original

if [ $? -ne 0 ]; then
    echo "❌ Original 版本评估失败"
    exit 1
fi
echo "✓ Original 版本评估完成"

echo ""
echo "步骤 2: 运行 v6_cot 版本评估..."
echo "=========================================="
uv run python scripts/run_eval.py \
  --data "$DATA_FILE" \
  --llm-type "$LLM_TYPE" \
  --model-name "$MODEL_NAME" \
  --device "$DEVICE" \
  -n $SAMPLE_N \
  --prompt-version v6_cot \
  --outdir outputs/1021/v6_cot

if [ $? -ne 0 ]; then
    echo "❌ v6_cot 版本评估失败"
    exit 1
fi
echo "✓ v6_cot 版本评估完成"

echo ""
echo "步骤 3: 进行 LLM 对比判别..."
echo "=========================================="

# 找到结果文件
RESULTS_A="outputs/1021/original/eval_results.csv"
RESULTS_B="outputs/1021/v6_cot/eval_results.csv"

# 检查结果文件是否存在
if [ ! -f "$RESULTS_A" ]; then
    echo "❌ Original 结果文件不存在: $RESULTS_A"
    exit 1
fi

if [ ! -f "$RESULTS_B" ]; then
    echo "❌ v6_cot 结果文件不存在: $RESULTS_B"
    exit 1
fi

echo "Original 结果文件: $RESULTS_A"
echo "v6_cot 结果文件: $RESULTS_B"
echo "对比结果输出目录: outputs/1021/compare"
echo ""

uv run python scripts/llm_judge.py \
  "$RESULTS_A" \
  "$RESULTS_B" \
  --outdir outputs/1021/compare

if [ $? -ne 0 ]; then
    echo "❌ LLM 对比判别失败"
    exit 1
fi
echo "✓ LLM 对比判别完成"

echo ""
echo "=========================================="
echo "所有任务完成: $(date)"
echo "=========================================="
echo ""
echo "📊 结果统计："
echo "  Original 结果: outputs/1021/original/"
echo "  v6_cot 结果: outputs/1021/v6_cot/"
echo "  对比结果: outputs/1021/compare/"
echo ""
echo "=========================================="
