#!/bin/bash
#SBATCH --job-name=plan-data-test
#SBATCH --output=logs/plan_data_test_%j.out
#SBATCH --error=logs/plan_data_test_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# 设置环境变量
export HF_HOME=/ibex/user/wuj0c/cache/HF
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "计划数据任务多模式测试"
echo "=========================================="
echo "开始时间: $(date)"
echo "HF_HOME: $HF_HOME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# 进入项目目录
cd /ibex/user/wuj0c/Projects/LLM/Safety/LLM-Agent-queryplan-eval
source .venv/bin/activate

# 设置数据文件和参数
DATA_FILE="data/plan_data.xlsx"
SAMPLE_N=20
DEVICE="cuda"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
LLM_TYPE="local"
BASE_OUTPUT_DIR="outputs/plan_data_test"

# 创建必要的目录
mkdir -p logs
mkdir -p ${BASE_OUTPUT_DIR}/single
mkdir -p ${BASE_OUTPUT_DIR}/multi
mkdir -p ${BASE_OUTPUT_DIR}/single_think
mkdir -p ${BASE_OUTPUT_DIR}/multi_think

echo ""
echo "数据文件: $DATA_FILE"
echo "采样数量: $SAMPLE_N"
echo "LLM 模型: $MODEL_NAME"
echo "设备: $DEVICE"
echo "基础输出目录: $BASE_OUTPUT_DIR"
echo ""

# 测试四种模式
MODES=("single" "multi" "single_think" "multi_think")

for MODE in "${MODES[@]}"; do
    echo ""
    echo "步骤: 运行 $MODE 模式评估..."
    echo "=========================================="
    
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODE}"
    
    uv run python scripts/run_plan_data.py \
        --mode "$MODE" \
        --path "$DATA_FILE" \
        --sample-n $SAMPLE_N \
        --use-cot \
        --output-dir "$OUTPUT_DIR" \
        --llm-model "$MODEL_NAME" \
        --llm-type "$LLM_TYPE" \
        --device "$DEVICE"
    
    if [ $? -ne 0 ]; then
        echo "❌ $MODE 模式评估失败"
        exit 1
    fi
    echo "✓ $MODE 模式评估完成"
    echo "结果保存到: $OUTPUT_DIR"
    echo ""
done

echo ""
echo "=========================================="
echo "所有模式测试完成: $(date)"
echo "=========================================="
echo ""
echo "📊 结果统计："
echo "  Single 模式结果: ${BASE_OUTPUT_DIR}/single/"
echo "  Multi 模式结果: ${BASE_OUTPUT_DIR}/multi/"
echo "  Single Think 模式结果: ${BASE_OUTPUT_DIR}/single_think/"
echo "  Multi Think 模式结果: ${BASE_OUTPUT_DIR}/multi_think/"
echo ""
echo "=========================================="
