#!/bin/bash
#SBATCH --job-name=test-local-llm-v100
#SBATCH --output=logs/test_local_model_v100_%j.out
#SBATCH --error=logs/test_local_model_v100_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# 设置环境变量
export HF_HOME=/ibex/user/wuj0c/cache/HF
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "本地 HuggingFace 模型测试 (V100 版本)"
echo "=========================================="
echo "开始时间: $(date)"
echo "HF_HOME: $HF_HOME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# 进入项目目录
cd /ibex/user/wuj0c/Projects/LLM/Safety/LLM-Agent-queryplan-eval

# 创建日志目录
mkdir -p logs

echo ""
echo "步骤 0: 安装缺少的依赖..."
uv pip install accelerate

echo ""
echo "步骤 1: 测试 GPU 可用性..."
python -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 设备数: {torch.cuda.device_count()}')
    print(f'当前设备: {torch.cuda.current_device()}')
    print(f'设备名称: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'显存大小: {props.total_memory / 1e9:.2f} GB')
"

echo ""
echo "步骤 2: 测试评估脚本 (本地模型，2个样本)..."
uv run python scripts/run_eval_ragtruth.py \
  --llm-type local \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --device cuda \
  --task-types Summary \
  --sample 2 \
  --temperature 0.0 \
  --outdir outputs/test_local_model_v100

echo ""
echo "步骤 3: 测试评估脚本 (OpenAI API，作为对比)..."
uv run python scripts/run_eval_ragtruth.py \
  --llm-type openai \
  --model qwen-flash \
  --task-types Summary \
  --sample 2 \
  --temperature 0.0 \
  --outdir outputs/test_openai_model_v100

echo ""
echo "=========================================="
echo "测试完成: $(date)"
echo "=========================================="
