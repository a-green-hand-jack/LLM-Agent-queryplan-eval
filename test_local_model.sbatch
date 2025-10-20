#!/bin/bash
#SBATCH --job-name=test-local-llm
#SBATCH --output=logs/test_local_model_%j.out
#SBATCH --error=logs/test_local_model_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# 设置环境变量
export HF_HOME=/ibex/user/wuj0c/cache/HF
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "本地 HuggingFace 模型测试"
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
echo "步骤 1: 测试 GPU 可用性..."
python -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 设备数: {torch.cuda.device_count()}')
    print(f'当前设备: {torch.cuda.current_device()}')
    print(f'设备名称: {torch.cuda.get_device_name(0)}')
    print(f'显存大小: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
"

echo ""
echo "步骤 2: 测试 HuggingFace 模型加载..."
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = 'Qwen/Qwen2.5-7B-Instruct'
print(f'加载模型: {model_name}')

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print('✓ Tokenizer 加载成功')
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    print('✓ 模型加载成功')
    print(f'模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B')
except Exception as e:
    print(f'✗ 加载失败: {e}')
    exit(1)
"

echo ""
echo "步骤 3: 测试 Outlines 集成..."
python -c "
import outlines
import transformers
from pydantic import BaseModel
import torch

class TestOutput(BaseModel):
    reasoning: str
    answer: str

try:
    model_name = 'Qwen/Qwen2.5-7B-Instruct'
    print(f'加载 Tokenizer...')
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print('✓ Tokenizer 加载成功')
    
    print(f'加载模型...')
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map='cuda'
    )
    print('✓ 模型加载成功')
    
    print(f'使用 Outlines 包装模型...')
    model = outlines.from_transformers(hf_model, tokenizer)
    print('✓ Outlines 模型加载成功')
    
except Exception as e:
    print(f'✗ Outlines 加载失败: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

echo ""
echo "步骤 4: 测试评估脚本 (小规模测试)..."
uv run python scripts/run_eval_ragtruth.py \
  --llm-type local \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --device cuda \
  --task-types Summary \
  --sample 2 \
  --temperature 0.0 \
  --outdir outputs/test_local_model

echo ""
echo "=========================================="
echo "测试完成: $(date)"
echo "=========================================="
