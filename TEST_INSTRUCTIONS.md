# 本地模型测试说明

## 任务提交

已提交 sbatch 任务来测试本地模型功能：

```bash
Job ID: 41260834
```

## 监控任务进度

使用提供的监控脚本：

```bash
./watch_job.sh 41260834
```

或者手动查看状态：

```bash
# 查看任务状态
squeue -j 41260834

# 查看输出日志
tail -f logs/test_local_model_41260834.out

# 查看错误日志
tail -f logs/test_local_model_41260834.err
```

## 测试步骤

sbatch 脚本会执行以下步骤：

### 1. GPU 可用性测试
- 检查 PyTorch CUDA 支持
- 显示 GPU 设备信息和显存大小

### 2. HuggingFace 模型加载测试
- 加载 Qwen/Qwen2.5-7B-Instruct 模型
- 检查 Tokenizer 和模型是否成功加载
- 显示模型参数量

### 3. Outlines 集成测试
- 使用 Outlines 加载本地 transformers 模型
- 验证结构化输出的 Pydantic 集成

### 4. 完整评估脚本测试
- 运行实际的评估脚本
- 使用本地模型处理 2 个 Summary 任务样本
- 输出到 `outputs/test_local_model/`

## 预期结果

测试成功时应该看到：

```
✓ Tokenizer 加载成功
✓ 模型加载成功
✓ Outlines 模型加载成功
✓ 评估完成
```

## 常见问题

### 如果任务未能分配 GPU
检查可用的 GPU 资源：
```bash
sinfo -N -o "%N %t %G"
```

### 如果 HuggingFace 模型下载失败
确保 HF_HOME 环境变量已正确设置：
```bash
echo $HF_HOME
# 应该显示: /ibex/user/wuj0c/cache/HF
```

### 查看完整的错误信息
```bash
cat logs/test_local_model_41260834.err
```

## 下一步

如果测试成功，可以运行完整的评估：

```bash
# 使用本地模型
uv run python scripts/run_eval_ragtruth.py \
  --llm-type local \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --device cuda \
  --sample 100

# 与 OpenAI API 对比
uv run python scripts/run_eval_ragtruth.py \
  --llm-type openai \
  --model qwen-flash \
  --sample 100
```
