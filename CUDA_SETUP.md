# CUDA 环境设置指南

本文档提供在项目中设置和使用 CUDA 环境的详细说明，适用于 CUDA 12.x 版本。

## 环境要求

- Python 3.9+ (推荐 3.10)
- CUDA 12.x (已在 CUDA 12.1 和 12.8 上测试)
- uv 包管理器

## 安装步骤

### 1. 创建并激活虚拟环境

```bash
# 在项目根目录下
uv venv
source .venv/bin/activate
```

### 2. 安装 PyTorch 和相关依赖

#### 使用 uv 安装 CUDA 版本的 PyTorch

```bash
# 安装 PyTorch CUDA 12.1+ 版本 (兼容 CUDA 12.8)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装 Transformers 和 Accelerate
uv pip install transformers accelerate
```

#### 使用 pyproject.toml 安装

项目的 `pyproject.toml` 已配置好 GPU 和 CPU 环境的可选依赖，可以通过以下命令安装：

```bash
# GPU 环境
uv pip install -e ".[gpu]" --index-url https://download.pytorch.org/whl/cu121

# CPU 环境
uv pip install -e ".[cpu]"
```

### 3. 验证安装

我们提供了一个测试脚本来验证环境设置：

```bash
python test_gpu_setup.py
```

在 GPU 计算节点上，应该能看到 CUDA 可用性、GPU 设备信息等输出。

## 在计算节点上使用

在登录节点上，`torch.cuda.is_available()` 通常会返回 `False`，因为登录节点没有 GPU。要在 GPU 上运行代码，需要提交到计算节点：

```bash
# 提交到计算节点的示例命令
sbatch run_on_gpu.sh
```

## 常见问题

### 1. CUDA 版本不匹配

如果遇到 CUDA 版本不匹配的错误，请确保安装的 PyTorch 版本与系统 CUDA 版本兼容。CUDA 12.1 版本的 PyTorch 通常兼容 CUDA 12.x 系列。

### 2. 内存不足错误

如果遇到 CUDA out of memory 错误，可以尝试：
- 减小批次大小 (batch size)
- 使用梯度累积 (gradient accumulation)
- 使用混合精度训练 (mixed precision training)

### 3. 找不到 CUDA 设备

确保在计算节点而不是登录节点上运行代码。可以使用 `nvidia-smi` 命令检查可用的 GPU 设备。

## 其他资源

- [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/)
- [Transformers 文档](https://huggingface.co/docs/transformers/index)
