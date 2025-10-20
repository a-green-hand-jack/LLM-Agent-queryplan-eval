#!/usr/bin/env python3
"""
GPU 环境测试脚本

用于验证 PyTorch 和 Transformers 在 GPU 计算节点上的安装是否正常。

运行方式：
    python test_gpu_setup.py

预期输出（在 GPU 节点上）：
- PyTorch 版本信息
- CUDA 可用性和版本
- GPU 设备数量和名称
- Transformers 基本功能测试
"""

import torch
import transformers
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_pytorch_setup() -> None:
    """测试 PyTorch 安装和 CUDA 支持"""
    logging.info("=== PyTorch 环境测试 ===")
    
    # 基本版本信息
    logging.info(f"PyTorch 版本: {torch.__version__}")
    logging.info(f"CUDA 是否可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logging.info(f"CUDA 版本: {torch.version.cuda}")
        logging.info(f"cuDNN 版本: {torch.backends.cudnn.version()}")
        logging.info(f"GPU 设备数量: {torch.cuda.device_count()}")
        
        # 列出所有 GPU 设备
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            logging.info(f"GPU {i}: {gpu_name}")
            
        # 测试基本 GPU 操作
        device = torch.device("cuda:0")
        test_tensor = torch.randn(1000, 1000, device=device)
        result = torch.mm(test_tensor, test_tensor.t())
        logging.info(f"GPU 矩阵乘法测试成功，结果形状: {result.shape}")
        
    else:
        logging.warning("CUDA 不可用，可能在登录节点或 CPU 计算节点上")

def test_transformers_setup() -> None:
    """测试 Transformers 安装和基本功能"""
    logging.info("=== Transformers 环境测试 ===")
    
    # 版本信息
    logging.info(f"Transformers 版本: {transformers.__version__}")
    
    # 测试基本功能 - 加载一个小型模型的配置
    try:
        from transformers import AutoConfig
        
        # 测试加载配置（不下载模型权重）
        config = AutoConfig.from_pretrained("distilbert-base-uncased")
        logging.info(f"成功加载模型配置: {config.model_type}")
        
        # 如果有 GPU，测试设备设置
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            logging.info(f"GPU 设备设置测试: {device}")
            
    except Exception as e:
        logging.error(f"Transformers 功能测试失败: {e}")

def test_integration() -> None:
    """测试 PyTorch 和 Transformers 的集成"""
    logging.info("=== 集成测试 ===")
    
    try:
        from transformers import AutoTokenizer
        
        # 测试分词器
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        test_text = "Hello, this is a test sentence for tokenization."
        tokens = tokenizer(test_text, return_tensors="pt")
        
        logging.info(f"分词测试成功，token 数量: {tokens['input_ids'].shape[1]}")
        
        # 如果有 GPU，测试张量转移
        if torch.cuda.is_available():
            tokens_gpu = {k: v.cuda() for k, v in tokens.items()}
            logging.info(f"张量 GPU 转移测试成功")
            
    except Exception as e:
        logging.error(f"集成测试失败: {e}")

def main() -> None:
    """主测试函数"""
    logging.info("开始 GPU 环境测试...")
    
    test_pytorch_setup()
    test_transformers_setup()
    test_integration()
    
    logging.info("测试完成！")

if __name__ == "__main__":
    main()
