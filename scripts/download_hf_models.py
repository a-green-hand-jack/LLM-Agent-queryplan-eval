#!/usr/bin/env python3
"""
简单的 Hugging Face 模型下载脚本

使用方法:
    python download_hf_models.py <模型名称>
    
示例:
    python download_hf_models.py Qwen/Qwen2.5-14B-Instruct
    python download_hf_models.py microsoft/DialoGPT-small
    python download_hf_models.py distilbert-base-uncased
"""

import sys
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def download_model(model_name: str, download_only: bool = False) -> bool:
    """下载指定的 Hugging Face 模型
    
    Args:
        model_name: 模型名称，如 "Qwen/Qwen2.5-14B-Instruct"
        download_only: 是否仅下载不加载到内存
        
    Returns:
        bool: 是否下载成功
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        logging.info(f"📥 开始下载模型: {model_name}")
        
        # 下载分词器
        logging.info("📥 下载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logging.info(f"✅ 分词器下载完成 (词汇表大小: {tokenizer.vocab_size})")
        
        if download_only:
            logging.info("📥 仅下载模型权重...")
            # 仅下载模型权重，不加载到内存
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map=None,  # 不自动分配设备
                low_cpu_mem_usage=True
            )
            # 立即释放内存
            del model
            logging.info("✅ 模型权重下载完成")
        else:
            logging.info("📥 下载并加载模型...")
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # 简单测试
            logging.info("🧪 测试模型...")
            messages = [{"role": "user", "content": "Hello!"}]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            # 生成简短回复
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            outputs = model.generate(
                **inputs, 
                max_new_tokens=20,
                do_sample=False,
                temperature=0.7
            )
            
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:], 
                skip_special_tokens=True
            )
            logging.info(f"🤖 模型回复: {response}")
            
            # 释放内存
            del model
        
        logging.info(f"🎉 模型 {model_name} 下载完成！")
        return True
        
    except Exception as e:
        logging.error(f"❌ 下载失败: {e}")
        return False

def show_usage():
    """显示使用说明"""
    print("""
🤖 Hugging Face 模型下载工具

使用方法:
    python download_hf_models.py <模型名称> [选项]

示例:
    # 下载并测试模型
    python download_hf_models.py Qwen/Qwen2.5-14B-Instruct
    
    # 仅下载模型权重（不加载到内存）
    python download_hf_models.py microsoft/DialoGPT-small --download-only
    
    # 下载其他类型模型
    python download_hf_models.py distilbert-base-uncased

选项:
    --download-only    仅下载模型权重，不加载到内存（节省内存）
    --help, -h         显示此帮助信息

常用模型推荐:
    # 大语言模型
    Qwen/Qwen2.5-14B-Instruct
    Qwen/Qwen2.5-7B-Instruct
    microsoft/DialoGPT-small
    
    # 文本处理
    distilbert-base-uncased
    bert-base-uncased
    
    # 多模态
    microsoft/git-base
    """)

def main():
    """主函数"""
    if len(sys.argv) < 2:
        show_usage()
        return
    
    model_name = sys.argv[1]
    download_only = "--download-only" in sys.argv
    
    if model_name in ["--help", "-h"]:
        show_usage()
        return
    
    logging.info(f"🚀 开始处理模型: {model_name}")
    logging.info("📁 缓存位置: /ibex/user/wuj0c/cache/HF")
    
    success = download_model(model_name, download_only)
    
    if success:
        logging.info("✅ 任务完成！")
    else:
        logging.error("❌ 任务失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()