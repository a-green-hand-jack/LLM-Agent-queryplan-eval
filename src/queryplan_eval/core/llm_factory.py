"""LLM 工厂函数，用于创建不同类型的 LLM 实例"""

import os
import logging
from typing import Optional

from .base_llm import BaseLLM
from ..llms.openai_llm import OpenAILLM
from ..llms.huggingface_llm import HuggingFaceLLM

logger = logging.getLogger(__name__)


def create_llm(
    model: str,
    api_key: Optional[str] = None,
    llm_type: Optional[str] = None,
    base_url: Optional[str] = None,
    device: str = "cuda"
) -> BaseLLM:
    """创建 LLM 实例的工厂函数
    
    根据模型名称和类型自动选择对应的 LLM 实现。
    如果未指定 llm_type，则根据模型名称自动判断：
    - 包含 "/" 的模型名（如 "Qwen/Qwen2.5-14B-Instruct"）视为 HuggingFace 模型
    - 其他模型名视为 OpenAI API 模型
    
    Args:
        model: 模型名称或标识符
        api_key: API 密钥（仅在使用 OpenAI API 时需要）
        llm_type: LLM 类型，"openai" 或 "local"（可选，会自动判断）
        base_url: API 基础 URL（仅在使用 OpenAI API 时使用，默认从环境变量读取）
        device: 设备类型（仅在使用本地模型时使用，默认: "cuda"）
        
    Returns:
        初始化好的 LLM 实例
        
    Raises:
        ValueError: 当参数无效或缺少必要的配置时
    """
    # 自动判断 LLM 类型
    if llm_type is None:
        # HuggingFace 模型通常包含 "/"（如 "Qwen/Qwen2.5-14B-Instruct"）
        if "/" in model:
            llm_type = "local"
            logger.info(f"根据模型名称自动判断为本地模型: {model}")
        else:
            llm_type = "openai"
            logger.info(f"根据模型名称自动判断为 OpenAI API 模型: {model}")
    
    # 创建对应的 LLM 实例
    if llm_type == "openai":
        # OpenAI API 模式
        if api_key is None:
            api_key = os.environ.get("qwen_key") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "缺少 API 密钥。请通过参数传递 api_key 或在环境变量中设置 "
                    "qwen_key 或 OPENAI_API_KEY"
                )
        
        if base_url is None:
            base_url = os.environ.get(
                "QWEN_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        
        logger.info(f"初始化 OpenAI LLM: {model}")
        logger.info(f"Base URL: {base_url}")
        
        return OpenAILLM(
            model_name=model,
            base_url=base_url,
            api_key=api_key
        )
    
    elif llm_type == "local":
        # 本地 HuggingFace 模型
        logger.info(f"初始化本地 HuggingFace LLM: {model}")
        logger.info(f"设备: {device}")
        
        return HuggingFaceLLM(
            model_name=model,
            device=device
        )
    
    else:
        raise ValueError(
            f"无效的 llm_type: {llm_type}。"
            f"支持的值为 'openai' 或 'local'"
        )

