"""HuggingFace 本地模型的 LLM 实现"""

import time
import json
import logging
from typing import Optional, Tuple, Type, TypeVar

import torch
import outlines
import transformers
from pydantic import BaseModel

from ..core.base_llm import BaseLLM

T = TypeVar('T', bound=BaseModel)
logger = logging.getLogger(__name__)


class HuggingFaceLLM(BaseLLM):
    """使用本地 HuggingFace Transformers 模型的实现
    
    通过 Outlines 提供结构化输出，与 OpenAI API 使用相同的逻辑
    """
    
    def __init__(self, model_name: str, device: str = "cuda"):
        """初始化本地模型
        
        Args:
            model_name: 模型名称（HuggingFace Hub ID，如 "Qwen/Qwen2.5-7B-Instruct"）
            device: 设备（"cuda", "cpu"）
        """
        self.model_name = model_name
        
        # 确定实际使用的设备
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA 不可用，将使用 CPU")
            self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"正在加载 HuggingFace 模型: {model_name} (device: {self.device})")
        
        try:
            # 显式加载 tokenizer 和模型
            logger.info("加载 Tokenizer...")
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            logger.info("加载模型...")
            # 先加载到 CPU，避免 device_map='auto' 导致的 torchvision 版本问题
            # 然后再移动到指定设备
            hf_model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="cpu"  # 先加载到 CPU
            )
            
            # 移动到指定设备
            if self.device == "cuda":
                logger.info(f"将模型移动到 {self.device}...")
                hf_model = hf_model.to(self.device)
            
            # 使用 outlines.from_transformers 包装模型
            logger.info("使用 Outlines 包装模型...")
            self.model = outlines.from_transformers(hf_model, tokenizer)
            
            logger.info(f"✓ 已初始化 HuggingFace 本地模型: {model_name}")
        except Exception as e:
            logger.error(f"初始化本地模型失败: {e}")
            raise
    
    def generate_structured(
        self, 
        chat: list[dict[str, str]], 
        output_schema: Type[T],
        temperature: float = 0.0,
        **kwargs
    ) -> Tuple[Optional[T], Optional[str], float]:
        """使用 Outlines 生成结构化输出
        
        Args:
            chat: 聊天消息列表
            output_schema: 输出类型（Pydantic 模型）
            temperature: 采样温度
            **kwargs: 其他参数
            
        Returns:
            (parsed_obj, raw_response, latency)
        """
        t0 = time.time()
        try:
            # 使用 outlines.Generator 生成结构化输出
            generator = outlines.Generator(self.model, output_schema)
            
            # 构建 chat 消息为提示文本
            prompt = self._format_chat_to_prompt(chat)
            
            # 生成结构化输出
            result = generator(
                prompt,
                max_new_tokens=1024,
                temperature=temperature
            )
            dt = time.time() - t0
            
            # 处理结果（与 OpenAI 完全一致的逻辑）
            parsed: Optional[T] = None
            if isinstance(result, str):
                raw = result
                try:
                    if hasattr(output_schema, "model_validate_json"):
                        parsed = output_schema.model_validate_json(raw)
                except Exception as e:
                    logger.debug(f"解析结构化输出失败: {e}")
                    parsed = None
            else:
                raw = json.dumps(
                    result.model_dump() if hasattr(result, 'model_dump') else result,
                    ensure_ascii=False
                )
                parsed = result
            
            return parsed, raw, dt
        except Exception as e:
            dt = time.time() - t0
            logger.error(f"调用本地模型失败: {e}")
            return None, None, dt
    
    def _format_chat_to_prompt(self, chat: list[dict[str, str]]) -> str:
        """将聊天消息列表格式化为提示文本
        
        Args:
            chat: 聊天消息列表 [{"role": "system", "content": "..."}, ...]
            
        Returns:
            格式化后的提示文本
        """
        prompt_parts = []
        for msg in chat:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt = "\n".join(prompt_parts)
        if chat and chat[-1].get("role") != "assistant":
            prompt += "\nAssistant:"
        
        return prompt
