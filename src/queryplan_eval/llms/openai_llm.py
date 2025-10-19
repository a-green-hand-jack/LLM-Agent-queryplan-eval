"""OpenAI API 的 LLM 实现"""

import time
import json
import logging
from typing import Optional, Tuple, Type, TypeVar

import openai
import outlines
from pydantic import BaseModel

from ..core.base_llm import BaseLLM

T = TypeVar('T', bound=BaseModel)
logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """使用 Outlines + OpenAI API 的 LLM 实现
    
    通过 Outlines 提供结构化输出，与本地模型使用相同的逻辑
    """
    
    def __init__(self, model_name: str, base_url: str, api_key: str):
        """初始化 OpenAI LLM
        
        Args:
            model_name: 模型名称（如 "qwen-flash", "qwen3-max"）
            base_url: API 基础 URL
            api_key: API 密钥
        """
        self.model_name = model_name
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        # 使用 outlines 包装 OpenAI 客户端
        self.model = outlines.from_openai(self.client, model_name)
        logger.info(f"已初始化 OpenAI LLM: {model_name}")
    
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
            # 调用 outlines 模型
            result = self.model(
                outlines.inputs.Chat(chat),
                output_schema,
                temperature=temperature
            )
            dt = time.time() - t0
            
            # 处理结果（Outlines 可能返回字符串或对象）
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
            logger.error(f"调用 OpenAI LLM 失败: {e}")
            return None, None, dt
