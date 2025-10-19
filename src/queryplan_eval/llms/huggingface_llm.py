"""HuggingFace 本地模型的 LLM 实现"""

import time
import json
import logging
from typing import Optional, Tuple, Type, TypeVar

import outlines
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer # type: ignore

from ..core.base_llm import BaseLLM

T = TypeVar('T', bound=BaseModel)
logger = logging.getLogger(__name__)


class HuggingFaceLLM(BaseLLM):
    """使用本地 HuggingFace Transformers 模型的实现
    
    通过 Outlines 提供结构化输出，与 OpenAI API 使用相同的逻辑
    """
    
    def __init__(self, model_path: str, device: str = "auto"):
        """初始化本地模型
        
        Args:
            model_path: 模型路径（本地路径或 HuggingFace Hub ID）
            device: 设备（"cuda", "cpu", "auto"）
        """
        self.model_path = model_path
        self.device = device

        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        try:
            # 使用 outlines 加载本地模型
            self.model = outlines.from_transformers(model, tokenizer, device_dtype=device)
            logger.info(f"已初始化 HuggingFace 本地模型: {model_path}")
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
        """使用 Outlines 生成结构化输出（与 OpenAI 逻辑一致）
        
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
            # 调用 outlines 模型（本地模型也使用相同的接口）
            result = self.model(
                outlines.inputs.Chat(chat),
                output_schema,
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
