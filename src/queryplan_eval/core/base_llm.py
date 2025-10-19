"""LLM 抽象基类，统一 API 和本地模型的接口"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Type, TypeVar
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)


class BaseLLM(ABC):
    """LLM 抽象基类
    
    统一 API 和本地模型的调用接口，都使用 Outlines 提供结构化输出
    """
    
    @abstractmethod
    def generate_structured(
        self, 
        chat: list[dict[str, str]], 
        output_schema: Type[T],
        temperature: float = 0.0,
        **kwargs
    ) -> Tuple[Optional[T], Optional[str], float]:
        """生成结构化输出
        
        Args:
            chat: 聊天消息列表，格式 [{"role": "system", "content": "..."}, ...]
            output_schema: 期望的输出类型（Pydantic 模型）
            temperature: 采样温度
            **kwargs: 其他模型特定参数
            
        Returns:
            (parsed_obj, raw_response, latency)
            - parsed_obj: 解析后的对象实例，解析失败时为 None
            - raw_response: 原始返回字符串
            - latency: 执行耗时（秒）
        """
        pass
