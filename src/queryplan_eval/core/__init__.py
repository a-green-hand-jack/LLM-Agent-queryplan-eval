"""核心组件模块"""

from .base_llm import BaseLLM
from .base_task import BaseTask
from .prompt_manager import PromptManager, PatentPromptManager

__all__ = ["BaseLLM", "BaseTask", "PromptManager", "PatentPromptManager"]
