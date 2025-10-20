"""任务基类"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
from datasets import Dataset

from .base_llm import BaseLLM
from .prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class BaseTask(ABC):
    """任务基类
    
    每个任务类封装：
    1. 数据集加载和处理（包括分割）
    2. Chat 消息构建
    3. 单个样本的结果处理
    4. 任务指标计算
    5. 结果保存
    6. 评估流程编排
    """
    
    def __init__(
        self,
        data_path: str,
        llm: BaseLLM,
        prompt_manager: PromptManager,
        output_dir: str,
    ):
        """初始化任务
        
        Args:
            data_path: 数据文件路径
            llm: LLM 实例
            prompt_manager: Prompt 管理器实例
            output_dir: 输出目录
        """
        self.data_path = data_path
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据集
        self.dataset = self.load_dataset(data_path)
        logger.info(f"任务初始化完成，数据集大小: {len(self.dataset)}")
    
    @abstractmethod
    def load_dataset(self, path: str) -> Dataset:
        """加载 HuggingFace 风格的数据集
        
        Args:
            path: 数据文件路径
            
        Returns:
            加载后的 HuggingFace Dataset
        """
        pass
    
    def split_dataset(self, **split_config) -> Dict[str, Dataset]:
        """数据集分割方法（可选）
        
        Args:
            **split_config: 分割配置参数
            
        Returns:
            分割后的数据集字典
        """
        raise NotImplementedError("任务需要实现自己的 split_dataset 方法")
    
    @abstractmethod
    def build_chat(self, item: Any) -> list[dict[str, str]]:
        """构建 chat 消息列表
        
        Args:
            item: 数据集中的单个样本
            
        Returns:
            chat 消息列表 [{"role": "system", "content": "..."}, ...]
        """
        pass
    
    @abstractmethod
    def process_single_result(
        self, 
        item: Any, 
        parsed: Optional[Any], 
        raw: Optional[str], 
        latency: float
    ) -> Dict:
        """处理单个样本的结果
        
        Args:
            item: 原始样本
            parsed: 解析后的模型输出
            raw: 原始模型输出字符串
            latency: 执行耗时
            
        Returns:
            处理后的结果字典
        """
        pass
    
    @abstractmethod
    def compute_metrics(self, results: List[Dict]) -> Dict:
        """计算任务特定的评估指标
        
        Args:
            results: 所有样本的结果列表
            
        Returns:
            评估指标字典
        """
        pass
    
    @abstractmethod
    def save_results(self, results: List[Dict], metrics: Dict) -> None:
        """保存评估结果和指标
        
        Args:
            results: 所有样本的结果列表
            metrics: 评估指标
        """
        pass
    
    @abstractmethod
    def get_output_schema(self) -> type:
        """返回期望的输出 Pydantic 模型
        
        Returns:
            Pydantic 模型类
        """
        pass
    
    def run_evaluation(self, temperature: float = 0.0, **kwargs) -> Dict:
        """运行完整的评估流程（通用实现）
        
        Args:
            temperature: 采样温度
            **kwargs: 其他参数
            
        Returns:
            评估指标字典
        """
        from tqdm import tqdm
        
        logger.info(f"开始评估，数据集大小: {len(self.dataset)}")
        results = []
        
        for item in tqdm(self.dataset, desc="评估中"):
            chat = self.build_chat(item)
            parsed, raw, latency = self.llm.generate_structured(
                chat, 
                self.get_output_schema(),
                temperature=temperature
            )
            
            result = self.process_single_result(item, parsed, raw, latency)
            results.append(result)
        
        # 计算指标
        metrics = self.compute_metrics(results)
        
        # 保存结果
        self.save_results(results, metrics)
        
        logger.info(f"评估完成，结果已保存到: {self.output_dir}")
        return metrics
