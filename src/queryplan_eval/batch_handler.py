"""批量处理模块，支持 OpenAI 兼容的 Batch API。

该模块提供三个核心类用于批量调用 LLM API：
- BatchRequestBuilder: 构建符合 OpenAI Batch 格式的 JSONL 请求
- BatchResponseProcessor: 处理和解析批量响应
- BatchExecutor: 协调整个批处理流程
"""

from __future__ import annotations

import json
import time
import logging
from pathlib import Path
from typing import Any, Optional, Type, TypeVar, List, Tuple
from dataclasses import dataclass, field
import tempfile

import openai
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class BatchRequest:
    """单个批处理请求项"""
    custom_id: str
    method: str = "POST"
    url: str = "/v1/chat/completions"
    body: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "custom_id": self.custom_id,
            "method": self.method,
            "url": self.url,
            "body": self.body,
        }


@dataclass
class BatchResult:
    """批处理单个结果项"""
    custom_id: str
    status_code: int
    request_id: Optional[str] = None
    body: Optional[dict] = None
    error: Optional[dict] = None


class BatchRequestBuilder:
    """构建符合 OpenAI Batch 格式的 JSONL 请求文件"""
    
    def __init__(self, model: str) -> None:
        """初始化请求构建器
        
        Args:
            model: 模型名称（例如 'qwen-plus'）
        """
        self.model = model
        self.requests: List[BatchRequest] = []
    
    def add_request(
        self,
        custom_id: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        extra_params: Optional[dict[str, Any]] = None,
    ) -> None:
        """添加一个聊天完成请求
        
        Args:
            custom_id: 唯一请求标识符
            messages: 聊天消息列表
            temperature: 采样温度
            extra_params: 其他 API 参数
        """
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if extra_params:
            body.update(extra_params)
        
        request = BatchRequest(
            custom_id=custom_id,
            body=body,
        )
        self.requests.append(request)
    
    def build_jsonl_string(self) -> str:
        """将请求列表转换为 JSONL 格式字符串
        
        Returns:
            JSONL 格式的字符串（每行一个 JSON 对象）
        """
        lines = [json.dumps(req.to_dict(), ensure_ascii=False) for req in self.requests]
        return "\n".join(lines)
    
    def save_to_file(self, filepath: Path | str) -> Path:
        """将请求保存到文件
        
        Args:
            filepath: 目标文件路径
            
        Returns:
            保存的文件路径
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        content = self.build_jsonl_string()
        filepath.write_text(content, encoding="utf-8")
        
        logger.info(f"批处理请求已保存到: {filepath} ({len(self.requests)} 条请求)")
        return filepath
    
    def clear(self) -> None:
        """清空所有请求"""
        self.requests.clear()
    
    def __len__(self) -> int:
        """返回请求数量"""
        return len(self.requests)


class BatchResponseProcessor:
    """处理和解析批量响应"""
    
    @staticmethod
    def parse_response_jsonl(content: str) -> List[BatchResult]:
        """解析响应 JSONL 内容
        
        Args:
            content: JSONL 格式的响应内容
            
        Returns:
            解析后的 BatchResult 列表
        """
        results = []
        for line in content.strip().split("\n"):
            if not line:
                continue
            
            try:
                data = json.loads(line)
                result = BatchResult(
                    custom_id=data.get("custom_id", ""),
                    status_code=data.get("status_code", 0),
                    request_id=data.get("request_id"),
                    body=data.get("body"),
                    error=data.get("error"),
                )
                results.append(result)
            except json.JSONDecodeError as e:
                logger.error(f"无法解析响应行: {line[:100]}... 错误: {e}")
                continue
        
        return results
    
    @staticmethod
    def extract_structured_output(
        result: BatchResult,
        output_type: Type[T],
    ) -> Tuple[Optional[T], Optional[str], Optional[str]]:
        """从响应中提取结构化输出
        
        Args:
            result: 批处理结果项
            output_type: 期望的输出类型（Pydantic 模型）
            
        Returns:
            元组包含：
            - parsed: 解析后的对象实例，如果解析失败则为 None
            - raw: 原始返回字符串
            - error: 错误信息（如果有）
        """
        if result.error:
            error_msg = result.error.get("message", str(result.error))
            return None, None, error_msg
        
        if result.status_code != 200:
            return None, None, f"HTTP {result.status_code}"
        
        if not result.body:
            return None, None, "Empty response body"
        
        try:
            # 从响应体中提取 choices[0].message.content
            choices = result.body.get("choices", [])
            if not choices:
                return None, None, "No choices in response"
            
            message_content = choices[0].get("message", {}).get("content")
            if not message_content:
                return None, None, "No message content in response"
            
            # 尝试解析为 JSON 并转换为 Pydantic 对象
            if isinstance(message_content, str):
                raw = message_content
                # 尝试使用 Pydantic 的 model_validate_json 方法
                if hasattr(output_type, "model_validate_json"):
                    parsed = output_type.model_validate_json(raw)
                else:
                    # 如果不支持 model_validate_json，尝试解析 JSON 后用 model_validate
                    data = json.loads(raw)
                    parsed = output_type.model_validate(data)
            else:
                raw = json.dumps(message_content, ensure_ascii=False)
                parsed = output_type.model_validate(message_content)
            
            return parsed, raw, None
        
        except Exception as e:
            error_msg = f"解析失败: {str(e)}"
            raw = json.dumps(result.body, ensure_ascii=False) if result.body else None
            return None, raw, error_msg


class BatchExecutor:
    """协调批处理执行流程"""
    
    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        batch_size: int = 100,
        poll_interval: float = 10.0,
        max_wait_time: float = 1800.0,  # 30 分钟
    ) -> None:
        """初始化批处理执行器
        
        Args:
            client: OpenAI 兼容客户端
            model: 模型名称
            batch_size: 每个批次的大小
            poll_interval: 轮询间隔（秒）
            max_wait_time: 最大等待时间（秒）
        """
        self.client = client
        self.model = model
        self.batch_size = batch_size
        self.poll_interval = poll_interval
        self.max_wait_time = max_wait_time
        self.builder = BatchRequestBuilder(model)
    
    def submit_batch(self, requests: List[BatchRequest]) -> str:
        """提交一批请求到 API
        
        Args:
            requests: 请求列表
            
        Returns:
            批处理任务 ID
        """
        self.builder.requests = requests
        
        # 创建临时文件保存请求
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".jsonl",
            delete=False,
            encoding="utf-8",
        ) as f:
            temp_path = Path(f.name)
            f.write(self.builder.build_jsonl_string())
        
        try:
            # 上传文件
            logger.info(f"上传请求文件: {temp_path}")
            with open(temp_path, "rb") as f:
                response = self.client.files.create(
                    file=f,
                    purpose="batch",
                )
            
            file_id = response.id
            logger.info(f"文件已上传，ID: {file_id}")
            
            # 提交批处理任务
            batch_response = self.client.batches.create(
                input_file_id=file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            
            batch_id = batch_response.id
            logger.info(f"批处理任务已提交，ID: {batch_id}")
            
            return batch_id
        
        finally:
            # 清理临时文件
            temp_path.unlink(missing_ok=True)
    
    def wait_for_completion(self, batch_id: str) -> bool:
        """等待批处理任务完成
        
        Args:
            batch_id: 批处理任务 ID
            
        Returns:
            任务是否成功完成
        """
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > self.max_wait_time:
                logger.error(f"等待超时，已等待 {elapsed:.1f} 秒")
                return False
            
            try:
                batch = self.client.batches.retrieve(batch_id)
                
                status = batch.status
                logger.info(
                    f"批处理状态: {status} "
                    f"(已处理: {batch.request_counts.completed}/{batch.request_counts.total})"
                )
                
                if status == "completed":
                    logger.info(f"批处理完成，成功: {batch.request_counts.succeeded}, 失败: {batch.request_counts.failed}")
                    return True
                elif status == "failed":
                    logger.error("批处理失败")
                    return False
                elif status == "expired":
                    logger.error("批处理过期")
                    return False
                
                # 等待后继续轮询
                time.sleep(self.poll_interval)
            
            except Exception as e:
                logger.error(f"轮询失败: {e}")
                return False
    
    def retrieve_results(self, batch_id: str) -> Optional[List[BatchResult]]:
        """检索批处理结果
        
        Args:
            batch_id: 批处理任务 ID
            
        Returns:
            解析后的批处理结果列表，失败时返回 None
        """
        try:
            batch = self.client.batches.retrieve(batch_id)
            
            if not batch.output_file_id:
                logger.error("未找到输出文件")
                return None
            
            logger.info(f"检索输出文件: {batch.output_file_id}")
            
            # 下载结果文件
            response = self.client.files.content(batch.output_file_id)
            content = response.text
            
            # 解析结果
            results = BatchResponseProcessor.parse_response_jsonl(content)
            logger.info(f"已解析 {len(results)} 条结果")
            
            return results
        
        except Exception as e:
            logger.error(f"检索结果失败: {e}")
            return None
    
    def execute_batch(
        self,
        requests: List[BatchRequest],
    ) -> Optional[List[BatchResult]]:
        """执行一个完整的批处理流程
        
        Args:
            requests: 请求列表
            
        Returns:
            解析后的结果列表，失败时返回 None
        """
        logger.info(f"开始批处理，共 {len(requests)} 条请求")
        
        # 提交批处理
        batch_id = self.submit_batch(requests)
        
        # 等待完成
        if not self.wait_for_completion(batch_id):
            return None
        
        # 检索结果
        results = self.retrieve_results(batch_id)
        
        return results


def batch_split(items: List[Any], batch_size: int) -> List[List[Any]]:
    """将项目列表分割成多个批次
    
    Args:
        items: 项目列表
        batch_size: 每个批次的大小
        
    Returns:
        按批次分割的项目列表
    """
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i : i + batch_size])
    return batches
