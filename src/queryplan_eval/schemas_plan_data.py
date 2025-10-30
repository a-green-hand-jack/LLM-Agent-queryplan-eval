"""计划数据任务输出模型 - PlanData 专用结果格式

支持四种模式输出：
- single: type, domain, sub, is_personal, time, food
- multi: time_frame, domain, sub, is_personal, time, food
- single_think: domain, sub, is_personal, time, food, think
- multi_think: domain, sub, is_personal, time, food, think
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class PlanBase(BaseModel):
    """通用计划字段"""
    domain: str = Field(description="问题所属的主题场景")
    sub: Optional[str] = Field(default=None, description="主题的子场景（可选）")
    is_personal: bool = Field(description="是否需要查询个人数据")
    time: Optional[str] = Field(default=None, description="查询时间区间（仅当 is_personal=true 时）")
    food: Optional[str] = Field(default=None, description="提到的具体食物名称")


class PlanDataSingleResult(BaseModel):
    """Single 模式输出 - 仅输入 query 和 date，输出包含 type"""
    type: str = Field(
        description="用户查询的时间分类（如：本周、上个月、几周前、昨天、今天、上周、本月、本年、去年、近期、无时间）"
    )
    domain: str
    sub: Optional[str] = None
    is_personal: bool
    time: Optional[str] = None
    food: Optional[str] = None


class PlanDataMultiResult(BaseModel):
    """Multi 模式输出 - 输入 history、query、date，输出包含 time_frame"""
    time_frame: str = Field(
        description="用户查询的时间分类（如：本周、上个月、几周前、昨天、今天、上周、本月、本年、去年、近期、无时间）"
    )
    domain: str
    sub: Optional[str] = None
    is_personal: bool
    time: Optional[str] = None
    food: Optional[str] = None


class PlanDataSingleThinkResult(BaseModel):
    """SingleThink 模式输出 - 仅输入 query 和 date，包含模型推理过程 think"""
    domain: str
    sub: Optional[str] = None
    is_personal: bool
    time: Optional[str] = None
    food: Optional[str] = None
    think: str = Field(
        description="模型的 Chain-of-Thought 推理过程，用于与标注的真实 think 进行对比"
    )


class PlanDataMultiThinkResult(BaseModel):
    """MultiThink 模式输出 - 输入 history、query、date，包含模型推理过程 think"""
    domain: str
    sub: Optional[str] = None
    is_personal: bool
    time: Optional[str] = None
    food: Optional[str] = None
    think: str = Field(
        description="模型的 Chain-of-Thought 推理过程，用于与标注的真实 think 进行对比"
    )


# Union type for any plan data result
PlanDataResult = PlanDataSingleResult | PlanDataMultiResult | PlanDataSingleThinkResult | PlanDataMultiThinkResult


def normalize_plan_data_result(
    obj: PlanDataResult,
    include_reasoning: bool = False
) -> dict:
    """将 PlanDataResult 对象转换为字典格式以便序列化
    
    Args:
        obj: PlanDataResult 对象（可能是四种模型之一）
        include_reasoning: 是否包含 think 字段（若存在）
    
    Returns:
        规范化后的字典
    """
    result = obj.model_dump(exclude_none=False)
    return result
