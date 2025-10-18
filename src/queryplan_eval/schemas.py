from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict

class Plan(BaseModel):
    """查询计划"""
    domain: str
    sub: Optional[str] = None
    is_personal: bool
    time: Optional[str] = None
    food: Optional[str] = None


class QueryResult(BaseModel):
    """统一的查询结果格式，避免 Union 类型以兼容 Qwen API
    
    模型可以选择：
    - 生成计划列表：plans 字段包含 Plan 对象列表，refused=False
    - 拒绝回答：refused=True，refuse_reason 包含拒绝原因，plans 为空
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "plans": [
                {
                    "domain": "health",
                    "sub": "exercise",
                    "is_personal": True,
                    "time": "morning",
                    "food": "breakfast"
                }
            ],
            "refused": False,
            "refuse_reason": None
        }
    })
    
    plans: List[Plan] = Field(
        default_factory=list,
        description="成功生成的计划列表，拒绝时为空列表"
    )
    
    refused: bool = Field(
        default=False, 
        description="是否拒绝回答查询"
    )
    
    refuse_reason: str = Field(
        default="",
        description="拒绝的原因（refused=True 时填充，否则为空字符串）"
    )


def normalize_result(obj: QueryResult) -> dict:
    """将 QueryResult 对象规范化为字典格式以便序列化
    
    Args:
        obj: QueryResult 对象
        
    Returns:
        规范化后的字典，格式统一便于下游处理
    """
    if obj.refused:
        return {"refuse": True, "reason": obj.refuse_reason}
    else:
        return {"plans": [p.model_dump() for p in obj.plans]}
