from __future__ import annotations
from typing import List, Optional, Union
from pydantic import BaseModel, Field

class Plan(BaseModel):
    domain: str
    sub: Optional[str] = None
    is_personal: bool
    time: Optional[str] = None
    food: Optional[str] = None

class Refuse(BaseModel):
    refuse: bool = Field(default=True)
    reason: str

# Outlines will use this union as JSON schema (anyOf)
ResultType = Union[List[Plan], Refuse]

def normalize_result(obj: Union[List[Plan], Refuse]):
    """Normalize to a single python dict for easy downstream use."""
    if isinstance(obj, list):
        return {"plans": [p.model_dump() for p in obj]}
    elif isinstance(obj, Refuse):
        return {"refuse": True, "reason": obj.reason}
    # If already a raw json string (provider dependent), caller will handle.
    return obj
