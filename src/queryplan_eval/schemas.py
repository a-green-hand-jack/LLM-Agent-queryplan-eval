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


class ReasoningSteps(BaseModel):
    """CoT 推理步骤"""
    query_analysis: str = Field(
        description="分析用户问题的意图和关键信息"
    )
    domain_identification: str = Field(
        description="识别问题所属的领域（健康/运动/其他）"
    )
    time_calculation: Optional[str] = Field(
        default=None,
        description="如有时间相关内容，展示计算过程"
    )
    refuse_check: str = Field(
        description="检查是否应该拒答及原因"
    )
    final_decision: str = Field(
        description="最终决策及参数提取结果"
    )


class QueryResult(BaseModel):
    """统一的查询结果格式，避免 Union 类型以兼容 Qwen API
    
    模型可以选择：
    - 生成计划列表：plans 字段包含 Plan 对象列表，refused=False
    - 拒绝回答：refused=True，refuse_reason 包含拒绝原因，plans 为空
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "reasoning": {
                "query_analysis": "用户询问今天的睡眠情况",
                "domain_identification": "属于健康领域-睡眠场景",
                "time_calculation": "今天是2025年10月19日，查询今天的睡眠",
                "refuse_check": "不属于拒答情形，是正常的个人数据查询",
                "final_decision": "提取睡眠领域的个人查询计划"
            },
            "plans": [
                {
                    "domain": "睡眠",
                    "sub": None,
                    "is_personal": True,
                    "time": "2025年10月19日",
                    "food": None
                }
            ],
            "refused": False,
            "refuse_reason": ""
        }
    })
    
    reasoning: Optional[ReasoningSteps] = Field(
        default=None,
        description="Chain-of-Thought 推理过程（可选，启用 CoT 时填充）"
    )
    
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


def normalize_result(obj: QueryResult, include_reasoning: bool = False) -> dict:
    """将 QueryResult 对象规范化为字典格式以便序列化
    
    Args:
        obj: QueryResult 对象
        include_reasoning: 是否在输出中包含 reasoning 字段
        
    Returns:
        规范化后的字典，格式统一便于下游处理
    """
    result: dict = {}
    
    if obj.refused:
        result = {"refuse": True, "reason": obj.refuse_reason}
    else:
        result = {"plans": [p.model_dump() for p in obj.plans]}
    
    # 如果启用且存在 reasoning，添加到结果中
    if include_reasoning and obj.reasoning is not None:
        result["reasoning"] = obj.reasoning.model_dump()
    
    return result


class DimensionScores(BaseModel):
    """各维度评分"""
    structure_a: float = Field(ge=0, le=10, description="候选A的结构完整性得分")
    structure_b: float = Field(ge=0, le=10, description="候选B的结构完整性得分")
    semantic_a: float = Field(ge=0, le=10, description="候选A的语义准确性得分")
    semantic_b: float = Field(ge=0, le=10, description="候选B的语义准确性得分")
    completeness_a: float = Field(ge=0, le=10, description="候选A的信息完整度得分")
    completeness_b: float = Field(ge=0, le=10, description="候选B的信息完整度得分")
    format_a: float = Field(ge=0, le=10, description="候选A的格式规范性得分")
    format_b: float = Field(ge=0, le=10, description="候选B的格式规范性得分")


class JudgementResult(BaseModel):
    """LLM 判别结果"""
    winner: str = Field(
        description="获胜者: 'candidate_a', 'candidate_b', 或 'tie'"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="判断置信度，范围 0.0-1.0"
    )
    reason: str = Field(
        max_length=200,
        description="判断理由，不超过100字"
    )
    dimensions: DimensionScores = Field(
        description="各维度的详细评分"
    )


class HallucinationReasoning(BaseModel):
    """幻觉检测的 CoT 推理步骤
    
    支持三种任务类型的推理步骤：
    - Data-to-Text: content_analysis, reference_comparison, hallucination_identification, span_extraction, final_verdict
    - Q&A: question_analysis, answer_analysis, reference_verification, hallucination_identification, final_assessment  
    - Summarization: document_analysis, summary_analysis, fidelity_verification, hallucination_identification, final_evaluation
    """
    
    # Data-to-Text 任务推理步骤
    content_analysis: Optional[str] = Field(
        default=None,
        description="分析生成内容和识别关键声明（Data-to-Text 任务）"
    )
    reference_comparison: Optional[str] = Field(
        default=None,
        description="与参考数据进行系统性对比（Data-to-Text 任务）"
    )
    
    # Q&A 任务推理步骤
    question_analysis: Optional[str] = Field(
        default=None,
        description="分析问题并识别所需信息类型（Q&A 任务）"
    )
    answer_analysis: Optional[str] = Field(
        default=None,
        description="分解答案内容并识别所有事实声明（Q&A 任务）"
    )
    reference_verification: Optional[str] = Field(
        default=None,
        description="对照参考文档验证答案声明（Q&A 任务）"
    )
    
    # Summarization 任务推理步骤
    document_analysis: Optional[str] = Field(
        default=None,
        description="分析原始文档内容、关键事实和主题（Summarization 任务）"
    )
    summary_analysis: Optional[str] = Field(
        default=None,
        description="分解摘要内容并识别所有事实声明（Summarization 任务）"
    )
    fidelity_verification: Optional[str] = Field(
        default=None,
        description="验证摘要与文档的忠实度（Summarization 任务）"
    )
    
    # 通用推理步骤（所有任务共享）
    hallucination_identification: str = Field(
        description="识别特定的矛盾、无支持或误表的内容"
    )
    
    # 任务特定的最终步骤
    span_extraction: Optional[str] = Field(
        default=None,
        description="提取精确的幻觉片段（Data-to-Text 任务）"
    )
    final_assessment: Optional[str] = Field(
        default=None,
        description="提供答案忠实度的整体评估（Q&A 任务）"
    )
    final_evaluation: Optional[str] = Field(
        default=None,
        description="提供摘要准确性的整体评估（Summarization 任务）"
    )
    final_verdict: Optional[str] = Field(
        default=None,
        description="总结发现并提供最终评估（Data-to-Text 任务）"
    )


class HallucinationResult(BaseModel):
    """幻觉检测结果
    
    模型输出的幻觉片段列表，每个片段必须是 response 中的精确子串。
    如果没有检测到幻觉，hallucination_list 为空列表。
    支持 Chain-of-Thought 推理过程。
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "reasoning": {
                "content_analysis": "Article claims 15% revenue increase and expansion to 5 countries",
                "reference_comparison": "Reference shows 10% revenue growth and 3 countries expansion",
                "hallucination_identification": "Two numerical contradictions: 15% vs 10% revenue, 5 vs 3 countries",
                "span_extraction": "Extracting exact contradictory spans: '15%' and '5 new countries'",
                "final_verdict": "Article contains factual inaccuracies in key metrics"
            },
            "hallucination_list": ["15%", "5 new countries"]
        }
    })
    
    reasoning: Optional[HallucinationReasoning] = Field(
        default=None,
        description="Chain-of-Thought 推理过程（可选，启用 CoT 时填充）"
    )
    
    hallucination_list: List[str] = Field(
        default_factory=list,
        description="检测到的幻觉片段列表，每个片段必须是 response 中的精确子串"
    )


def normalize_hallucination_result(obj: HallucinationResult, include_reasoning: bool = False) -> dict:
    """将 HallucinationResult 对象规范化为字典格式以便序列化
    
    Args:
        obj: HallucinationResult 对象
        include_reasoning: 是否在输出中包含 reasoning 字段
        
    Returns:
        规范化后的字典，格式统一便于下游处理
    """
    result: dict = {
        "hallucination_list": obj.hallucination_list
    }
    
    # 如果启用且存在 reasoning，添加到结果中
    if include_reasoning and obj.reasoning is not None:
        result["reasoning"] = obj.reasoning.model_dump()
    
    return result
