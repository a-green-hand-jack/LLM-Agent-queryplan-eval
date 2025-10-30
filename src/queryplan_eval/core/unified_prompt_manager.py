"""统一的 Prompt Manager - 使用生产级 prompt-manager 库

该模块提供了一个统一的接口来管理所有任务的 prompts，
使用生产级的 prompt-manager 库进行底层实现。

支持的任务：
- query_plan: 查询计划提取
- ragtruth: RAGTruth 幻觉检测
- patent: 专利肽段表示转换
- judgement: 查询计划质量评估

Usage:
    >>> from queryplan_eval.core.unified_prompt_manager import UnifiedPromptManager
    >>>
    >>> # 初始化
    >>> manager = UnifiedPromptManager()
    >>>
    >>> # Query Plan 任务
    >>> messages = manager.render_query_plan(
    ...     version="v6_cot",
    ...     today="2025年10月30日"
    ... )
    >>>
    >>> # RAGTruth 任务
    >>> messages = manager.render_ragtruth(
    ...     task_type="Summary",
    ...     use_cot=True,
    ...     reference="原文...",
    ...     response="摘要..."
    ... )
    >>>
    >>> # Patent 任务
    >>> messages = manager.render_patent(
    ...     version="v2",
    ...     sequence="Ala Ser Lys",
    ...     features=[...]
    ... )
    >>>
    >>> # Judgement 任务
    >>> messages = manager.render_judgement(
    ...     question="...",
    ...     gold_standard={...},
    ...     candidate_a={...},
    ...     candidate_b={...}
    ... )
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class UnifiedPromptManager:
    """统一的 Prompt Manager

    使用生产级 prompt-manager 库管理所有任务的 prompts。
    提供简化的任务特定接口，同时保持底层的灵活性。

    Args:
        prompts_dir: prompts 目录路径（默认为项目根目录下的 prompts/）
        dev_mode: 开发模式，启用热重载（默认 False）
        enable_cache: 启用缓存以提升性能（默认 True）

    Attributes:
        manager: 底层的 PromptManager 实例
    """

    def __init__(
        self,
        prompts_dir: Optional[Path] = None,
        dev_mode: bool = False,
        enable_cache: bool = True,
    ):
        """初始化 UnifiedPromptManager

        Args:
            prompts_dir: prompts 根目录路径
            dev_mode: 开发模式，启用热重载
            enable_cache: 启用缓存以提升性能
        """
        # 默认使用项目根目录下的 prompts/
        if prompts_dir is None:
            # 从当前文件往上找 3 层到项目根目录
            project_root = Path(__file__).parent.parent.parent.parent
            prompts_dir = project_root / "prompts"

        self.prompts_dir = Path(prompts_dir)

        # 验证目录存在
        if not self.prompts_dir.exists():
            raise FileNotFoundError(
                f"Prompts 目录不存在: {self.prompts_dir}\n"
                f"请确保已正确设置 prompt manager 目录结构"
            )

        # 为每个任务创建独立的 PromptManager 实例
        self.managers = {}
        self.dev_mode = dev_mode
        self.enable_cache = enable_cache

        # 初始化各个任务的 manager
        for task in ["query_plan", "ragtruth", "patent", "judgement"]:
            task_dir = self.prompts_dir / task
            if task_dir.exists():
                self.managers[task] = PromptManager(
                    prompts_dir=str(task_dir),
                    dev_mode=dev_mode,
                    enable_cache=enable_cache,
                )
                logger.info(f"初始化 {task} PromptManager: {task_dir}")
            else:
                logger.warning(f"任务目录不存在，跳过: {task_dir}")

        if not self.managers:
            raise FileNotFoundError(
                f"未找到任何任务目录在: {self.prompts_dir}\n"
                f"请确保至少有一个任务目录（query_plan, ragtruth, patent, judgement）"
            )

        logger.info(f"UnifiedPromptManager 初始化完成，prompts 目录: {self.prompts_dir}")
        logger.info(f"已加载的任务: {list(self.managers.keys())}")
        logger.info(f"开发模式: {dev_mode}, 缓存: {enable_cache}")

    # ============ Query Plan 任务 ============

    def render_query_plan(
        self,
        version: str = "v6_cot",
        today: Optional[str] = None,
        domains: Optional[Dict] = None,
        subs_map: Optional[Dict] = None,
        **kwargs
    ) -> List[Dict[str, str]]:
        """渲染 Query Plan 提示词

        Args:
            version: 版本号（v5, v6_cot, original, plan_data_*）
            today: 当前日期，格式如 "2025年10月30日"
            domains: 自定义 domain 枚举
            subs_map: 自定义 sub 枚举映射
            **kwargs: 其他参数

        Returns:
            OpenAI 格式的消息列表

        Example:
            >>> messages = manager.render_query_plan(
            ...     version="v6_cot",
            ...     today="2025年10月30日"
            ... )
        """
        # 渲染 system prompt
        system_content = self.managers["query_plan"].render(
            prompt_name="query_plan",
            template_type="system",
            version=version,
            today=today,
            domains=domains,
            subs_map=subs_map,
            **kwargs
        )

        # 返回 OpenAI 格式的消息列表
        return [{"role": "system", "content": system_content}]

    def get_query_plan_llm_config(self) -> Dict[str, Any]:
        """获取 Query Plan 的 LLM 配置"""
        return self.managers["query_plan"].get_llm_config("query_plan")

    # ============ RAGTruth 任务 ============

    def render_ragtruth(
        self,
        task_type: str,
        use_cot: bool = True,
        reference: Optional[str] = None,
        response: Optional[str] = None,
        question: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, str]]:
        """渲染 RAGTruth 提示词

        Args:
            task_type: 任务类型（'Summary', 'QA', 'Data2txt'）
            use_cot: 是否使用 Chain-of-Thought 推理
            reference: 参考内容
            response: 模型生成的回复
            question: 问题（仅 QA 任务需要）
            **kwargs: 其他参数

        Returns:
            OpenAI 格式的消息列表

        Example:
            >>> messages = manager.render_ragtruth(
            ...     task_type="Summary",
            ...     use_cot=True,
            ...     reference="原文内容...",
            ...     response="摘要内容..."
            ... )
        """
        # 根据 task_type 和 use_cot 确定版本
        version_map = {
            ("Summary", True): "summarization_with_cot",
            ("Summary", False): "summarization_without_cot",
            ("QA", True): "qa_with_cot",
            ("QA", False): "qa_without_cot",
            ("Data2txt", True): "data_to_text_with_cot",
            ("Data2txt", False): "data_to_text_without_cot",
        }

        version = version_map.get((task_type, use_cot))
        if version is None:
            raise ValueError(
                f"不支持的任务类型和 CoT 组合: task_type={task_type}, use_cot={use_cot}\n"
                f"task_type 必须是 'Summary', 'QA', 'Data2txt' 之一"
            )

        # 渲染 system prompt
        system_content = self.managers["ragtruth"].render(
            prompt_name="ragtruth",
            template_type="system",
            version=version,
            task_type=task_type,
            use_cot=use_cot,
            reference=reference,
            response=response,
            question=question,
            **kwargs
        )

        # 返回 OpenAI 格式的消息列表
        return [{"role": "system", "content": system_content}]

    def get_ragtruth_llm_config(self) -> Dict[str, Any]:
        """获取 RAGTruth 的 LLM 配置"""
        return self.managers["ragtruth"].get_llm_config("ragtruth")

    # ============ Patent 任务 ============

    def render_patent(
        self,
        version: str = "v2",
        sequence: Optional[str] = None,
        features: Optional[List] = None,
        peptide_json: Optional[Dict] = None,
        **kwargs
    ) -> List[Dict[str, str]]:
        """渲染 Patent 提示词

        Args:
            version: 版本号（v1, v2）
            sequence: 由空格分隔的三字母残基序列
            features: 修饰信息列表
            peptide_json: 完整的肽段 JSON 对象
            **kwargs: 其他参数

        Returns:
            OpenAI 格式的消息列表

        Example:
            >>> messages = manager.render_patent(
            ...     version="v2",
            ...     sequence="Ala Ser Lys",
            ...     features=[...]
            ... )
        """
        # 渲染 system prompt
        system_content = self.managers["patent"].render(
            prompt_name="patent",
            template_type="system",
            version=version,
            sequence=sequence,
            features=features,
            peptide_json=peptide_json,
            **kwargs
        )

        # 返回 OpenAI 格式的消息列表
        return [{"role": "system", "content": system_content}]

    def get_patent_llm_config(self) -> Dict[str, Any]:
        """获取 Patent 的 LLM 配置"""
        return self.managers["patent"].get_llm_config("patent")

    # ============ Judgement 任务 ============

    def render_judgement(
        self,
        version: str = "v1",
        question: Optional[str] = None,
        gold_standard: Optional[Dict] = None,
        candidate_a: Optional[Dict] = None,
        candidate_b: Optional[Dict] = None,
        evaluation_criteria: Optional[Dict] = None,
        **kwargs
    ) -> List[Dict[str, str]]:
        """渲染 Judgement 提示词

        Args:
            version: 版本号（v1）
            question: 用户的原始问题
            gold_standard: 金标准答案
            candidate_a: 候选A 的模型输出
            candidate_b: 候选B 的模型输出
            evaluation_criteria: 自定义评估标准
            **kwargs: 其他参数

        Returns:
            OpenAI 格式的消息列表

        Example:
            >>> messages = manager.render_judgement(
            ...     question="...",
            ...     gold_standard={...},
            ...     candidate_a={...},
            ...     candidate_b={...}
            ... )
        """
        # 渲染 system prompt
        system_content = self.managers["judgement"].render(
            prompt_name="judgement",
            template_type="system",
            version=version,
            question=question,
            gold_standard=gold_standard,
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            evaluation_criteria=evaluation_criteria,
            **kwargs
        )

        # 返回 OpenAI 格式的消息列表
        return [{"role": "system", "content": system_content}]

    def get_judgement_llm_config(self) -> Dict[str, Any]:
        """获取 Judgement 的 LLM 配置"""
        return self.managers["judgement"].get_llm_config("judgement")

    # ============ 通用方法 ============

    def render_messages(
        self,
        task_name: str,
        prompt_name: str,
        version: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, str]]:
        """通用的消息渲染方法

        Args:
            task_name: 任务名称（query_plan, ragtruth, patent, judgement）
            prompt_name: prompt 名称
            version: 版本号
            **kwargs: 模板参数

        Returns:
            OpenAI 格式的消息列表
        """
        if task_name not in self.managers:
            raise ValueError(f"未知的任务: {task_name}. 可用任务: {list(self.managers.keys())}")

        return self.managers[task_name].render_messages(
            prompt_name=prompt_name,
            version=version,
            **kwargs
        )

    def get_llm_config(self, task_name: str, prompt_name: str) -> Dict[str, Any]:
        """获取指定 prompt 的 LLM 配置

        Args:
            task_name: 任务名称
            prompt_name: prompt 名称

        Returns:
            LLM 配置字典
        """
        if task_name not in self.managers:
            raise ValueError(f"未知的任务: {task_name}. 可用任务: {list(self.managers.keys())}")

        return self.managers[task_name].get_llm_config(prompt_name)

    def list_prompts(self) -> Dict[str, List[str]]:
        """列出所有可用的 prompt 名称（按任务分组）"""
        result = {}
        for task_name, manager in self.managers.items():
            result[task_name] = manager.list_prompts()
        return result

    def list_versions(self, task_name: str, template_type: str = "system") -> List[str]:
        """列出指定任务的所有可用版本

        Args:
            task_name: 任务名称
            template_type: 模板类型（system/user）

        Returns:
            版本列表
        """
        if task_name not in self.managers:
            raise ValueError(f"未知的任务: {task_name}. 可用任务: {list(self.managers.keys())}")

        return self.managers[task_name].list_versions(template_type)

    def cache_stats(self) -> Dict[str, Any]:
        """获取所有任务的缓存统计信息"""
        stats = {}
        for task_name, manager in self.managers.items():
            stats[task_name] = manager.cache_stats()
        return stats

    def clear_cache(self):
        """清除所有任务的缓存"""
        for task_name, manager in self.managers.items():
            manager.clear_cache()
        logger.info("所有任务的缓存已清除")

    def reload(self):
        """重新加载所有任务的配置和模板（开发模式下自动完成）"""
        for task_name, manager in self.managers.items():
            manager.reload()
        logger.info("所有任务的配置和模板已重新加载")


# 全局单例（可选）
_global_manager: Optional[UnifiedPromptManager] = None


def get_unified_manager(
    prompts_dir: Optional[Path] = None,
    dev_mode: bool = False,
    enable_cache: bool = True,
) -> UnifiedPromptManager:
    """获取全局 UnifiedPromptManager 单例

    Args:
        prompts_dir: prompts 目录路径
        dev_mode: 开发模式
        enable_cache: 启用缓存

    Returns:
        UnifiedPromptManager 实例
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = UnifiedPromptManager(
            prompts_dir=prompts_dir,
            dev_mode=dev_mode,
            enable_cache=enable_cache,
        )
    return _global_manager


def reset_unified_manager():
    """重置全局 UnifiedPromptManager 单例"""
    global _global_manager
    _global_manager = None
