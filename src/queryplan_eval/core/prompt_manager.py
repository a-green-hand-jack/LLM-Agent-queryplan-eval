"""Prompt 管理器，按任务和版本组织"""

import logging
from pathlib import Path
from typing import Dict, Optional

import yaml
from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound

logger = logging.getLogger(__name__)


class PromptManager:
    """Prompt 管理器
    
    按任务和版本组织 prompt 文件，支持 Jinja2 模板渲染
    """
    
    def __init__(self, task_name: str, version: str = "latest"):
        """初始化 Prompt Manager
        
        Args:
            task_name: 任务名称（如 "query_plan", "judgement"）
            version: 版本号（如 "v6_cot", "latest" 表示选择最新版本）
        """
        self.task_name = task_name
        self.version = version
        self.prompts_root = Path(__file__).parent.parent / "prompts"
    
    def load(self, **render_vars) -> str:
        """加载并渲染 prompt
        
        Args:
            **render_vars: Jinja2 模板变量（如 today, domains 等）
            
        Returns:
            渲染后的 prompt 字符串
        """
        prompt_path = self._resolve_prompt_path()
        logger.info(f"加载 prompt: {prompt_path}")
        
        if prompt_path.suffix == ".j2":
            # Jinja2 模板
            text = prompt_path.read_text(encoding="utf-8")
            tpl = Template(text)
            return tpl.render(**render_vars)
        else:
            # 纯文本
            return prompt_path.read_text(encoding="utf-8")
    
    def _resolve_prompt_path(self) -> Path:
        """解析 prompt 文件路径
        
        Returns:
            找到的 prompt 文件路径
            
        Raises:
            FileNotFoundError: 如果找不到指定版本的 prompt
        """
        task_dir = self.prompts_root / self.task_name
        
        if not task_dir.exists():
            raise FileNotFoundError(f"任务目录不存在: {task_dir}")
        
        if self.version == "latest":
            # 自动选择最新版本（按文件名排序）
            candidates = sorted(task_dir.glob("*.j2")) + sorted(task_dir.glob("*.txt"))
            if not candidates:
                raise FileNotFoundError(f"未找到任务 {self.task_name} 的 prompt")
            logger.info(f"自动选择最新版本: {candidates[-1].name}")
            return candidates[-1]
        else:
            # 指定版本
            for ext in [".j2", ".txt"]:
                path = task_dir / f"{self.version}{ext}"
                if path.exists():
                    return path
            raise FileNotFoundError(f"未找到 {self.task_name}/{self.version}")


class RAGPromptManager:
    """RAGTruth 任务的 Prompt 管理器
    
    使用Jinja2模板引擎根据任务类型和是否使用CoT来选择和渲染对应的prompt模板
    
    支持三种任务类型（原始数据集命名）：
    - Summary: 摘要任务
    - QA: 问答任务  
    - Data2txt: 数据转文本任务
    
    支持两种推理模式：
    - with_cot: 使用Chain-of-Thought推理
    - without_cot: 不使用推理
    
    注意：本管理器直接使用原始任务类型命名，与 RAGTruthDataset 保持一致，
    不进行命名映射。
    
    Args:
        templates_dir: prompt模板文件目录（默认为 prompts/ragtruth/templates）
        config_file: prompt配置文件路径（默认为 prompts/ragtruth/prompt_config.yaml）
    
    Examples:
        >>> pm = RAGPromptManager()
        >>> # Summarization任务
        >>> prompt = pm.get_prompt(
        ...     task_type="Summary",
        ...     use_cot=True,
        ...     reference="原文内容...",
        ...     response="摘要内容..."
        ... )
        >>> # Question Answering任务
        >>> prompt = pm.get_prompt(
        ...     task_type="QA",
        ...     use_cot=True,
        ...     question="问题内容...",
        ...     reference="相关段落...",
        ...     response="答案内容..."
        ... )
        >>> # Data-to-Text任务
        >>> prompt = pm.get_prompt(
        ...     task_type="Data2txt",
        ...     use_cot=False,
        ...     reference='{\"key\": \"value\"}',
        ...     response="生成的文本..."
        ... )
    """
    
    def __init__(
        self,
        templates_dir: Optional[Path] = None,
        config_file: Optional[Path] = None,
    ):
        """初始化 RAGPromptManager
        
        Args:
            templates_dir: 模板目录路径
            config_file: 配置文件路径
        """
        # 使用默认路径
        if templates_dir is None:
            templates_dir = Path(__file__).parent.parent / "prompts" / "ragtruth" / "templates"
        
        if config_file is None:
            config_file = Path(__file__).parent.parent / "prompts" / "ragtruth" / "prompt_config.yaml"
        
        self.templates_dir = Path(templates_dir)
        self.config_file = Path(config_file)
        
        # 验证模板目录存在
        if not self.templates_dir.exists():
            error_msg = f"模板目录不存在: {self.templates_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # 加载配置
        self.config = self._load_config()
        
        # 初始化Jinja2环境
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )
        
        logger.info(f"RAGPromptManager初始化完成,模板目录: {self.templates_dir}")
    
    def _load_config(self) -> Dict:
        """加载prompt配置文件
        
        Returns:
            配置字典
        """
        if not self.config_file.exists():
            logger.warning(f"配置文件不存在: {self.config_file},使用默认配置")
            return self._get_default_config()
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"成功加载配置文件: {self.config_file}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e},使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """返回默认配置
        
        Returns:
            默认配置字典
        """
        return {
            "templates": {
                "summarization": {
                    "with_cot": {
                        "file": "summarization_with_cot.jinja2",
                        "variables": ["reference", "response"],
                    },
                    "without_cot": {
                        "file": "summarization_without_cot.jinja2",
                        "variables": ["reference", "response"],
                    },
                },
                "question_answering": {
                    "with_cot": {
                        "file": "qa_with_cot.jinja2",
                        "variables": ["question", "reference", "response"],
                    },
                    "without_cot": {
                        "file": "qa_without_cot.jinja2",
                        "variables": ["question", "reference", "response"],
                    },
                },
                "data_to_text": {
                    "with_cot": {
                        "file": "data_to_text_with_cot.jinja2",
                        "variables": ["reference", "response"],
                    },
                    "without_cot": {
                        "file": "data_to_text_without_cot.jinja2",
                        "variables": ["reference", "response"],
                    },
                },
            }
        }
    
    def _get_template_name(self, task_type: str, use_cot: bool) -> str:
        """根据任务类型和CoT选项获取模板文件名
        
        Args:
            task_type: 任务类型 ('Summary', 'QA', 'Data2txt')
            use_cot: 是否使用Chain-of-Thought
        
        Returns:
            模板文件名
        
        Raises:
            ValueError: 如果task_type不合法
            KeyError: 如果配置中缺少对应的模板
        """
        # 验证task_type（使用原始命名，与 RAGTruthDataset 保持一致）
        valid_tasks = ["Summary", "QA", "Data2txt"]
        if task_type not in valid_tasks:
            raise ValueError(f"task_type必须是{valid_tasks}之一,当前值: {task_type}")
        
        # 映射到内部配置的标准化名称
        task_mapping = {
            "Summary": "summarization",
            "QA": "question_answering",
            "Data2txt": "data_to_text"
        }
        normalized_task = task_mapping[task_type]
        
        # 选择模板配置
        cot_key = "with_cot" if use_cot else "without_cot"
        
        try:
            template_config = self.config["templates"][normalized_task][cot_key]
            template_name = template_config["file"]
            return template_name
        except KeyError as e:
            error_msg = f"配置中缺少模板定义: task_type={task_type}, use_cot={use_cot}, 错误: {e}"
            logger.error(error_msg)
            raise KeyError(error_msg)
    
    def _load_template(self, template_name: str) -> Template:
        """从文件加载Jinja2模板
        
        Args:
            template_name: 模板文件名
        
        Returns:
            Jinja2 Template对象
        
        Raises:
            FileNotFoundError: 如果模板文件不存在
        """
        try:
            template = self.jinja_env.get_template(template_name)
            logger.debug(f"成功加载模板: {template_name}")
            return template
        except TemplateNotFound:
            error_msg = f"模板文件不存在: {self.templates_dir / template_name}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
    
    def get_prompt(
        self,
        task_type: str,
        use_cot: bool = True,
        **kwargs
    ) -> str:
        """获取并渲染prompt模板
        
        Args:
            task_type: 任务类型 ('Summary', 'QA', 'Data2txt')
                      必须使用原始数据集命名，与 RAGTruthDataset 保持一致
            use_cot: 是否使用Chain-of-Thought推理
            **kwargs: 用于填充模板的变量
                - reference: 参考文本/数据（所有任务必需）
                - response: 模型生成的回复（所有任务必需）
                - question: 问题文本（仅QA任务需要）
        
        Returns:
            渲染后的prompt字符串
        
        Raises:
            ValueError: 如果task_type不合法或缺少必需的变量
            FileNotFoundError: 如果模板文件不存在
        
        Examples:
            >>> pm = RAGPromptManager()
            >>> # Summarization
            >>> prompt = pm.get_prompt(
            ...     task_type="Summary",
            ...     use_cot=True,
            ...     reference="原文...",
            ...     response="摘要..."
            ... )
            >>> # Question Answering
            >>> prompt = pm.get_prompt(
            ...     task_type="QA",
            ...     use_cot=False,
            ...     question="问题?",
            ...     reference="段落...",
            ...     response="答案..."
            ... )
            >>> # Data-to-Text
            >>> prompt = pm.get_prompt(
            ...     task_type="Data2txt",
            ...     use_cot=True,
            ...     reference='{\"key\": \"value\"}',
            ...     response="文本..."
            ... )
        """
        # 获取模板文件名
        template_name = self._get_template_name(task_type, use_cot)
        
        # 加载模板
        template = self._load_template(template_name)
        
        # 验证必需的变量
        self._validate_variables(task_type, kwargs)
        
        # 渲染模板
        try:
            prompt = template.render(**kwargs)
            logger.debug(f"成功渲染prompt: task={task_type}, use_cot={use_cot}")
            return prompt
        except Exception as e:
            error_msg = f"模板渲染失败: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _validate_variables(self, task_type: str, variables: Dict) -> None:
        """验证是否提供了所有必需的变量
        
        Args:
            task_type: 任务类型
            variables: 提供的变量字典
        
        Raises:
            ValueError: 如果缺少必需的变量
        """
        # 定义每个任务的必需变量（使用原始命名）
        required_vars = {
            "Summary": {"reference", "response"},
            "QA": {"question", "reference", "response"},
            "Data2txt": {"reference", "response"},
        }
        
        required = required_vars.get(task_type, set())
        provided = set(variables.keys())
        missing = required - provided
        
        if missing:
            error_msg = f"任务 {task_type} 缺少必需的变量: {missing}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.debug(f"变量验证通过: task={task_type}, variables={list(provided)}")
    
    def list_available_templates(self) -> Dict[str, Dict]:
        """列出所有可用的模板配置
        
        Returns:
            包含所有模板信息的字典
        """
        return self.config.get("templates", {})
