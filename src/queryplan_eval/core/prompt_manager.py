"""Prompt 管理器，按任务和版本组织"""

import logging
from pathlib import Path
from jinja2 import Template

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
