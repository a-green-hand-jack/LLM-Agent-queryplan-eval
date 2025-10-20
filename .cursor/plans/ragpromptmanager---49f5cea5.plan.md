<!-- 49f5cea5-08b6-4bcc-bb1d-3c0596fd0110 876ab8c6-8006-4930-8ea7-5ac16b2255fb -->
# RAGPromptManager 重构计划

## 目标

将 `prompts/ragtruth/prompt_manager.py` 的功能迁移到 `core/prompt_manager.py`，使 RAGTruth 的 prompt 管理器与项目统一架构对齐，同时保持其现有的丰富功能。

## 核心变更

### 1. 在 core/prompt_manager.py 中添加 RAGPromptManager 类

在 `src/queryplan_eval/core/prompt_manager.py` 末尾添加新类：

```python
class RAGPromptManager:
    """RAGTruth 任务的 Prompt 管理器
    
    支持三种任务类型（summarization, question_answering, data_to_text）
    和两种推理模式（with_cot, without_cot）的组合
    
    Args:
        templates_dir: 模板目录路径（默认为 prompts/ragtruth/templates）
        config_file: 配置文件路径（默认为 prompts/ragtruth/prompt_config.yaml）
    
    Examples:
        >>> pm = RAGPromptManager()
        >>> # Summarization任务
        >>> prompt = pm.get_prompt(
        ...     task_type="summarization",
        ...     use_cot=True,
        ...     reference="原文内容...",
        ...     response="摘要内容..."
        ... )
    """
    
    def __init__(self, templates_dir: Optional[Path] = None, config_file: Optional[Path] = None):
        # 实现内容从 prompts/ragtruth/prompt_manager.py 迁移
        pass
    
    def _load_config(self) -> Dict:
        # 迁移原有逻辑
        pass
    
    def _get_default_config(self) -> Dict:
        # 迁移原有逻辑
        pass
    
    def _get_template_name(self, task_type: str, use_cot: bool) -> str:
        # 迁移原有逻辑
        pass
    
    def _load_template(self, template_name: str) -> Template:
        # 迁移原有逻辑
        pass
    
    def get_prompt(self, task_type: str, use_cot: bool = True, **kwargs) -> str:
        # 迁移原有逻辑
        pass
    
    def _validate_variables(self, task_type: str, variables: Dict) -> None:
        # 迁移原有逻辑
        pass
    
    def list_available_templates(self) -> Dict[str, Dict]:
        # 迁移原有逻辑
        pass
```

**关键调整点：**

- 默认的 `templates_dir` 改为 `Path(__file__).parent.parent / "prompts" / "ragtruth" / "templates"`
- 默认的 `config_file` 改为 `Path(__file__).parent.parent / "prompts" / "ragtruth" / "prompt_config.yaml"`
- 保持所有现有方法和接口不变

### 2. 更新 core/prompt_manager.py 的导入

在文件顶部添加必要的导入：

```python
from typing import Dict, Optional
import yaml
from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound
```

### 3. 更新 prompts/ragtruth/**init**.py

修改导入路径，指向新位置：

```python
"""
Prompt模板管理模块

负责加载和渲染不同任务的prompt模板
"""

from queryplan_eval.core.prompt_manager import RAGPromptManager

__all__ = ["RAGPromptManager"]
```

### 4. 删除旧文件

删除 `src/queryplan_eval/prompts/ragtruth/prompt_manager.py`

### 5. 验证迁移后的目录结构

迁移后 `prompts/ragtruth/` 目录应包含：

- `__init__.py` - 导出 RAGPromptManager
- `prompt_config.yaml` - 配置文件
- `templates/` - 6个 jinja2 模板文件

## 实现顺序

1. 在 `core/prompt_manager.py` 中添加 `RAGPromptManager` 类（迁移所有功能）
2. 更新 `prompts/ragtruth/__init__.py` 的导入
3. 删除 `prompts/ragtruth/prompt_manager.py`
4. 运行测试验证迁移正确性

## 需要注意的测试影响

如果 `tests/test_ragtruth.py` 或其他测试文件中有导入 `PromptManager`，需要更新为：

```python
from queryplan_eval.prompts.ragtruth import RAGPromptManager
# 或
from queryplan_eval.core.prompt_manager import RAGPromptManager
```

### To-dos

- [ ] 在 core/prompt_manager.py 中添加 RAGPromptManager 类，迁移所有功能和方法
- [ ] 更新 prompts/ragtruth/__init__.py 导入新的 RAGPromptManager
- [ ] 删除 prompts/ragtruth/prompt_manager.py
- [ ] 检查并更新所有引用旧 PromptManager 的代码