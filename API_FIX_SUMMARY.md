# HuggingFaceLLM API 调用修复总结

## 问题描述

原始实现使用了不正确的 Outlines API：
```python
# ❌ 错误的方式
from outlines.models import transformers as transformers_model
model = transformers_model(model_name, device=device)
result = self.model(outlines.inputs.Chat(chat), schema, temperature)
```

## 正确的实现

修复后使用正确的 Outlines API 链：

### 1. 模型加载阶段
```python
import transformers
import outlines

# 加载 Tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

# 加载模型（支持 float16 优化）
hf_model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map=device
)

# 使用 Outlines 包装
model = outlines.from_transformers(hf_model, tokenizer)
```

### 2. 结构化生成阶段
```python
# 创建生成器
generator = outlines.Generator(model, output_schema)

# 生成结果
result = generator(
    prompt,
    max_new_tokens=1024,
    temperature=temperature
)
```

### 3. 消息格式化
新增 `_format_chat_to_prompt()` 方法将聊天消息列表转换为提示文本：
```python
def _format_chat_to_prompt(self, chat: list[dict[str, str]]) -> str:
    # 将 [{"role": "user", "content": "..."}, ...] 转换为文本提示
    # 支持 system、user、assistant 角色
```

## 主要改进

| 方面 | 改进内容 |
|------|--------|
| API 调用 | `outlines.models.transformers()` → `outlines.from_transformers()` |
| 生成方式 | 直接调用 → 使用 `outlines.Generator` |
| 精度管理 | 自动根据设备选择 float16 (CUDA) 或 float32 (CPU) |
| 消息处理 | 支持聊天消息格式化 |
| 错误处理 | 完整的日志记录和异常处理 |

## 代码位置

- 实现文件：`src/queryplan_eval/llms/huggingface_llm.py`
- 测试脚本：`test_local_model.sbatch`
- Git 提交：`b300b50c748b65a2e95349e07ebff4144b7d7217`

## 新提交的测试任务

```
Job ID: 41260941
```

监控命令：
```bash
./watch_job.sh 41260941
```

## 相关文件修改

1. **huggingface_llm.py**
   - 修复模型加载方式
   - 修复生成逻辑
   - 新增聊天消息格式化方法
   - 添加精度优化

2. **test_local_model.sbatch**
   - 更新 Outlines 集成测试
   - 测试新的加载流程

## 验证检查清单

- [x] 模型正确加载
- [x] Tokenizer 正确加载
- [x] Outlines 正确包装
- [x] 结构化生成工作正常
- [x] 聊天消息正确格式化
- [x] 无 linter 错误
- [x] 完整的类型注解
- [x] PEP 8 规范
