<!-- 82c47de7-9598-4c17-beb6-d7ca05218287 0d92d17a-6113-4dd1-ae30-5571ac328ba7 -->
# 实现 LLM 类型选择功能

## 需要修改的文件

### 1. 修复 `src/queryplan_eval/llms/huggingface_llm.py`

**问题**：`outlines.from_transformers()` API 调用可能不正确，且没有正确处理 HF_HOME 环境变量。

**修复内容**：

- 正确使用 `outlines` 库的 API 来加载本地 transformers 模型
- 支持从 HF_HOME 缓存目录加载预下载的模型
- 参考 OpenAILLM 的实现模式，确保结构化输出兼容

### 2. 更新 `src/queryplan_eval/llms/__init__.py`

**修改内容**：

- 导出 HuggingFaceLLM 类（当前被注释）

### 3. 更新 `scripts/run_eval_ragtruth.py`

**添加命令行参数**：

- `--llm-type` (choices: "openai", "local", default: "openai")
- `--model-name` (default: "Qwen/Qwen2.5-7B-Instruct") - 用于指定本地模型
- `--device` (default: "cuda") - 用于指定本地模型的设备

**修改初始化逻辑**：

- 根据 `--llm-type` 参数选择实例化 OpenAILLM 或 HuggingFaceLLM
- 为 HuggingFaceLLM 传递设备参数
- 增加配置日志输出，显示所选的 LLM 类型和设备信息

## 实现步骤

1. 修复 HuggingFaceLLM 的 outlines 调用
2. 更新 **init**.py 导出
3. 在 run_eval_ragtruth.py 中添加 --llm-type 等参数
4. 实现条件逻辑选择 LLM 实现

## 关键注意事项

- 使用 HF_HOME 环境变量（已配置为 /ibex/user/wuj0c/cache/HF）
- 确保两个 LLM 实现返回格式一致（遵循 BaseLLM 接口）
- 本地模型设备支持 "cuda" 和 "cpu"，自动验证 CUDA 可用性

### To-dos

- [ ] 修复 huggingface_llm.py 中的 outlines API 调用，正确加载本地 transformers 模型和处理设备指定
- [ ] 更新 src/queryplan_eval/llms/__init__.py，导出 HuggingFaceLLM 类
- [ ] 在 run_eval_ragtruth.py 中添加 --llm-type、--model-name 和 --device 命令行参数
- [ ] 在 run_eval_ragtruth.py 中实现基于 --llm-type 参数的 LLM 实例化逻辑
- [ ] 增强日志输出，显示选定的 LLM 类型和设备信息