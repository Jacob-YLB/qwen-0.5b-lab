# qwen-0.5b-lab

一个围绕 Qwen-0.5B-Instruct 小模型构建的完整实验项目，涵盖模型推理、轻量微调、本地部署和简单性能测试。

本项目用于学习和实践小模型在真实工程中的全流程应用。

## 项目目标

本项目目标不是训练大模型，而是在资源有限的情况下（单卡 / CPU）完成一个完整 LLM 工程闭环：

- 能跑：Inference
- 能改：Fine-tune
- 能服务：API deployment
- 能测：Benchmark

## 支持功能

### 模型推理

- 本地加载 Qwen-0.5B-Instruct
- 支持对话模式
- 支持 streaming 输出
- OpenAI-style API 封装（可选）

### 轻量微调（LoRA）

- 基于 HuggingFace Transformers
- 支持 LoRA / QLoRA
- 自定义小数据集训练
- 保存 adapter 权重

示例任务：

- 运维问答优化
- 指令跟随能力增强
- 特定领域风格调整

### 模型部署

- FastAPI 服务封装
- `/chat/completions` 接口
- Streaming 返回
- Docker 部署支持

### Benchmark 测试

- 单请求延迟
- 并发吞吐测试
- token/s 统计
- CPU / GPU 资源占用分析

## 系统架构

```text
用户请求
-> API Server（FastAPI）
-> Model Loader（Qwen-0.5B）
-> Tokenizer + Generation
-> Streaming Response
```

## 技术栈

- Python
- PyTorch
- Transformers
- PEFT（LoRA）
- FastAPI
- Docker（可选）

## 项目结构

```text
qwen-0.5b-lab/
├── inference/  # 推理代码
├── finetune/   # 微调代码（LoRA / QLoRA）
├── data/       # 训练数据
├── serving/    # API 服务
├── benchmark/  # 性能测试
├── scripts/    # 启动脚本
└── README.md
```

## 使用方式

### 推理

```bash
python inference/chat.py
```

### 启动服务

```bash
uvicorn serving.app:app --host 0.0.0.0 --port 8000
```

### 微调

```bash
python finetune/train_lora.py
```

### 压测

```bash
python benchmark/run_bench.py
```
