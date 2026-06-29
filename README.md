# qwen-0.5b-lab

一个围绕 Qwen 0.5B 级别小模型构建的 LLM 工程实验项目。目标不是从零训练大模型，而是在单卡、低显存或 CPU 资源有限的条件下，把「推理 -> 数据 -> 训练/微调 -> 评估 -> vLLM 部署 -> 压测」这条链路跑通。

当前仓库还处于初始化阶段，只有 README。下面的目录和命令是项目要补齐的目标形态，后续实现时应优先保证每一步都能被单独运行和验证。

## 项目目标

- 能跑：本地加载基座模型并完成 baseline inference。
- 能改：用小数据集做 SFT/LoRA/QLoRA 微调，产出 adapter 或合并后的模型。
- 能训：明确训练边界，支持继续预训练或全量 SFT 的实验入口，但不把从零预训练作为主目标。
- 能部署：用 vLLM 暴露 OpenAI-compatible API，也保留 FastAPI 包装层作为教学版本。
- 能测：对比原始模型、adapter 模型和合并模型的效果与吞吐。

## 推荐模型

默认使用 Hugging Face 上的 Qwen 小模型，例如：

```bash
export BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct
```

如果本地硬件更宽裕，可以把 `BASE_MODEL` 换成 `Qwen/Qwen2.5-1.5B-Instruct` 做同一套流程验证。

## 训练、微调和部署闭环

```text
准备环境
  -> 下载/缓存基座模型
  -> baseline 推理
  -> 准备数据集
  -> 选择训练路线
       A. LoRA / QLoRA 微调：主要路线，低成本产出 adapter
       B. 全量 SFT：对照路线，需要更多显存
       C. 继续预训练：可选路线，用领域语料补知识，不直接替代指令微调
  -> 评估与回归测试
  -> 导出产物
       A. adapter：轻量、便于多版本管理
       B. merged model：便于普通推理服务直接加载
  -> vLLM 部署
  -> benchmark 压测
```

## 项目结构

```text
qwen-0.5b-lab/
├── configs/
│   ├── model.yaml              # 模型名、精度、上下文长度等
│   ├── lora.yaml               # LoRA/QLoRA 参数
│   └── train.yaml              # 训练超参
├── data/
│   ├── raw/                    # 原始数据
│   ├── processed/              # 清洗后的 jsonl
│   └── examples/sft_demo.jsonl # 最小可跑样例
├── inference/
│   ├── chat.py                 # Transformers 本地推理
│   └── generate.py             # 单轮生成脚本
├── finetune/
│   ├── train_lora.py           # LoRA/QLoRA 微调入口
│   ├── train_sft_full.py       # 全量 SFT，可选
│   ├── train_continue.py       # 继续预训练，可选
│   └── merge_lora.py           # adapter 合并到基座模型
├── serving/
│   ├── app.py                  # FastAPI 教学版服务
│   └── vllm_client.py          # OpenAI-compatible 客户端测试
├── benchmark/
│   ├── run_bench.py            # 延迟/吞吐测试
│   └── eval_prompts.jsonl      # 固定评估问题
├── scripts/
│   ├── prepare_env.sh
│   ├── train_lora.sh
│   ├── serve_vllm_base.sh
│   ├── serve_vllm_lora.sh
│   └── bench_vllm.sh
├── requirements.txt
├── .env.example
└── README.md
```

## 环境准备

建议先把基础推理/微调环境和 vLLM 部署环境分开。vLLM 对 Python、CUDA、PyTorch 组合比较敏感，隔离环境能减少依赖冲突。

训练/微调环境：

```bash
conda create -n qwen-lab python=3.10 -y
conda activate qwen-lab
python -m pip install --upgrade pip
pip install torch transformers datasets accelerate peft bitsandbytes sentencepiece
pip install fastapi uvicorn openai
```

vLLM 服务环境，Linux / WSL2 + NVIDIA GPU 优先：

```bash
conda create -n qwen-vllm python=3.12 -y
conda activate qwen-vllm
python -m pip install --upgrade pip uv
uv pip install vllm --torch-backend=auto
```

说明：vLLM 官方快速开始文档当前推荐 Linux、Python 3.10-3.13，并建议用 `uv pip install vllm --torch-backend=auto` 自动匹配 PyTorch 后端。如果你当前是 Windows 原生环境，普通 Transformers 推理和 LoRA 微调可以先在 `qwen-lab` 环境里跑；vLLM 部署建议放到 WSL2 Ubuntu、Linux 服务器或 Docker Linux 容器中验证。

## 数据格式

SFT 数据建议统一成 JSONL，每行一个样本，先支持最小 chat 格式：

```json
{"messages":[{"role":"system","content":"你是一个严谨的运维助手。"},{"role":"user","content":"Nginx 502 常见原因有哪些？"},{"role":"assistant","content":"常见原因包括上游服务不可用、端口配置错误、超时、网关与后端网络不通等。"}]}
```

继续预训练数据则使用纯文本或拼接后的 `text` 字段：

```json
{"text":"这里是一段领域语料，用于继续预训练或领域语言建模。"}
```

数据处理阶段需要补齐：

- 去重、去空、长度过滤。
- train/validation/test 切分。
- 敏感信息清理。
- 固定一小份 smoke 数据，保证训练脚本能在几分钟内跑完。

## Baseline 推理

在训练前先跑原始模型，保存基线输出，后续才能判断微调是否真的有效。

```bash
python inference/chat.py \
  --model "$BASE_MODEL" \
  --prompt "请用三句话解释 LoRA 微调的作用。"
```

需要补齐的脚本能力：

- 支持 CPU/GPU 自动选择。
- 支持 `--max-new-tokens`、`--temperature`、`--stream`。
- 输出模型名、耗时、tokens/s。

## 训练路线

### 路线 A：LoRA / QLoRA 微调

这是本项目主路线，适合单卡低显存实验。

```bash
python finetune/train_lora.py \
  --model "$BASE_MODEL" \
  --train-file data/processed/train.jsonl \
  --eval-file data/processed/valid.jsonl \
  --output-dir outputs/lora/qwen-lab \
  --num-train-epochs 3 \
  --learning-rate 2e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --bf16
```

产物：

```text
outputs/lora/qwen-lab/
├── adapter_config.json
├── adapter_model.safetensors
└── tokenizer files
```

### 路线 B：全量 SFT

全量 SFT 会更新全部模型参数，显存和训练时间成本高于 LoRA。它适合作为对照实验，不建议作为第一阶段目标。

```bash
python finetune/train_sft_full.py \
  --model "$BASE_MODEL" \
  --train-file data/processed/train.jsonl \
  --eval-file data/processed/valid.jsonl \
  --output-dir outputs/full/qwen-lab
```

### 路线 C：继续预训练

继续预训练用于让模型吸收领域语料的语言分布和知识，不保证直接提升指令跟随。推荐流程是先继续预训练，再用 SFT/LoRA 做指令对齐。

```bash
python finetune/train_continue.py \
  --model "$BASE_MODEL" \
  --train-file data/processed/domain_text.jsonl \
  --output-dir outputs/continue/qwen-lab
```

## LoRA 合并

如果要用普通 Transformers 或某些部署方式直接加载完整模型，可以把 adapter 合并到基座模型：

```bash
python finetune/merge_lora.py \
  --base-model "$BASE_MODEL" \
  --adapter outputs/lora/qwen-lab \
  --output-dir outputs/merged/qwen-lab
```

如果使用 vLLM，也可以不合并，直接让 vLLM 挂载 LoRA adapter。

## vLLM 部署

vLLM 提供 OpenAI-compatible HTTP API，常用接口包括 `/v1/models`、`/v1/completions`、`/v1/chat/completions` 和 `/metrics`。

### 部署基座模型

```bash
conda activate qwen-vllm

vllm serve "$BASE_MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 2048 \
  --generation-config vllm
```

验证：

```bash
curl http://localhost:8000/v1/models

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "system", "content": "你是一个简洁的中文助手。"},
      {"role": "user", "content": "用两句话解释什么是 vLLM。"}
    ],
    "temperature": 0.2,
    "max_tokens": 128
  }'
```

### 部署 LoRA adapter

静态加载 adapter：

```bash
vllm serve "$BASE_MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-lora \
  --max-lora-rank 16 \
  --lora-modules qwen-lab=outputs/lora/qwen-lab
```

如果要保留基座模型血缘信息，可使用 vLLM 新的 JSON 格式：

```bash
vllm serve "$BASE_MODEL" \
  --enable-lora \
  --lora-modules '{"name":"qwen-lab","path":"outputs/lora/qwen-lab","base_model_name":"Qwen/Qwen2.5-0.5B-Instruct"}'
```

请求 adapter 时，`model` 字段使用 LoRA 名称：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-lab",
    "messages": [
      {"role": "user", "content": "按训练后的风格回答：Nginx 502 怎么排查？"}
    ],
    "max_tokens": 256
  }'
```

动态加载/卸载 LoRA adapter 适合本地开发验证，不建议直接作为生产管理方式：

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{"lora_name":"qwen-lab","lora_path":"outputs/lora/qwen-lab"}'

curl -X POST http://localhost:8000/v1/unload_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{"lora_name":"qwen-lab"}'
```

## Benchmark 和验收

每次训练后至少对比三类目标：

- 质量：固定 `benchmark/eval_prompts.jsonl`，人工或脚本对比回答完整性、事实错误、风格一致性。
- 性能：首 token 延迟、总延迟、tokens/s、并发吞吐。
- 稳定性：长输入、空输入、中文/英文混合输入、流式输出。

目标命令：

```bash
python benchmark/run_bench.py \
  --base-url http://localhost:8000/v1 \
  --model qwen-lab \
  --prompt-file benchmark/eval_prompts.jsonl \
  --concurrency 1,4,8 \
  --max-tokens 256
```

## 当前还缺什么

优先级从高到低：

1. `requirements.txt` 或 `pyproject.toml`：锁定训练环境依赖。
2. `data/examples/sft_demo.jsonl`：提供最小可跑训练样例。
3. `inference/chat.py`：先打通 baseline 推理。
4. `finetune/train_lora.py`：主训练入口，支持 LoRA/QLoRA。
5. `finetune/merge_lora.py`：导出 merged model。
6. `scripts/serve_vllm_base.sh` 和 `scripts/serve_vllm_lora.sh`：把 vLLM 命令固化。
7. `serving/vllm_client.py`：用 OpenAI SDK 调用本地 vLLM 服务。
8. `benchmark/run_bench.py`：统一延迟和吞吐指标。
9. `.env.example`：集中放 `BASE_MODEL`、`HF_HOME`、`CUDA_VISIBLE_DEVICES`、服务端口。
10. smoke test：用极小数据验证推理、训练、合并、vLLM 请求四步不会断。

## 参考文档

- vLLM Quickstart: https://docs.vllm.ai/en/latest/getting_started/quickstart/
- vLLM Online Serving: https://docs.vllm.ai/en/latest/serving/online_serving/
- vLLM LoRA Adapters: https://docs.vllm.ai/en/latest/features/lora/
