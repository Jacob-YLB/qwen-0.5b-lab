# 踩坑记录（TROUBLESHOOTING）

> 随着项目推进持续更新。每条尽量给「症状 → 原因 → 处理命令」。
> 推理阶段的概念性易错点见 [docs/01-inference.md §3](docs/01-inference.md)，本篇偏「环境和报错」。

---

## 0. 第一件事：看懂报错

- 先看 **完整的 Traceback 最后一行**，那通常是真正的原因（`RuntimeError: CUDA out of memory`、`KeyError: 'Qwen2ForCausalLM'`、`OSError: ... timeout`）。
- 中间的栈只是调用链，新手容易被中间一堆 frames 吓到，其实多数信息无关。
- 报错搜之前，把里面的**模型名 / 路径 / 个人路径**去掉再搜，命中率更高。

---

## 1. 环境与安装

### 1.1 `KeyError: 'Qwen2ForCausalLM'` / 提示要 `trust_remote_code`
- **原因**：transformers 版本太旧，不认识 Qwen2.5 的架构。
- **处理**：升级到 `transformers>=4.44`。
  ```bash
  pip install -U "transformers>=4.44.0"
  python -c "import transformers; print(transformers.__version__)"
  ```
- 说明：Qwen2.5 已合入主干，**不需要** `trust_remote_code=True`；老教程让你开它，现在多数是过时建议，开了反而有安全风险。

### 1.2 `torch.cuda.is_available()` 返回 `False`（明明有 N 卡）
- **原因**：torch 版本和本机 CUDA 不匹配，或装成了 CPU 版 torch。
- **处理**：
  ```bash
  python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
  ```
  - 如果 `torch.version.cuda` 是 `None`，说明装了 CPU 版，重装带 CUDA 的 torch（见 [pytorch.org](https://pytorch.org/get-started/locally/) 选对应 CUDA 的 pip 命令）。
  - 如果 torch 自带 CUDA 与驱动差距太大，也可能不可用（见 1.3）。

### 1.3 WSL2 里看不到 / 用不了 GPU（本项目重点）
WSL2 跑 GPU 的正确姿势，和原生 Linux **不一样**，很多人卡在这里：
- **驱动**：装 **NVIDIA 的 Windows 版驱动**（Game Ready / Studio 驱动即可），**不要**在 WSL 里装 CUDA Toolkit。
- **验证**：在 WSL2 里能跑通 `nvidia-smi` 看到显存，就说明直通成功。
  ```bash
  nvidia-smi          # 看到你的显卡 + Driver Version + CUDA Version（这是驱动支持的最高 CUDA）
  ```
- **torch**：WSL2 里照常用 Linux 的 CUDA 版 torch（`pip install torch` 默认就带 CUDA），不需要特殊版本。
- **CUDA 版本怎么看**：`nvidia-smi` 右上角的 CUDA Version 是「驱动支持的上限」，不是「已安装的 CUDA Toolkit」。WSL 跑 PyTorch 不需要单独装 toolkit。

### 1.4 `bitsandbytes` 装不上 / import 报错
- **原因**：bitsandbytes（QLoRA 用）只支持 **Linux + NVIDIA GPU**，Windows / Mac 原生装不上。
- **处理**：现阶段 baseline 推理**用不到**它，先忽略；等做 QLoRA 时在 WSL2/Linux 环境里再装。

### 1.5 vLLM 环境：Python 版本 / torch 后端冲突
- vLLM 对 Python、CUDA、torch 组合敏感，所以**单独建 `qwen-vllm` 环境**（见 README）。
- 推荐用官方：`uv pip install vllm --torch-backend=auto`，自动匹配 torch 后端，别手动硬拼版本。

---

## 2. 模型下载

### 2.1 下载卡住 / `HTTPSConnectionPool ... timed out` / 连不上 huggingface.co
- **原因**：国内直连 HF 经常超时。
- **处理**：用镜像站（最常用）：
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  python inference/chat.py --prompt "..."      # 在同一 shell 里跑
  ```
  或写进 `.env` / `~/.bashrc`。也可以把 `.env.example` 里那行 `HF_ENDPOINT` 取消注释。

### 2.2 想提前下载 / 离线使用
- **处理**：用 `huggingface-cli` 显式拉到本地：
  ```bash
  pip install -U "huggingface_hub[cli]"
  hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/qwen25-0.5b
  ```
  之后 `--model models/qwen25-0.5b` 直接走本地，不再联网。

### 2.3 下载到一半中断，缓存损坏 / `OSError: ... is not a valid ...`
- **处理**：清掉这块缓存重下：
  ```bash
  rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct
  ```
  （或你自定义的 `HF_HOME` 下对应目录）。

### 2.4 `401 Unauthorized` / `403 Forbidden` / `Gated repo`
- **原因**：模型是 gated（需要申请）或要登录。
- **处理**：`Qwen/Qwen2.5-0.5B-Instruct` **不是 gated**，一般不会遇到；若换别的模型遇到，去模型页申请权限，再 `huggingface-cli login` 用 token 登录。

### 2.5 磁盘空间不足
- **原因**：模型权重 + 多版本缓存会占几 GB。
- **处理**：把缓存挪到大盘：`export HF_HOME=/path/to/bigdisk/hf_cache`；定期清旧模型缓存。

---

## 3. 显存与设备

### 3.1 `CUDA out of memory`
- **原因**：权重 + KV cache + 激活超过显存。
- **按从轻到重的顺序处理**：
  1. 确认用的是 `bfloat16`（脚本在 GPU 上默认就是）。
  2. 调小 `--max-new-tokens` 和输入长度（KV cache 随长度线性增长）。
  3. 确认没在训练模式（`model.eval()` + `@torch.inference_mode()`）。
  4. 确认没有遗留进程占显存：
     ```bash
     nvidia-smi            # 看 GPU 上还有谁
     # 必要时清掉卡死的 python 进程
     ```
  5. 实在不够：CPU 跑（`--device cpu`，0.5B 可接受，就是慢），或换更小模型。

### 3.2 0.5B 到底吃多少显存？（建立直觉）
- **权重**：bf16 下 0.5B × 2 字节 ≈ **1GB**；fp32 ≈ 2GB。
- **KV cache**：随「输入长度 + 生成长度」线性增长，短问答可忽略，长文本会显著增加。
- **结论**：短上下文推理 2~3GB 显存足够；别因为「才 0.5B」就以为几百 MB 够用。

### 3.3 CPU 推理慢得无法接受
- **原因**：CPU 没有张量核心、又是 fp32，本来就慢。
- **处理**：0.5B 在 CPU 上几十 tokens/s 量级，跑通 demo 没问题；要压测/部署必须上 GPU（或后面用 vLLM）。

### 3.4 `expected all tensors to be on the same device`
- **原因**：模型和输入不在同一设备。
- **处理**：输入也要 `.to(device)`。本项目脚本已统一处理；自己写代码时务必成对搬移。

---

## 4. 推理运行

### 4.1 输出像「复读 / 接话」，不像在回答
- **原因**：跳过了 chat template。
- **处理**：见 [docs/01-inference.md §3.1](docs/01-inference.md)。

### 4.2 生成结果是空的 / 只有一两个标点
- **可能原因**：
  - `max_new_tokens` 设太小或为 0；
  - 模型第一个 token 就输出了 EOS（多见于指令没套好）；
  - `temperature` 过高导致概率塌缩。
- **处理**：加 `--show-prompt` 看输入是否正确；调大 `--max-new-tokens`；用 `--temperature 0` 排查。

### 4.3 中文显示成乱码（Windows / 部分终端）
- **原因**：终端编码不是 UTF-8。
- **处理**：用 UTF-8 终端（Windows Terminal、VSCode 集成终端）；或脚本输出前设：
  ```bash
  export PYTHONIOENCODING=utf-8
  ```
  本项目所有文件均以 UTF-8 读写，乱码基本是终端侧问题。

### 4.4 一堆 `UserWarning`（torch_dtype / pad_token / generation config）
- 多数是无害提示（如「未设 pad_token_id，已用 eos」）。本项目脚本已尽量消掉噪声；想完全屏蔽：
  ```bash
  export TOKENIZERS_PARALLELISM=false
  ```
- 真正要关心的是 `Setting `pad_token_id`` 之外的**错误**（Error / Traceback），warning 一般可以忽略。

### 4.5 同一 prompt 多次结果不同，无法复现
- **原因**：采样模式（`do_sample=True`）。
- **处理**：baseline 对比统一用 `--temperature 0`（贪心）。详见 [docs/01-inference.md §3.8](docs/01-inference.md)。

---

## 5. 编码 / 系统 / WSL2

### 5.1 读写 jsonl 报 `UnicodeDecodeError`
- **原因**：文件不是 UTF-8，或读取时没指定编码。
- **处理**：本项目脚本统一 `encoding="utf-8"`；外部数据先转码：
  ```bash
  file -i data/xxx.jsonl                 # 查看实际编码
  iconv -f GBK -t UTF-8 in.jsonl > out.jsonl
  ```

### 5.2 路径里有中文 / 空格导致脚本异常
- **处理**：尽量用英文、无空格路径；引用时加引号。模型缓存路径也尽量纯英文。

### 5.3 WSL2 里 `nvidia-smi` 正常，但 `torch.cuda` 还是 False
- 复查：torch 是不是 CPU 版（见 1.2）；驱动是否够新；极少数情况需要 `wsl --update` 升级 WSL 内核后再试。

---

## 6. 找不到答案时

- 报错信息（去掉个人路径/模型名）+ transformers/torch 版本，搜索 GitHub Issues 和 HuggingFace 社区。
- 模型本身问题：去 `https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct` 看 Model Card 和 Discussions。
- vLLM 相关：`https://docs.vllm.ai/` 和 vLLM GitHub Issues（部署篇会再补充）。

---

## 变更记录

- 2026-06-29：初版，覆盖环境安装、模型下载、显存设备、推理运行、WSL2 五大类常见坑。
