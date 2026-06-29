# 01 · Baseline 推理：概念笔记与易错点

> 这一篇对应 `inference/chat.py` 和 `inference/generate.py`。
> 目标：理解「加载模型 → 套 chat template → 解码生成 → 算 tokens/s」每一步在干什么，
> 以及新手最容易在哪一步翻车。

---

## 0. 这一步在做什么、为什么要先做

整条学习链路是：**推理 → 数据 → 微调 → 评估 → 部署 → 压测**。

推理为什么排第一？

- 微调的意义是「改变模型行为」。但你不知道**改之前**模型长什么样，就没法判断微调到底有没有效、是变好了还是变差了。
- 所以第一步永远是用**原始基座模型**跑一批固定问题，把输出和速度记录下来，作为**对照基线（baseline）**。
- 后面用同一批问题去跑 adapter / merged 模型，再和 baseline 横向对比，才能下结论。

> 关键纪律：**评估用的问题集合必须固定**。建议把它落在 `benchmark/eval_prompts.jsonl`，贯穿始终，每次都用同一份。

---

## 1. 核心概念

### 1.1 CausalLM（因果语言模型 / 自回归生成）

- `AutoModelForCausalLM` 里的 **Causal** = 「因果」= 只看左边、不看右边，也就是「根据已有的 token 预测下一个 token」。
- 生成是**自回归**的：每生成一个 token，就把它拼回输入，再预测下一个，循环直到达到 `max_new_tokens` 或遇到结束符（EOS）。
- 所以「生成速度」主要由**要生成的 token 数**决定，而不是输入长度（输入只算一次前向）。

### 1.2 Tokenizer：文本 ↔ token id

模型不认识字符，只认识整数 id。tokenizer 负责在「字符串」和「id 序列」之间转换。

- **编码**：`tokenizer(text)` → `input_ids`（一串整数）。
- **解码**：`tokenizer.decode(ids)` → 还原成字符串。
- **分词粒度**：中文**一个字 ≠ 一个 token**。「微调」可能是 1 个 token，也可能是 2 个；英文一个常见词常是 1 个 token，生僻词会被拆成多个子词（BPE）。
  - 推论：你看到的「生成 100 个字」和「100 tokens」不是一回事。`tokens/s` 统计的是 **token**，不是字数。

### 1.3 Chat Template（最重要的概念，最容易踩坑）

指令模型（Qwen-Instruct、ChatGLM、Llama-Chat……）训练时，对话是被**特殊格式**包起来的，不是裸文本。以 Qwen2.5 为例，一条对话会被渲染成：

```text
<|im_start|>system
你是一个严谨的助手。
<|im_end|>
<|im_start|>user
请用三句话解释 LoRA。
<|im_end|>
<|im_start|>assistant
```

其中 `<|im_start|>` / `<|im_end|>` 是 **special tokens**（特殊 token），模型把它们当作结构标记。

- `tokenizer.apply_chat_template(messages, add_generation_prompt=True)` 就是帮你做这件事：把 `[{"role":"system",...},{"role":"user",...}]` 渲染成上面那串字符串。
  - `add_generation_prompt=True`：在末尾补上 `<|im_start|>assistant\n`，等于「告诉模型：现在轮到你说了」。
- **如果你跳过 chat template，直接把 `"请用三句话解释 LoRA"` 喂给 `model.generate()`**，模型收到的就是一段没头没尾的裸文本，它会当成「续写」而不是「回答问题」，输出会很奇怪（复读、接话、胡言乱语）。

> 用本项目脚本看一眼真实输入：`--show-prompt` 会把渲染后的完整字符串打印出来，强烈建议第一次跑时加上，建立直觉。

### 1.4 加载模型：dtype 与 device

| 场景 | 推荐 dtype | 原因 |
|---|---|---|
| NVIDIA GPU | `bfloat16` | Qwen2.5 训练精度，速度快、显存省、精度无损 |
| CPU | `float32` | **CPU 算不了 bf16/fp16**，强行用会报错或慢 10 倍以上 |
| Apple MPS | `float32` | MPS 对低精度支持有限，稳妥用 fp32 |

- `model.to(device)`：把权重搬到目标设备。
- `model.eval()`：切到推理模式，关闭 dropout 等只在训练用的行为。
- `@torch.inference_mode()`：装饰推理函数，告诉 PyTorch「不需要算梯度」，省显存、提速。推理一定要加。

### 1.5 生成参数（Generation）

| 参数 | 含义 | 取值建议 |
|---|---|---|
| `max_new_tokens` | **最多新生成**多少 token | baseline 常用 256~512 |
| `do_sample` | True=采样，False=贪心 | 可复现对比用 False |
| `temperature` | 越高越随机；→0 趋近贪心 | 0.7（Qwen 官方推荐） |
| `top_p` | 核采样，只在累积概率前 p 的候选里抽 | 0.8（Qwen 官方推荐） |
| `repetition_penalty` | 抑制重复，>1 减少复读 | 1.05 ~ 1.1 |
| `pad_token_id` | 填充 token，避免警告 | 一般设成 `eos_token_id` |

**贪心 vs 采样**：

- 贪心（`do_sample=False`）：每步选概率最高的 token，**确定、可复现**。适合 baseline 对比。
- 采样（`do_sample=True`）：按概率随机抽，每次输出可能不同，更有「创造性」，但不可复现。

### 1.6 max_new_tokens vs max_length（极易混淆）

- `max_new_tokens`：限制**新生成**的 token 数，与输入长度无关。**用这个**。
- `max_length`：限制**输入 + 输出的总长度**。当你的 prompt 很长时，`max_length - 输入长度` 可能只剩很少，模型几乎不生成就停了。
- 本项目脚本统一用 `max_new_tokens`。

### 1.7 KV Cache 与 tokens/s

- 生成第 N 个 token 时，模型需要「重新看」前 N-1 个 token 的中间结果（Key/Value）。如果每次都重算，复杂度是 O(N²)。
- **KV cache**：把每层的 K、V 缓存下来，下一步直接复用，复杂度降到 O(N)。代价是**显存随上下文长度线性增长**——长对话 / 长输出会更吃显存。
  - 这也是 vLLM「PagedAttention」要优化的核心对象，部署篇会深入。
- **tokens/s** = 新生成 token 数 / 纯生成耗时。注意：第一个 token 的延迟（TTFT）通常明显高于后续（要先把 prompt 算一遍），所以短输出时 tokens/s 偏低是正常的。

### 1.8 流式输出（Streaming）原理

`model.generate()` 是**阻塞**的：它会一口气生成完才返回。要「边生成边打印」，得：

1. 用 `TextIteratorStreamer` 包一层，让它把生成的 token 一边算一边吐出来。
2. 因为 `generate` 阻塞，只能放到**子线程**里跑，主线程从 streamer 里 `for piece in streamer` 取文本打印。

→ 见 `chat.py` 的 `generate_stream()`。

---

## 2. 脚本怎么用

环境（首次）：

```bash
conda create -n qwen-lab python=3.10 -y
conda activate qwen-lab
pip install -r requirements-train.txt   # baseline 只需 torch + transformers + accelerate + sentencepiece
cp .env.example .env                     # 按需改 BASE_MODEL
```

单次问答（看 chat template 真身 + 流式）：

```bash
python inference/chat.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "请用三句话解释 LoRA 微调的作用。" \
  --show-prompt --stream
```

确定性输出（baseline 对比推荐 `--temperature 0`）：

```bash
python inference/chat.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "Nginx 502 常见原因有哪些？" \
  --temperature 0 --max-new-tokens 256
```

批量生成并落盘（微调前先存一份基座 baseline）：

```bash
python inference/generate.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --input-file benchmark/eval_prompts.jsonl \
  --output-file outputs/baseline/base.jsonl \
  --temperature 0
```

> `benchmark/eval_prompts.jsonl` 还没建，下一步会随数据准备一起补上。你也可以先手写几行 `{"prompt": "..."}` 跑通。

---

## 3. 易错点 & 注意点（重点）

按踩坑频率排序。每条给「现象 → 原因 → 怎么办」。

### 3.1 跳过 chat template，输出像在「复读 / 接话」
- **现象**：模型不正面回答，而是接着你的话往下续，或反复重复。
- **原因**：直接把 prompt 原文喂给 `generate`，模型进入「续写」模式而非「对话」模式。
- **怎么办**：一律走 `apply_chat_template(..., add_generation_prompt=True)`。本项目脚本已内置。

### 3.2 把整个 `output_ids` 直接 decode，输出里混进了问题
- **现象**：返回的文本开头是用户自己的提问。
- **原因**：`generate` 返回的是 `[输入 + 输出]` 完整序列，需要切片把输入部分去掉。
- **怎么办**：用 `output_ids[0, input_len:]` 只取新生成部分。本项目脚本已处理。

### 3.3 CPU 上设了 `torch_dtype=torch.bfloat16`
- **现象**：报错 `"MPS/cpu" backends ...`，或速度慢得离谱。
- **原因**：CPU / 早期 MPS 不支持半精度计算。
- **怎么办**：仅 GPU 用 bf16；CPU 回退 fp32。脚本里已按设备自动选择。

### 3.4 用 `max_length` 而不是 `max_new_tokens`
- **现象**：prompt 稍长，模型只蹦出一两个字就停。
- **原因**：`max_length` 是「输入+输出」上限，扣掉输入后所剩无几。
- **怎么办**：统一用 `max_new_tokens`。

### 3.5 `temperature=0` 时仍传 `do_sample=True` / 采样参数
- **现象**：transformers 打 warning，或采样参数「不生效」，行为与预期不符。
- **原因**：贪心模式下 `temperature/top_p` 没有意义。
- **怎么办**：`temperature > 0` 才 `do_sample=True` 并传采样参数；否则 `do_sample=False`。脚本里 `build_gen_kwargs` 已做分支。

### 3.6 没设 `pad_token_id`，控制台一堆 warning
- **现象**：`Setting pad_token_id to eos_token_id ... for open-end generation.`
- **原因**：Qwen tokenizer 没有显式 pad token。
- **怎么办**：显式 `pad_token_id = tokenizer.eos_token_id`。无害但建议消掉噪声。

### 3.7 忘了 `.eval()` 或 `inference_mode()`
- **现象**：推理偏慢、显存偏高（0.5B 上体感不明显，但养坏习惯到大模型会吃亏）。
- **原因**：仍保留梯度计算图和 dropout。
- **怎么办**：加载后 `model.eval()`，推理函数加 `@torch.inference_mode()`。

### 3.8 想要「可复现」却用了采样
- **现象**：同一句话每次答案都不一样，没法稳定对比。
- **原因**：`do_sample=True` 引入随机。
- **怎么办**：baseline 对比用 `--temperature 0`（贪心）；若一定要采样复现，设 `torch.manual_seed(...)` 且固定 `do_sample=True`。

### 3.9 模型权重 / device 不匹配
- **现象**：`expected all tensors to be on the same device` 之类报错。
- **原因**：模型在 CUDA、输入在 CPU（或反之）。
- **怎么办**：输入也要 `.to(device)`，和模型在同一设备。

### 3.10 显存估算没概念，长输入直接 OOM
- **现象**：prompt 一长就 `CUDA out of memory`。
- **原因**：除了权重，KV cache 随上下文线性增长。
- **怎么办**：0.5B 在 bf16 下权重约 1GB，但加上 KV cache 和长上下文，留足余量；长输入先调小 `max_new_tokens` 或换 1.5B 以上的卡再试。详见 [TROUBLESHOOTING.md](../TROUBLESHOOTING.md)。

### 3.11 中文 token 数被误读成「字数」
- **现象**：以为生成 200 字 ≈ 200 tokens，速度估算全错。
- **原因**：中文一个字常对应 1~2 个 token。
- **怎么办**：以脚本统计的 `num_new_tokens` 为准。

### 3.12 batch 推理时 padding 侧反了（进阶，单条推理不受影响）
- **现象**：多条 prompt 一起生成时，结果错乱或末尾截断。
- **原因**：生成任务要 **left padding**（左侧补齐），分类/编码任务才用 right padding。
- **怎么办**：批量生成前 `tokenizer.padding_side = "left"`。本项目默认单条推理，暂不涉及，部署篇会用到。

---

## 4. 练习 / 思考题

跑通 baseline 后，建议亲手验证一遍，把直觉建立起来：

1. 同一 prompt 分别用 `--temperature 0` 和 `--temperature 0.9` 各跑 3 次，观察输出稳定性的差异。
2. 加 `--show-prompt`，对照「渲染前」和「渲染后」的字符串，找到 `<|im_start|>` 等特殊标记出现在哪。
3. 故意把一段长文本当 `--prompt`，观察 tokens/s 是否随输出变长而变化（体会 KV cache）。
4. 用 `generate.py` 生成一批输出，打开 `outputs/baseline/base.jsonl`，确认每条记录的字段含义。
5. 思考：为什么 `max_new_tokens` 是「生成速度」的主导因素，而输入长度不是？

---

## 5. 下一步

- [ ] 用 `generate.py` + 一份固定的 `eval_prompts.jsonl`，正式产出基座 baseline，存到 `outputs/baseline/`。
- [ ] 进入 `docs/02-finetune-lora.md`（微调）：在 baseline 基础上做 LoRA，再用同样的问题对比。
- [ ] 部署篇 `docs/03-vllm.md` 会重新讲 KV cache / PagedAttention / continuous batching，到时候回过头看这篇会更顺。
