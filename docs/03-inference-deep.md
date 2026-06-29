# 03 · 推理进阶：多轮对话与性能剖析

> 这一篇对应 `inference/chat_multiturn.py`（多轮交互）和 `inference/profile.py`（性能剖析）。
> 在 [01-inference.md](01-inference.md) 的单轮推理基础上，深入两个主题：
> 「多轮上下文怎么累积、为什么越聊越慢」和「一次推理的时间到底花在哪」。

---

## 0. 为什么单独讲这两块

单轮推理（01）只解决「问一句答一句」。真实使用中：

- **多轮对话**才是常态，而它暴露了 LLM 推理最重要的一个工程现实——**上下文是有成本的**。
- **性能剖析**让你看懂延迟结构，否则部署时「为什么慢」「为什么 OOM」全是黑盒，没法优化。

这两块也是后面理解 vLLM（PagedAttention、prefix caching、continuous batching）的前置知识。

---

## 1. 多轮对话：上下文是怎么喂进去的

### 1.1 上下文累积

每一轮，脚本都把**完整历史**重新拼成一份 messages：

```text
[system] + [user1, assistant1, user2, assistant2, ...] + [当前 user]
→ apply_chat_template → 送进 generate
```

模型本身没有「记忆器官」。它之所以显得记得前面的话，**纯粹是因为你每轮都把历史又喂了一遍**。

### 1.2 为什么每轮都要重跑一遍 prompt（关键认知）

`model.generate()` 是**无状态**的：

- 在**一次** generate 内部，KV cache 会被复用（生成第 N 个 token 时不用重算前 N-1 个），这是单次生成快的根本原因。
- 但**跨调用**（下一轮对话），上一轮的 KV cache 不会被保留。所以多轮时，模型每轮都要对「整段历史」重新做一次 prefill。

后果：**历史越长，每轮越慢、越费显存**。聊到第 10 轮时，哪怕你只问「然后呢」，模型也要把前 9 轮全部重新处理一遍。

> 这正是 vLLM 的 **prefix caching** 要解决的：它发现多轮对话里 `system + 历史` 是每轮重复的前缀，于是把这部分 KV 缓存下来跨请求复用。原生 Transformers 没有这个能力。

### 1.3 历史管理（防止上下文爆炸）

历史无限增长会撞上模型的 `max_model_len`（最大上下文），轻则被截断、重则报错/ OOM。常见管理策略：

| 策略 | 做法 | 适用 |
|---|---|---|
| 滑窗裁剪 | 只保留最近 N 轮（本脚本默认） | 简单、教学 |
| 按 token 裁剪 | 保留不超过 X 个 token 的历史 | 更精确 |
| 摘要压缩 | 用模型把旧历史压成一段摘要 | 长对话、成本敏感 |

`chat_multiturn.py` 用 `--max-history` 做最简单的滑窗：保留 `system` + 最近 N 轮。

### 1.4 多轮对话直接变训练数据

脚本里 `/save chat.jsonl` 会把当前历史存成：

```json
{"messages":[{"role":"system",...},{"role":"user",...},{"role":"assistant",...},...]}
```

这正是 `data/prepare_data.py` 能读的 SFT 格式。所以你可以：**手工调教一段满意的对话 → 存下来 → 清洗 → 喂给微调**。这是一个很实用的「数据构造」工作流。

---

## 2. 性能剖析：一次推理的时间花在哪

### 2.1 prefill vs decode（最重要的二分）

一次「输入 prompt + 生成 N 个 token」的推理，可以分成两个阶段：

| 阶段 | 做什么 | 成本特征 | 耗时随什么增长 |
|---|---|---|---|
| **prefill** | 并行处理整段 prompt，算出第 1 个 token | 计算密集 | **输入长度** |
| **decode** | 逐个生成第 2…N 个 token | **访存密集** | 生成数量（每步还要读越来越大的 KV cache） |

为什么 decode 是访存密集？因为生成第 k 个 token 时，要把前 k-1 个 token 的 KV cache 全读一遍——计算量很小（一次矩阵向量乘），但要把一大块 KV cache 从显存搬进计算单元，**搬数据的时间远大于算的时间**。

### 2.2 TTFT 与 tokens/s

- **TTFT（Time To First Token）**：从输入到第一个 token 产出的延迟，≈ prefill 耗时。输入越长越高。
- **decode 吞吐**：生成阶段的 tokens/s，反映模型「说起来」有多快。
- **overall 吞吐** = 生成总数 / 总耗时，**含 TTFT**，会被首 token 拖低。

> 衡量「生成速度」应看 **decode 吞吐**，别看 overall。overall 把一次性的 prefill 摊进了每个 token，数值偏低且不稳定。

### 2.3 两次 generate 测量法

`profile.py` 用两次 generate 拆出各指标，不依赖计时器精度：

```text
t1 = 生成 1  个 token 的耗时   ≈ prefill + 1 个 decode
tN = 生成 N  个 token 的耗时   ≈ prefill + N 个 decode
→ TTFT        ≈ t1
→ decode 吞吐 ≈ (N - 1) / (tN - t1)
```

### 2.4 显存构成

推理时的显存 = **权重** + **KV cache** + **激活**：

- 权重：固定。0.5B 在 bf16 下约 1GB。
- KV cache：随 `(输入长度 + 生成长度)` **线性增长**，长对话/长输出的主要开销。
- 激活：prefill 时大（并行算很多 token），decode 时小。

`profile.py` 在 GPU 上会报告峰值显存（`torch.cuda.max_memory_allocated`）。

### 2.5 为什么必须 warmup

第一次 `generate()` 会特别慢：CUDA kernel 首次编译、cudnn autotuning 选最优卷积配置、lazy 初始化……这些一次性开销会让第一个测量点严重虚高。所以正式测量前要扔掉几次预热结果。

---

## 3. 两个脚本怎么用

多轮对话（交互式）：

```bash
python inference/chat_multiturn.py --model Qwen/Qwen2.5-0.5B-Instruct
# 进入后正常打字对话；试 /show 看历史，/save chat.jsonl 存档
```

单次性能剖析：

```bash
python inference/profile.py --model Qwen/Qwen2.5-0.5B-Instruct
```

扫描输入长度（看 prefill/TTFT 怎么随输入涨）：

```bash
python inference/profile.py --model Qwen/Qwen2.5-0.5B-Instruct --sweep-input 16,128,512,1024
```

---

## 4. 易错点 & 注意点

### 多轮对话

1. **忘了把回复 append 回历史** → 下一轮模型「失忆」。脚本已处理，但自己写时这是头号坑。
2. **以为 Transformers 会自动记上下文** → 不会，每次 generate 无状态，必须手动喂完整历史。
3. **历史无限增长** → 撞 `max_model_len` 报错或被截断；用滑窗/按 token 裁剪。
4. **裁剪时丢了 system** → 模型人设崩塌；裁剪必须保留开头的 system。
5. **忽略每轮成本** → 聊得越久每轮越慢（每轮重跑 prefill），这是 Transformers 的固有特性。

### 性能剖析

6. **不 warmup 就测** → 第一个点虚高，结论失真。务必预热。
7. **用 overall tokens/s 当速度** → 被 TTFT 拖低，偏低且不稳；用 decode 吞吐。
8. **混淆 prefill 和 decode** → 输入长度主要影响 TTFT/prefill，不太影响 decode 吞吐。
9. **CPU 上用 bf16 跑性能** → 报错或极慢，CPU 必须 fp32（速度本就慢，对比 GPU 不公平）。
10. **忽略 KV cache 的显存** → 以为「0.5B 只要 1GB」，长上下文下 KV cache 可能反超权重。
11. **想靠增大 batch 提速**（进阶） → 单进程 Transformers 的 batch 提升有限；真正的并发吞吐要靠 vLLM 的 continuous batching，部署篇讲。

---

## 5. 练习 / 思考题

1. 在多轮对话里连续问 5 个相关问题，观察每轮 `[历史已累积到 X token]` 的增长和单轮耗时的变化。
2. 用 `/save` 存一段对话，再用 `data/prepare_data.py --input 那个文件` 跑一遍，确认能被清洗。
3. 跑 `profile.py --sweep-input 16,128,512,1024`，画出「输入长度 → TTFT」的趋势，验证 prefill 是线性增长的。
4. 对比同一 prompt 的 TTFT 和 decode 吞吐，体会「首字慢、后续快」。
5. 思考：为什么多轮对话里，TTFT 会随轮次增加而升高？（提示：每轮的输入 = 全部历史。）

---

## 6. 下一步

- [ ] 用 `chat_multiturn.py` 跑几轮对话，亲手感受上下文累积的成本。
- [ ] 用 `profile.py` 在你的机器上测出 baseline 的 TTFT / decode 吞吐，留作后续 vLLM 对比的基线。
- [ ] 进入 `docs/04-finetune-lora.md`（微调）：在多轮/单轮数据上做 LoRA，再用这里的方法对比性能与质量。
- [ ] 部署篇会把这些性能概念升级到服务侧：PagedAttention（KV cache 管理）、prefix caching（多轮前缀复用）、continuous batching（并发吞吐）。
