# 02 · 数据准备：概念笔记与易错点

> 这一篇对应 `data/prepare_data.py`、`data/examples/sft_demo.jsonl`、`benchmark/eval_prompts.jsonl`。
> 目标：理解「数据格式 → 清洗 → 去重 → 切分」每一步在干什么，以及数据侧最容易埋的雷。

---

## 0. 这一步在做什么、为什么

模型微调的本质，是用一批「输入 → 期望输出」的样本去调整参数。所以**数据质量直接决定微调上限**——再好的训练方法也救不回脏数据。

这一步要做的事：

1. 把原始数据整理成模型/训练框架认识的**格式**。
2. **清洗**：去重、去空、过滤异常长度、脱敏。
3. **切分**：train / valid / test 三份，各自承担不同职责。
4. 顺带准备一份**固定的评估问题集**（`eval_prompts.jsonl`），贯穿 baseline → 微调 → 部署的对比。

> 关键纪律：**评估集一旦定下就不要再动**。它是你判断「微调有没有效」的标尺，标尺变了就没法横向对比。

---

## 1. 核心概念

### 1.1 SFT 数据格式

本项目统一用 **JSONL**（每行一个独立 JSON 对象），分两种主要格式：

**(a) messages 格式 —— 指令/对话微调（本项目主用）**

```json
{"messages":[{"role":"system","content":"你是运维助手。"},{"role":"user","content":"Nginx 502 怎么排查？"},{"role":"assistant","content":"先看 error.log..."}]}
```

- `role` 取值：`system`（可选）/ `user` / `assistant`。
- 训练框架（如 TRL/LLaMA-Factory）会按这套结构套用 chat template，并**只对 assistant 部分计算 loss**（user 部分只是条件，不该被「学习」）。

**(b) text 格式 —— 继续预训练（README 路线 C）**

```json
{"text":"这里是一段领域语料，用于继续预训练或领域语言建模。"}
```

- 纯文本，目标是「学会这种语言分布」，不做指令对齐。

> 还有个常见的 **ShareGPT 格式**（`{"conversations":[{"from":"human","value":"..."},...]}`），多数训练框架都支持自动转换。本项目先用 messages，够用且直观。

### 1.2 chat template 在数据侧的角色

在 [01-inference.md](01-inference.md) 里强调过：**推理时必须套 chat template**。训练时也一样——训练样本最终也会被渲染成带 `<|im_start|>` 等特殊标记的字符串喂给模型。

含义：你在数据里写的 `{"role":"user","content":"..."}` 不是模型直接看到的形态；训练/推理框架会替你套模板。所以**数据侧只管写结构化 messages，不要自己手拼 `<|im_start|>`**，否则会双重套模板导致训练错位。

### 1.3 去重（dedup）

为什么要去重？

- 重复样本等于**变相放大了它的权重**，模型会对这些问题过拟合，对其它问题欠拟合。
- 更糟的是：如果同一条问题同时进了 train 和 valid，valid 指标会虚高，你以为模型学得好，其实是「背答案」。

去重 key 的选择：

| key | 含义 | 适用 |
|---|---|---|
| `user`（默认） | 同一个问题只留一份 | 指令数据，避免同题多答 |
| `full` | 完全相同的整条对话才合并 | 更保守，保留不同答案 |

**最关键的一条：去重必须在切分前做（全局去重）。** 切分后再去重已经晚了——重复已经分散到不同 split 里了。

### 1.4 长度过滤：字符 vs token

模型按 **token** 切分、截断、打包。所以长度理应按 token 算：

- 传 `--tokenizer <模型>` → 用真实 tokenizer 数 token，**最准**。
- 不传 → 只能用字符数近似，**中文有偏差**（一个字 ≈ 1~2 token）。

过滤的目的：

- 去掉过短的垃圾（如只有一两个字符的脏数据）。
- 去掉过长的样本（超过 `max_len` 会被训练时截断，截断掉答案尾部等于教错了）。

### 1.5 切分：train / valid / test

| split | 作用 | 用法 |
|---|---|---|
| **train** | 训练时参与梯度更新 | 占大头（如 90%） |
| **valid** | 训练中定期评估，用于**选 epoch / 早停** | 小份（如 5%），不参与训练 |
| **test** | 训练完全结束后做最终评估 | 小份（如 5%），全程不碰 |

- **valid 的意义**：训练集 loss 一直在降，但 valid loss 可能在某个点反弹——那就是过拟合的开始。valid 帮你决定在第几个 epoch 停。
- **小数据保护**：数据很少时（比如本项目 smoke 只有十几条），valid/test 可能切出 0 条。`prepare_data.py` 会在总数 ≥ 3 时尽量给 valid/test 各留至少 1 条。

### 1.6 数据泄漏（data leakage）

指「评估时用到的信息，在训练时已经见过」。除了上面说的重复泄漏，常见还有：

- 同一个问题在 train 和 test 里换了措辞出现。
- 一篇长文档被切成多条，分别进了 train 和 test。
- 去重没做全局，导致跨 split 重复。

泄漏的后果：评估指标虚高，上线后效果远不如预期。

### 1.7 评估集 vs 训练集

本项目有两份不同用途的「问题」：

| 文件 | 用途 | 是否参与训练 |
|---|---|---|
| `data/processed/{train,valid,test}.jsonl` | 微调的输入（带标准答案） | 是 |
| `benchmark/eval_prompts.jsonl` | 只有问题，用于对比 baseline/微调/部署 | **否** |

`eval_prompts.jsonl` 故意和训练数据**同域（都是运维）但不完全重合**：一部分和训练样本接近（验证是否学到风格/记忆），一部分没直接覆盖（验证泛化）。它配合 `inference/generate.py` 产出可对比的输出。

---

## 2. 脚本怎么用

用自带 smoke 样例，不下载 tokenizer（按字符长度过滤，秒级跑完）：

```bash
python data/prepare_data.py --input data/examples/sft_demo.jsonl
```

精过滤（按真实 token 数过滤，更贴近训练实际）：

```bash
python data/prepare_data.py \
  --input data/examples/sft_demo.jsonl \
  --tokenizer Qwen/Qwen2.5-0.5B-Instruct
```

产出在 `data/processed/`（已被 `.gitignore` 忽略，不会误提交）：

```text
data/processed/
├── train.jsonl
├── valid.jsonl
└── test.jsonl
```

跑通后，用同一批评估问题产出基座 baseline：

```bash
python inference/generate.py \
  --input-file benchmark/eval_prompts.jsonl \
  --output-file outputs/baseline/base.jsonl \
  --temperature 0
```

> 12 条 smoke 样本切分后 valid/test 各只有 1 条，仅用于跑通流程。真正训练时要换更大、更多样的数据集。

---

## 3. 易错点 & 注意点（重点）

### 3.1 先切分后去重 → valid/test 与 train 泄漏
- **现象**：valid loss 异常低，上线后效果差很多。
- **原因**：重复样本在切分前没清掉，分散到了不同 split。
- **怎么办**：去重放在切分前、且是**全局**去重。`prepare_data.py` 的 step2 就是这么做的。

### 3.2 用字符长度估 token（中文偏差大）
- **现象**：按字符设的 `max_len`，实际 token 数远超，训练时被截断。
- **原因**：中文一个字常是 1~2 个 token，字符数 ≠ token 数。
- **怎么办**：正式过滤时传 `--tokenizer` 按 token 统计；字符长度只在快速 smoke 时用。

### 3.3 样本缺 assistant → 训练没有目标
- **现象**：训练 loss 算不出来或为 0。
- **原因**：SFT 的 loss 只在 assistant 部分计算，缺 assistant 的样本没有监督信号。
- **怎么办**：校验阶段就丢掉，`validate()` 会拦下「缺 user/assistant」「内容为空」的样本。

### 3.4 切分前忘了 shuffle → 顺序偏置
- **现象**：原始数据若按类别排序（比如前 100 条都是网络、后面都是数据库），不 shuffle 就切，会导致 valid 全是某一类。
- **怎么办**：切分前用**固定 seed** 打乱。`prepare_data.py` 用 `random.Random(seed)` 保证可复现。

### 3.5 小数据 LoRA 容易过拟合
- **现象**：训练 loss 一直降，但模型在新问题上没变好甚至变差（只会背训练集）。
- **原因**：样本太少、epoch 太多。
- **怎么办**：靠 valid loss 监控并早停；epoch 不要贪多（小数据 2~3 轮往往就够）；数据量才是根本。

### 3.6 system prompt 不一致 → 风格混乱
- **现象**：微调后回答风格不稳定。
- **原因**：训练样本的 system 各写各的，模型学不到统一风格。
- **怎么办**：同一批数据尽量用统一的 system prompt（本项目样例统一用「严谨、简洁的运维助手」）。

### 3.7 输出 jsonl 中文变 `\uXXXX`
- **现象**：写出的文件里中文全变成 `微调` 之类。
- **原因**：`json.dumps` 默认 `ensure_ascii=True`。
- **怎么办**：写文件时 `ensure_ascii=False`（`prepare_data.py` 已设置）。

### 3.8 评估集和训练集混淆
- **现象**：用训练集问题做评估，结论偏乐观。
- **原因**：评估问题混进了训练数据。
- **怎么办**：`benchmark/eval_prompts.jsonl` 只用于推理对比，**绝不**进入 `data/processed/`。两者物理隔离。

### 3.9 手拼了 special tokens → 双重套模板
- **现象**：训练后模型输出里出现重复的 `<|im_start|>`，或答非所问。
- **原因**：数据里自己写了 `<|im_start|>`，框架又套了一次。
- **怎么办**：数据只写结构化 `messages`，模板交给框架渲染。

---

## 4. 练习 / 思考题

1. 用 `prepare_data.py` 分别跑「字符长度」和「token 长度」两次，对比打印的长度分布差异，直观感受中文的字符/token 比。
2. 故意在 `sft_demo.jsonl` 里加一条只有 user、没有 assistant 的样本，看校验是否拦下。
3. 加一条和已有样本 user 完全相同的重复行，看去重计数是否 +1。
4. 改 `--seed` 重跑，观察 train/valid/test 的具体条目是否变化（体会 seed 的作用）。
5. 思考：为什么 valid 不能参与训练？如果用 test 来选 epoch，会出什么问题？

---

## 5. 下一步

- [ ] 用真实数据（或更大的公开运维问答集）替换 smoke 样例，正式产出 `data/processed/`。
- [ ] 用 `inference/generate.py` + `eval_prompts.jsonl` 产出基座 baseline，存到 `outputs/baseline/`。
- [ ] 进入 `docs/03-finetune-lora.md`（微调）：在 `data/processed/` 上做 LoRA，再用 `eval_prompts.jsonl` 对比 baseline。
