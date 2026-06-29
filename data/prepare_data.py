#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理脚本（data/prepare_data.py）
======================================

把 raw 的 SFT jsonl 清洗成 train / valid / test 三份，供微调使用。

流程：解析+校验 → 全局去重 → 长度过滤 → 固定 seed 打乱 → 按比例切分 → 写出
      每一步都打印计数，方便看到「原始 → 可用」到底丢掉了多少。

最小运行（用自带 smoke 样例，不下载 tokenizer，按字符长度过滤）：
    python data/prepare_data.py --input data/examples/sft_demo.jsonl

精过滤（用真实 tokenizer 按 token 数过滤，结果更贴近训练实际）：
    python data/prepare_data.py \\
        --input data/examples/sft_demo.jsonl \\
        --tokenizer Qwen/Qwen2.5-0.5B-Instruct

核心知识点（详解见 docs/02-data.md）：
    1. 校验：每条样本必须有 user + assistant，否则训练时没有目标可算 loss。
    2. 去重必须在切分前做（全局去重），否则 valid/test 会和 train 重复 → 数据泄漏。
    3. 长度应按 token 统计（模型按 token 切分/截断）；没 tokenizer 时只能用字符数近似。
    4. 切分前必须 shuffle + 固定 seed，保证可复现且无顺序偏置。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

VALID_ROLES = {"system", "user", "assistant"}

# 简单 PII 掩码（仅示例；生产脱敏需要更严谨的规则 + 审计）
_PII_PATTERNS = [
    (re.compile(r"1[3-9]\d{9}"), "[手机号]"),          # 中国大陆手机号
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[邮箱]"),  # 邮箱
]


def iter_records(paths):
    """逐行读取 jsonl，yield (record, src_path, lineno)。

    空行跳过；JSON 解析失败的不让整批挂掉，而是计数并继续（数据里混进脏行很常见）。
    """
    for p in paths:
        fp = Path(p)
        if not fp.exists():
            print(f"[warn] 输入文件不存在，跳过: {p}", file=sys.stderr)
            continue
        with fp.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line), str(fp), lineno
                except json.JSONDecodeError as e:
                    print(f"[warn] {fp}:{lineno} JSON 解析失败: {e}",
                          file=sys.stderr)


def validate(record):
    """校验单条 messages 格式。返回 (ok, reason)。

    易错点：SFT 样本至少要有 1 个 user 和 1 个 assistant。
    缺 assistant 的样本必须丢——训练时 loss 只在 assistant 部分计算，
    没有 assistant 就没有训练目标。
    """
    if not isinstance(record, dict) or "messages" not in record:
        return False, "缺少 messages 字段"
    msgs = record["messages"]
    if not isinstance(msgs, list) or len(msgs) == 0:
        return False, "messages 为空"
    roles = [m.get("role") for m in msgs]
    for r in roles:
        if r not in VALID_ROLES:
            return False, f"未知 role: {r}"
    if "user" not in roles:
        return False, "缺少 user"
    if "assistant" not in roles:
        return False, "缺少 assistant"
    for m in msgs:
        if not str(m.get("content", "")).strip():
            return False, f"role={m.get('role')} 内容为空"
    return True, ""


def first_user_text(record) -> str:
    for m in record["messages"]:
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def make_dedup_key(record, mode: str) -> str:
    """构造去重 key。

    - user（默认）：按第一个 user 内容去重 = "同一个问题只留一份答案"，适合指令数据。
    - full：按整条对话去重 = "完全相同的对话只留一份"，更保守。
    """
    if mode == "full":
        return json.dumps(record["messages"], ensure_ascii=False, sort_keys=True)
    return first_user_text(record).strip()


def redact(record):
    """对 messages 内容做基础脱敏（可选，默认关闭）。返回新 record。"""
    for m in record["messages"]:
        text = m.get("content", "")
        for pat, rep in _PII_PATTERNS:
            text = pat.sub(rep, text)
        m["content"] = text
    return record


def length_of(record, tokenizer=None) -> int:
    """返回用于长度过滤的数值（token 数 或 字符数）。

    关键点：训练时的截断/打包是按 token 算的。传了 tokenizer 就按 token 统计
    （最准）；没传只能用字符数近似——但中文一个字常 ≈ 1~2 token，字符数只能粗过滤。
    """
    if tokenizer is not None:
        ids = tokenizer.apply_chat_template(
            record["messages"], add_generation_prompt=False, tokenize=True
        )
        return len(ids)
    # 近似：拼接各轮内容的字符数
    return sum(len(m.get("content", "")) for m in record["messages"])


def split_counts(n: int, train_ratio: float, valid_ratio: float):
    """计算 train/valid/test 条数；n>=3 时尽量保证 valid/test 各至少 1 条。"""
    n_train = int(round(n * train_ratio))
    n_valid = int(round(n * valid_ratio))
    n_test = n - n_train - n_valid
    if n >= 3:
        if n_valid < 1:
            n_valid = 1
        if n_test < 1:
            n_test = 1
        n_train = max(0, n - n_valid - n_test)
    return n_train, n_valid, n_test


def main():
    parser = argparse.ArgumentParser(description="SFT 数据清洗与切分")
    parser.add_argument("--input", nargs="+", required=True,
                        help="raw jsonl 文件，可传多个")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--valid-ratio", type=float, default=0.05,
                        help="test 占比 = 1 - train-ratio - valid-ratio")
    parser.add_argument("--min-len", type=int, default=10,
                        help="最小长度（字符数 或 token 数）")
    parser.add_argument("--max-len", type=int, default=2048,
                        help="最大长度（字符数 或 token 数）")
    parser.add_argument("--dedup-key", choices=["user", "full"], default="user")
    parser.add_argument("--tokenizer", default=None,
                        help="传模型名则按 token 统计长度；不传则按字符数近似")
    parser.add_argument("--redact", action="store_true",
                        help="启用基础 PII 脱敏（手机号/邮箱）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0,
                        help="只取前 N 条（smoke 测试用，0 表示不限）")
    args = parser.parse_args()

    if args.train_ratio + args.valid_ratio >= 1.0:
        raise SystemExit("[error] --train-ratio + --valid-ratio 必须小于 1"
                         "（要给 test 留份额）")

    # 可选：加载 tokenizer 做精确长度统计
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        print(f"[info] 加载 tokenizer: {args.tokenizer}（按 token 统计长度）")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # ---------- step1: 解析 + 校验 ----------
    records = []
    n_raw = n_bad = 0
    for rec, src, lineno in iter_records(args.input):
        n_raw += 1
        ok, reason = validate(rec)
        if not ok:
            n_bad += 1
            print(f"[skip] {src}:{lineno} {reason}", file=sys.stderr)
            continue
        records.append(rec)
    print(f"[step1] 解析 {n_raw} 条，校验通过 {len(records)} 条，丢弃 {n_bad} 条")

    # ---------- step2: 全局去重（必须在切分前，否则 valid/test 与 train 泄漏）----------
    seen = set()
    deduped = []
    for rec in records:
        k = make_dedup_key(rec, args.dedup_key)
        if k in seen:
            continue
        seen.add(k)
        deduped.append(rec)
    n_dup = len(records) - len(deduped)
    print(f"[step2] 去重后 {len(deduped)} 条（key={args.dedup_key}，去掉 {n_dup} 条重复）")
    records = deduped

    # ---------- step2.5: 可选脱敏 ----------
    if args.redact:
        records = [redact(r) for r in records]
        print("[step2.5] 已对手机号/邮箱做基础脱敏")

    # ---------- step3: 长度过滤 ----------
    kept = []
    len_values = []
    for rec in records:
        length = length_of(rec, tokenizer)
        len_values.append(length)
        if args.min_len <= length <= args.max_len:
            kept.append(rec)
    unit = "token" if tokenizer is not None else "字符"
    print(f"[step3] 长度过滤 [{args.min_len},{args.max_len}]({unit}) 后 "
          f"{len(kept)} 条（去掉 {len(records) - len(kept)} 条）")
    if len_values:
        print(f"        长度分布: min={min(len_values)} max={max(len_values)} "
              f"avg={sum(len_values) / len(len_values):.0f}")
    records = kept

    # ---------- step3.5: 可选 smoke 限制 ----------
    if args.limit > 0:
        records = records[:args.limit]
        print(f"[step3.5] --limit={args.limit}，只取前 {len(records)} 条")

    # ---------- step4: 固定 seed 打乱后切分 ----------
    rng = random.Random(args.seed)
    rng.shuffle(records)
    n = len(records)
    n_train, n_valid, n_test = split_counts(n, args.train_ratio, args.valid_ratio)
    train = records[:n_train]
    valid = records[n_train:n_train + n_valid]
    test = records[n_train + n_valid:n_train + n_valid + n_test]
    print(f"[step4] 切分 train={len(train)} valid={len(valid)} test={len(test)} "
          f"(seed={args.seed})")

    # ---------- step5: 写出 ----------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("train", train), ("valid", valid), ("test", test)):
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for rec in data:
                # ensure_ascii=False：保留中文可读，避免全变成 \uXXXX
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[done] 写入 {path} ({len(data)} 条)")


if __name__ == "__main__":
    main()
