#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多轮交互式对话（inference/chat_multiturn.py）
==============================================

一个命令行 REPL：你说一句、模型回一句，并把整段历史累积起来，
让你直观看到「多轮上下文」是怎么喂给模型的。

和单轮 chat.py 的关键区别（也是这一篇要学的重点）：
    1. 上下文累积：每轮都把 [system + 之前所有 user/assistant + 当前 user] 一起送进去，
       模型才能"记住"前面说过的话。
    2. 每轮都要重新跑一遍完整 prompt —— Transformers 的 generate() 是【无状态】的，
       它不会自动保留上一轮的 KV cache。所以历史越长，每轮的输入越长、越慢、越费显存。
       （这正是 vLLM "prefix caching" 要优化的点，部署篇会讲。）
    3. 历史无限增长最终会超过模型的 max_model_len → 报错或被截断。
       本脚本用 --max-history 做朴素滑窗裁剪，演示如何管理上下文。

运行：
    python inference/chat_multiturn.py --model Qwen/Qwen2.5-0.5B-Instruct

交互命令：
    /help      帮助
    /show      打印当前完整历史
    /clear     清空历史（重开对话）
    /save <f>  把当前历史存成 jsonl（格式和 prepare_data.py 兼容，可直接当训练数据！）
    /exit      退出（Ctrl-C / Ctrl-D 也可以）
"""

from __future__ import annotations

import argparse
import json
import os
import time
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


def pick_device(prefer: str = "auto") -> torch.device:
    """按 auto/cpu/cuda 选择设备，避免写死 cuda 在无卡环境崩掉。"""
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但当前环境检测不到 CUDA。")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_name: str, device: torch.device):
    """加载 tokenizer 和模型。GPU 用 bf16，CPU 用 fp32。"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def trim_history(messages, max_turns):
    """朴素滑窗：保留 system + 最近 max_turns 轮（一轮 = user + assistant）。

    易错点：裁剪时一定要保留开头的 system，否则模型人设会丢。
    另外这只是"按轮数"裁剪，没按 token 数——长回答仍可能让单轮很长。
    生产环境应按 token 数裁剪，或用摘要压缩历史。
    """
    if max_turns <= 0:
        return messages
    sys_msgs = [m for m in messages if m["role"] == "system"]
    convo = [m for m in messages if m["role"] != "system"]
    keep = convo[-max_turns * 2:]  # 一轮 = 2 条
    return sys_msgs + keep


def stream_reply(tokenizer, model, device, messages, args):
    """流式生成 assistant 回复，逐片段 yield。"""
    input_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs = {
        **inputs,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }
    if args.temperature > 0:
        gen_kwargs.update(temperature=args.temperature, top_p=args.top_p)

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    for piece in streamer:
        yield piece
    thread.join()


def main():
    parser = argparse.ArgumentParser(description="Qwen 多轮交互对话")
    parser.add_argument(
        "--model",
        default=os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
    )
    parser.add_argument("--system", default="你是一名严谨、简洁的运维助手。")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--max-history", type=int, default=6,
        help="保留最近几轮（0 = 不裁剪；历史越长每轮越慢）",
    )
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"[info] 模型 {args.model} | 设备 {device} | 保留最近 {args.max_history} 轮")
    print("[info] 输入 /help 查看命令，/exit 退出。\n")
    tokenizer, model = load_model(args.model, device)

    history = [{"role": "system", "content": args.system}]

    try:
        while True:
            try:
                user = input("你> ").strip()
            except EOFError:
                break

            if not user:
                continue

            # ---- 命令处理 ----
            if user in ("/exit", "/quit"):
                break
            elif user == "/help":
                print("命令：/help  /show  /clear  /save <file>  /exit")
            elif user == "/show":
                print(json.dumps(history, ensure_ascii=False, indent=2))
            elif user == "/clear":
                history = [{"role": "system", "content": args.system}]
                print("[已清空历史]")
            elif user.startswith("/save "):
                path = user[len("/save "):].strip()
                with open(path, "w", encoding="utf-8") as f:
                    # 存成 {"messages": [...]}，正是 prepare_data.py 能读的 SFT 格式
                    f.write(json.dumps({"messages": history}, ensure_ascii=False) + "\n")
                print(f"[已保存到 {path}，可直接喂给 data/prepare_data.py]")
            elif user.startswith("/"):
                print(f"[未知命令] {user}，输入 /help 查看可用命令。")
            else:
                # ---- 正常对话 ----
                history.append({"role": "user", "content": user})
                if args.max_history > 0:
                    history = trim_history(history, args.max_history)

                print("助手> ", end="", flush=True)
                t0 = time.perf_counter()
                parts = []
                for piece in stream_reply(tokenizer, model, device, history, args):
                    parts.append(piece)
                    print(piece, end="", flush=True)
                elapsed = time.perf_counter() - t0
                reply = "".join(parts)

                # 易错点：必须把回复追加回历史，否则下一轮模型"失忆"
                history.append({"role": "assistant", "content": reply})

                n_tok = len(tokenizer(reply, add_special_tokens=False)["input_ids"])
                tps = n_tok / elapsed if elapsed > 0 else 0.0
                n_input = len(tokenizer(
                    tokenizer.apply_chat_template(history, tokenize=False),
                    add_special_tokens=False,
                )["input_ids"])
                print(f"\n  [本轮 {n_tok} tok / {elapsed:.2f}s ≈ {tps:.1f} tok/s "
                      f"| 历史已累积到 {n_input} token]\n")
    except KeyboardInterrupt:
        print("\n[bye]")


if __name__ == "__main__":
    main()
