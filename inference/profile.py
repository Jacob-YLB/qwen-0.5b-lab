#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理性能剖析（inference/profile.py）
====================================

拆解一次推理的耗时结构，建立「为什么会这样慢/快」的直觉。

核心概念：
    - prefill（预填充）：处理整段 prompt、算出第一个 token 的阶段。
      计算密集，耗时随【输入长度】增长。
    - decode（解码）：逐个生成后续 token 的阶段。
      访存密集（要把不断变大的 KV cache 反复读），每个 token 耗时相对稳定。
    - TTFT（Time To First Token）：首 token 延迟，主要由 prefill 决定。
    - tokens/s：生成吞吐，主要看 decode 阶段。

测量方法（两次 generate，可解释、不依赖计时器精度）：
    - 跑 max_new_tokens=1 → t1   ≈ prefill + 1 个 decode
    - 跑 max_new_tokens=N → tN
    - TTFT          ≈ t1
    - 单 token 耗时  ≈ (tN - t1) / (N - 1)
    - decode 吞吐   ≈ (N - 1) / (tN - t1)   tokens/s

易错点：
    - 第一次 generate 特别慢（kernel 编译 / cudnn autotuning），必须 warmup 再测。
    - tokens/s 要用 decode 阶段算才公平，别用 总token/总时间（会被 TTFT 拖低）。
    - GPU 才有显存可测；CPU 模式下显存项报告 N/A。

运行：
    # 单次详细剖析
    python inference/profile.py --model Qwen/Qwen2.5-0.5B-Instruct

    # 扫描不同输入长度，看 prefill/TTFT 如何随输入增长
    python inference/profile.py --sweep-input 16,128,512,1024
"""

from __future__ import annotations

import argparse
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device(prefer: str = "auto") -> torch.device:
    """按 auto/cpu/cuda 选择设备。"""
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


@torch.inference_mode()
def time_generate(model, tokenizer, device, prompt_text, max_new_tokens):
    """跑一次 generate（贪心，可复现），返回 (耗时秒, 生成 token 数)。"""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    t0 = time.perf_counter()
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    elapsed = time.perf_counter() - t0
    n_new = out.shape[1] - input_len
    return elapsed, n_new


def warmup(model, tokenizer, device, prompt_text, n=2):
    """预热：跑几次短生成，避免首次 kernel 编译/autotuning 污染测量。

    易错点：跳过 warmup 会让第一个测量点虚高（GPU 首次运行要编译/选 kernel）。
    """
    for _ in range(n):
        time_generate(model, tokenizer, device, prompt_text, max_new_tokens=4)
    if device.type == "cuda":
        torch.cuda.synchronize()


def profile_once(model, tokenizer, device, prompt_text, max_new_tokens):
    """单点剖析：用两次 generate 拆出 TTFT 与 decode 吞吐。"""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t1, _ = time_generate(model, tokenizer, device, prompt_text, max_new_tokens=1)
    tN, nN = time_generate(model, tokenizer, device, prompt_text, max_new_tokens)

    ttft = t1
    if nN > 1:
        decode_total = tN - t1
        per_token = decode_total / (nN - 1)
        decode_tps = (nN - 1) / decode_total if decode_total > 0 else 0.0
    else:
        per_token = 0.0
        decode_tps = 0.0

    overall_tps = nN / tN if tN > 0 else 0.0
    peak_mem = None
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MiB

    input_tokens = len(
        tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    )
    return {
        "input_tokens": input_tokens,
        "max_new": max_new_tokens,
        "ttft_s": ttft,
        "decode_tok_per_s": decode_tps,
        "per_decode_token_ms": per_token * 1000,
        "total_s": tN,
        "overall_tok_per_s": overall_tps,
        "peak_mem_mib": peak_mem,
    }


def main():
    parser = argparse.ArgumentParser(description="推理性能剖析")
    parser.add_argument(
        "--model",
        default=os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
    )
    parser.add_argument("--prompt", default="请用一段话介绍 Linux 系统的进程管理。")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--sweep-input", default=None,
        help="逗号分隔的输入长度列表，如 16,128,512,1024；扫描 prefill 成本",
    )
    args = parser.parse_args()

    device = pick_device(args.device)
    dtype_name = "bf16" if device.type == "cuda" else "fp32"
    print(f"[info] 模型 {args.model} | 设备 {device} | dtype {dtype_name}\n")
    tokenizer, model = load_model(args.model, device)

    print("[warmup] 预热中（避免首次 kernel 编译影响测量）...")
    warmup(model, tokenizer, device, args.prompt)

    if args.sweep_input:
        lengths = [int(x) for x in args.sweep_input.split(",")]
        base_ids = tokenizer(args.prompt, add_special_tokens=False)["input_ids"]
        if not base_ids:
            raise SystemExit("[error] prompt tokenize 后为空，换一段 prompt")
        print(f"\n[扫描] 输入长度对 TTFT/prefill 的影响（max_new={args.max_new_tokens}）\n")
        print(f"{'input_tokens':>12} | {'TTFT':>9} | {'decode tok/s':>12} | {'total':>7}")
        print("-" * 52)
        for L in lengths:
            # 用基线 token 循环填充到约 L 长，制造不同长度的输入
            padded = (base_ids * ((L // len(base_ids)) + 1))[:L]
            prompt_text = tokenizer.decode(padded, skip_special_tokens=True)
            r = profile_once(model, tokenizer, device, prompt_text, args.max_new_tokens)
            print(f"{r['input_tokens']:>12} | {r['ttft_s'] * 1000:>8.0f}ms | "
                  f"{r['decode_tok_per_s']:>12.1f} | {r['total_s']:>6.2f}s")
        print("\n观察：输入越长，TTFT 越高（prefill 成本随输入增长）；"
              "decode 吞吐基本不受输入长度影响。")
    else:
        r = profile_once(model, tokenizer, device, args.prompt, args.max_new_tokens)
        mem = f"{r['peak_mem_mib']:.0f}" if r['peak_mem_mib'] is not None else "N/A"
        print("\n[单次剖析结果]")
        print(f"  输入 {r['input_tokens']} token | 生成 {r['max_new']} token | "
              f"峰值显存 {mem} MiB")
        print(f"  TTFT            {r['ttft_s'] * 1000:>7.0f} ms")
        print(f"  decode 吞吐     {r['decode_tok_per_s']:>7.1f} tok/s "
              f"(每 token {r['per_decode_token_ms']:.1f} ms)")
        print(f"  overall 吞吐    {r['overall_tok_per_s']:>7.1f} tok/s "
              f"(含 TTFT，会被拖低)")
        print(f"  总耗时          {r['total_s']:>7.2f} s")
        print("\n解读：")
        print(f"  - TTFT 主要由 prefill（处理 prompt）决定，输入越长越高。")
        print(f"  - 衡量'生成速度'应该看 decode 吞吐，而不是 overall（overall 含首 token 延迟）。")
        print(f"  - 峰值显存含 权重 + KV cache + 激活；输入/输出越长，KV cache 越大。")
        print(f"  - 想看输入长度的影响：加 --sweep-input 16,128,512,1024")


if __name__ == "__main__":
    main()
