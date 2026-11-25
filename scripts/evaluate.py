"""在 100 条留出集上跑三项指标，微调前后各跑一次。

三种推理后端：
  --backend hf      本地 transformers 加载基座（可加 --lora 挂适配器），云 GPU 上用
  --backend openai  调 OpenAI 兼容接口（vLLM 服务），--url http://127.0.0.1:8001/v1 --model qwen-risk-lora
  --backend gold    直接用标准答案，CPU 上验证评估流程本身（应为满分）

    python scripts/evaluate.py --backend hf --model $BASE --tag before
    python scripts/evaluate.py --backend hf --model $BASE --lora outputs/qwen-risk-lora --tag after
    python scripts/compare.py reports/before.json reports/after.json

输出 reports/<tag>.json（汇总 + 每条预测）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetune.common import SFT_DIR, read_jsonl, to_messages  # noqa: E402
from finetune.metrics import aggregate, score_one  # noqa: E402


def gen_gold(samples, args):
    return [s["answer"] for s in samples]


def gen_openai(samples, args):
    out = []
    for s in samples:
        body = {"model": args.model, "messages": to_messages(s, with_answer=False),
                "temperature": 0, "max_tokens": args.max_new_tokens}
        req = urllib.request.Request(args.url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            out.append(json.loads(r.read())["choices"][0]["message"]["content"].strip())
    return out


def gen_hf(samples, args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map="auto",
                                                 trust_remote_code=True)
    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora)
    model.eval()
    out = []
    for i in range(0, len(samples), args.batch):
        chunk = samples[i:i + args.batch]
        texts = [tok.apply_chat_template(to_messages(s, with_answer=False), tokenize=False,
                                         add_generation_prompt=True) for s in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        out += [tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for g in gen]
        print(f"[eval] {len(out)}/{len(samples)}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "openai", "gold"], required=True)
    ap.add_argument("--model", default="", help="hf: 基座目录；openai: served model name")
    ap.add_argument("--lora", default="")
    ap.add_argument("--url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--data", default=str(SFT_DIR / "eval.jsonl"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()
    if args.backend != "gold" and not args.model:
        ap.error("真实推理必须指定 --model")
    if args.backend == "gold" and args.tag in {"before", "after"}:
        ap.error("gold 自检不能使用 before/after 标签")

    samples = read_jsonl(Path(args.data))
    t0 = time.time()
    answers = {"gold": gen_gold, "openai": gen_openai, "hf": gen_hf}[args.backend](samples, args)
    rows = [score_one(s, a) | {"answer": a} for s, a in zip(samples, answers)]
    summary = aggregate(rows)
    summary.update({"backend": args.backend, "model": args.model, "lora": args.lora,
                    "dataset_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                    "seconds": round(time.time() - t0, 1)})
    tag = args.tag or args.backend
    out = ROOT / "reports" / f"{tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_type"}, ensure_ascii=False))
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()
