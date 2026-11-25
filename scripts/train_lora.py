"""SFT + LoRA 微调（云 GPU 上运行，24G 显存可训 Qwen2.5-1.5B / 3B-Instruct）。

    python scripts/train_lora.py --model /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct \
        --out outputs/qwen-risk-lora

- 输入用 Qwen 自带 chat template 拼 system / user / assistant；只对 assistant 部分算 loss。
- LoRA 挂在全部注意力和 MLP 投影层（q/k/v/o/gate/up/down），r=16, alpha=32。
- 支持 bf16 的卡（A10/3090/4090/A100）自动用 bf16，否则 fp16。
- 输出目录只存 LoRA 适配器（几十 MB），vLLM 用 --enable-lora 直接挂载，不需要合并。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
                          TrainingArguments, set_seed)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finetune.common import SFT_DIR, read_jsonl, to_messages  # noqa: E402


class SftDataset(Dataset):
    def __init__(self, rows, tok, max_len: int):
        self.items = []
        skipped = 0
        for s in rows:
            prompt = tok.apply_chat_template(to_messages(s, with_answer=False), tokenize=False,
                                             add_generation_prompt=True)
            full = prompt + s["answer"] + tok.eos_token
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            ids = tok(full, add_special_tokens=False)["input_ids"]
            if len(ids) > max_len:
                skipped += 1
                continue
            labels = [-100] * len(p_ids) + ids[len(p_ids):]
            self.items.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})
        if skipped:
            print(f"[data] 超过 {max_len} token 跳过 {skipped} 条")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="基座模型目录（Qwen2.5-*-Instruct）")
    ap.add_argument("--out", default="outputs/qwen-risk-lora")
    ap.add_argument("--train", default=str(SFT_DIR / "train.jsonl"))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if bf16 else torch.float16,
                                                 trust_remote_code=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    ds = SftDataset(read_jsonl(Path(args.train)), tok, args.max_len)
    print(f"[data] 训练样本 {len(ds)} 条")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
            lr_scheduler_type="cosine", warmup_ratio=0.05, weight_decay=0.0,
            logging_steps=10, save_strategy="epoch", save_total_limit=1, report_to="none",
            bf16=bf16, fp16=not bf16, gradient_checkpointing=True, remove_unused_columns=False,
            dataloader_num_workers=2, seed=args.seed),
        train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"[done] LoRA 适配器已保存到 {args.out}")


if __name__ == "__main__":
    main()
