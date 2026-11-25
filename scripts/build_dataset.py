"""生成训练集 / 留出评估集（CPU，秒级）。

    python scripts/build_dataset.py            # 默认种子
    python scripts/build_dataset.py --seed 7   # 换一批数据

输出：
  data/sft/train.jsonl   约 1000 条，含 corrective
  data/sft/eval.jsonl    100 条留出集（faithful 34 / multi 33 / refuse 33）
  data/sft/train_chat.jsonl  OpenAI messages 格式，方便 LLaMA-Factory / ms-swift 等框架直接使用
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune.common import SFT_DIR, to_messages, write_jsonl  # noqa: E402
from finetune.datagen import Builder  # noqa: E402
from finetune.metrics import aggregate, score_one  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261122)
    args = ap.parse_args()

    train, evals = Builder(args.seed).build()
    write_jsonl(SFT_DIR / "train.jsonl", train)
    write_jsonl(SFT_DIR / "eval.jsonl", evals)
    write_jsonl(SFT_DIR / "train_chat.jsonl", ({"messages": to_messages(s)} for s in train))

    # 自检：标准答案自己跑指标必须满分，否则说明构造逻辑有错
    gold = aggregate([score_one(s, s["answer"]) for s in evals])
    print(f"训练集 {len(train)} 条：{dict(Counter(s['type'] for s in train))}")
    print(f"留出集 {len(evals)} 条：{dict(Counter(s['type'] for s in evals))}")
    print(f"标准答案自检：不编造率 {gold['不编造率']}%，引用正确率 {gold['引用正确率']}%，"
          f"正确拒答率 {gold['正确拒答率']}%，误拒率 {gold['误拒率']}%")
    if gold["不编造率"] < 100 or gold["引用正确率"] < 100 or gold["正确拒答率"] < 100 or gold["误拒率"] > 0:
        sys.exit("标准答案自检未通过")


if __name__ == "__main__":
    main()
