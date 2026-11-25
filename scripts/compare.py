"""对比微调前后的评估结果，打印表格并写 reports/对比.md。

    python scripts/compare.py reports/before.json reports/after.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finetune.metrics import METRIC_VERSION  # noqa: E402

KEYS = [("不编造率", True), ("引用正确率", True), ("引用召回", True), ("正确拒答率", True), ("误拒率", False)]


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = (json.loads(Path(p).read_text(encoding="utf-8"))["summary"] for p in sys.argv[1:])
    if any(x.get("backend") == "gold" or not x.get("model") for x in (a, b)):
        sys.exit("拒绝生成训练对比：gold 是标准答案自检，必须提供真实模型推理报告。")
    if a.get("dataset_sha256") != b.get("dataset_sha256") or not a.get("dataset_sha256"):
        sys.exit("拒绝比较：评估数据指纹缺失或不一致。")
    if any(x.get("metric_version") != METRIC_VERSION for x in (a, b)):
        sys.exit(f"拒绝比较：两份报告必须使用当前指标版本 {METRIC_VERSION}，请重新评估。")
    lines = ["| 指标 | 微调前 | 微调后 | 变化 |", "| --- | --- | --- | --- |"]
    for k, up in KEYS:
        d = round(b[k] - a[k], 1)
        good = (d > 0) == up if d else None
        mark = "" if good is None else (" ✅" if good else " ⚠️")
        lines.append(f"| {k} | {a[k]}% | {b[k]}% | {d:+}{mark} |")
    head = (f"# 微调前后对比（留出集 {b['n']} 条）\n\n- 微调前：{a['model']}\n"
            f"- 微调后：{b['model']} + LoRA `{b['lora'] or '-'}`\n\n")
    text = head + "\n".join(lines) + "\n\n误拒率越低越好，其余越高越好。\n"
    out = Path(sys.argv[2]).parent / "对比.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()
