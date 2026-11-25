"""训练 / 评估 / 线上服务共用的提示词与格式。

必须与风控系统 backend/app/services/rag.py 的 SYSTEM_PROMPT、call_llm 拼接格式保持一致，
否则模型在线上看到的输入分布和训练时不同，忠实引用效果会打折。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
CORPUS_FILE = ROOT / "data" / "corpus" / "kb.json"
SFT_DIR = ROOT / "data" / "sft"

SYSTEM_PROMPT = (
    "你是电商风控审核助手。只能依据【参考资料】回答，每个结论后用 [编号] 标注出处；"
    "资料中没有的信息不得编造；如果资料不足以回答，直接回答：根据现有资料无法确定。"
)
REFUSAL = "根据现有资料无法确定"


def format_refs(refs: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{i}]《{r['title']}》（{r['doc_code']}）：{r['content']}" for i, r in enumerate(refs, 1))


def user_prompt(question: str, refs: list[dict[str, Any]]) -> str:
    return f"【参考资料】\n{format_refs(refs)}\n\n【问题】\n{question}"


def to_messages(sample: dict[str, Any], with_answer: bool = True) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(sample["question"], sample["refs"])}]
    if with_answer:
        msgs.append({"role": "assistant", "content": sample["answer"]})
    return msgs


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_corpus() -> list[dict[str, Any]]:
    return json.loads(CORPUS_FILE.read_text(encoding="utf-8"))
