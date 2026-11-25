"""从风控系统同步知识库语料到 data/corpus/kb.json。

语料 = 风控系统的政策 / 判例 / FAQ（knowledge_base.json）+ 12 条默认规则生成的规则说明，
与线上 RAG 检索的是同一份内容。结果文件已提交在仓库里，云 GPU 机器上不需要风控项目；
只有风控系统的知识库或规则改了，才需要在本机重跑：

    python scripts/sync_corpus.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RISK_BACKEND = ROOT.parent / "lanhong_小盟_电商风控" / "backend"
OUT = ROOT / "data" / "corpus" / "kb.json"


class _Rule:  # 只为复用 rag.rule_doc 的文案，不连数据库
    def __init__(self, code, name, event_types, expression, score, description):
        self.code, self.name, self.event_types = code, name, event_types
        self.expression, self.score, self.description = expression, score, description


def main() -> None:
    if not RISK_BACKEND.exists():
        sys.exit(f"找不到风控系统后端：{RISK_BACKEND}")
    sys.path.insert(0, str(RISK_BACKEND))
    from app.services.bootstrap import DEFAULT_RULES  # noqa: E402
    from app.services.rag import KB_FILE, rule_doc  # noqa: E402

    docs = json.loads(Path(KB_FILE).read_text(encoding="utf-8"))
    docs += [rule_doc(_Rule(*r)) for r in DEFAULT_RULES]
    # 规则分值单独存一份，数据构造时用来算总分
    for d, r in zip(docs[-len(DEFAULT_RULES):], DEFAULT_RULES):
        d["score"] = r[4]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(docs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已写入 {OUT}：{len(docs)} 篇（规则 {len(DEFAULT_RULES)} 条）")


if __name__ == "__main__":
    main()
