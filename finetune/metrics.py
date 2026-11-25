"""三项评估指标（说明书第 5 节）：不编造率、引用正确率、正确拒答率。

全部是规则判定，不需要裁判大模型，结果可复现：
- 不编造率：答案中出现的数字、规则 / 判例 / 政策编号，必须能在标准答案、检索片段或样本 allowed
  （如分值求和结果）里找到；另外检查分值与加减方向是否有依据。统计全部样本。
- 引用正确率：仅统计应作答样本。答案至少有一个 [n]，且所有 [n] 都落在标准支撑片段内才算对。
  同时要求无已检出的事实错误且没有拒答。另给引用召回作参考。
- 正确拒答率：仅统计应拒答样本，答案里出现拒答表述即算对。另给误拒率（应作答却拒答）。
这是有限的规则检查，不是通用语义蕴含判断；满分不等于没有任何编造。
"""
from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

METRIC_VERSION = "9-bare-and-chinese-scores"
SCORE_SIGNS = str.maketrans({**dict.fromkeys("−－﹣‐‑‒–—负", "-"), **dict.fromkeys("＋﹢正", "+")})
CITE = re.compile(r"\[(\d+)\]")
CODE = re.compile(r"(?<![A-Za-z0-9])[RPCFM]-\d{3,4}(?!\d)")
NUM = re.compile(r"\d+(?:\.\d+)?")
REFUSE = re.compile(r"无法确定|无法回答|资料不足|没有(相关|足够)(资料|信息)|无参考")
CHINESE_DIGITS = str.maketrans(dict(zip("零〇一二两三四五六七八九", "001223456789")))
CHINESE_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}
SCORE_NUMBER = r"(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万]+(?:点[零〇一二两三四五六七八九]+)?)"
SCORE_CLAIM = re.compile(
    r"(?<![\d.零〇一二两三四五六七八九十百千万点])"
    r"(?:(?P<op>加|增加|减|减少|扣|扣除|累计|合计|总计|总分|风险分|得分|评分|分值|分数|成绩)"
    r"\s*(?:为|是|[:：])?\s*[(（]?\s*)?"
    rf"([-+]?)\s*({SCORE_NUMBER})(?![\d.零〇一二两三四五六七八九十百千万点])"
    # 无关键词时必须有“分”；分钟、百分之等不属于风险分。带关键词的加减项优先整段匹配。
    r"(?!\s*分[钟之钱])\s*(?(op)分?|分)")


def score_value(sign: str, number: str) -> Decimal:
    """只转换分值匹配内的常用中文数词，金额、时长等正文不作改写。"""
    number = number.translate(CHINESE_DIGITS)
    if any(unit in number for unit in CHINESE_UNITS):
        whole, dot, fraction = number.partition("点")
        total = section = digit = 0
        for char in whole:
            if char.isdigit():
                digit = digit * 10 + int(char)
            elif char == "万":
                total += (section + digit or 1) * 10000
                section = digit = 0
            else:
                section += (digit or 1) * CHINESE_UNITS[char]
                digit = 0
        number = str(total + section + digit) + ("." + fraction if dot else "")
    return Decimal(sign + number.replace("点", "."))


def canonical_score(value: Decimal) -> str:
    return format(value.normalize(), "f") if value else "0"


def score_claims(text: str) -> set[tuple[str, str]]:
    """绑定数值与方向，防止把窗口次数当分数、把加分改为减分。"""
    return {("plus" if op in {"加", "增加"} else "minus" if op in {"减", "减少", "扣", "扣除"}
             else "total", canonical_score(score_value(sign, n)))
            for op, sign, n in SCORE_CLAIM.findall(text.translate(SCORE_SIGNS))}


def allowed_totals(sample: dict[str, Any]) -> set[str]:
    # 只累加标注答案里已确认的命中项，不累加检索到的所有候选规则。
    gold = sample.get("answer", "")
    explicit = {n for op, n in score_claims(gold) if op == "total"}
    annotated = {canonical_score(Decimal(str(n))) for n in sample.get("allowed", [])}
    if explicit:
        return explicit | annotated
    claims = SCORE_CLAIM.findall(gold.translate(SCORE_SIGNS))
    terms = [score_value(sign, n) * (1 if op in {"加", "增加"} else -1)
             for op, sign, n in claims if op in {"加", "增加", "减", "减少", "扣", "扣除"}]
    if terms:
        raw = sum(terms)
        annotated.add(canonical_score(max(Decimal(0), min(Decimal(100), raw))))
    return annotated


def unsupported_claims(sample: dict[str, Any], answer: str) -> list[str]:
    # 问题可能包含诱导性错误陈述，不能作为事实依据。
    evidence = "\n".join(r["content"] for i, r in enumerate(sample["refs"], 1)
                         if i in sample.get("support", []))
    evidence += "\n" + sample.get("answer", "")
    supported = score_claims(evidence) | {("total", n) for n in allowed_totals(sample)}
    return [f"{op}:{n}" for op, n in sorted(score_claims(answer) - supported)]


def is_refusal(answer: str) -> bool:
    return bool(REFUSE.search(answer))


def unsupported_tokens(sample: dict[str, Any], answer: str) -> list[str]:
    ctx = "\n".join(f"{r['doc_code']} {r['title']} {r['content']}" for r in sample["refs"])
    # 标准答案含经标注确认的案件输入（例如第 2 次退款），不信任问题中的任意数值。
    ctx += "\n" + sample.get("answer", "")
    allowed = set(sample.get("allowed", []))
    body = CITE.sub(" ", answer)
    codes = set(CODE.findall(body))
    ctx_codes = set(CODE.findall(ctx))
    bad = [c for c in codes if c not in ctx_codes]
    body = CODE.sub(" ", body)
    ctx_nums = set(NUM.findall(CODE.sub(lambda m: " " + m.group()[2:] + " ", ctx))) | allowed
    ctx_nums |= {n.lstrip("-") for _, n in score_claims(ctx)}
    totals = allowed_totals(sample)
    bad += [n for n in set(NUM.findall(body))
            if n not in ctx_nums and canonical_score(Decimal(n)) not in ctx_nums | totals]
    return sorted(bad)


def score_one(sample: dict[str, Any], answer: str) -> dict[str, Any]:
    cites = {int(c) for c in CITE.findall(answer)}
    refused = is_refusal(answer)
    bad = unsupported_tokens(sample, answer)
    bad += unsupported_claims(sample, answer)
    if not answer.strip():
        bad.append("empty_answer")
    r: dict[str, Any] = {"id": sample["id"], "type": sample["type"], "refused": refused,
                         "fabricated": bool(bad), "unsupported": bad, "cites": sorted(cites)}
    if sample["answerable"]:
        sup = set(sample["support"])
        r["cite_ok"] = bool(cites) and cites <= sup and not bad and not refused
        r["cite_recall"] = len(cites & sup) / len(sup) if sup else 0.0
    return r


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ans = [r for r in rows if "cite_ok" in r]
    ref = [r for r in rows if "cite_ok" not in r]

    def pct(xs: list[bool]) -> float:
        return round(100 * sum(xs) / len(xs), 1) if xs else 0.0

    return {
        "metric_version": METRIC_VERSION,
        "limitations": "规则检查数字、编号、分值方向及引用范围；不等于完整语义忠实度，需人工复核。",
        "n": len(rows),
        "不编造率": pct([not r["fabricated"] for r in rows]),
        "引用正确率": pct([r["cite_ok"] for r in ans]),
        "引用召回": round(100 * sum(r["cite_recall"] for r in ans) / len(ans), 1) if ans else 0.0,
        "正确拒答率": pct([r["refused"] for r in ref]),
        "误拒率": pct([r["refused"] for r in ans]),
        "by_type": {t: {"n": len(g), "不编造率": pct([not r["fabricated"] for r in g])}
                    for t in sorted({r["type"] for r in rows}) for g in [[r for r in rows if r["type"] == t]]},
    }
