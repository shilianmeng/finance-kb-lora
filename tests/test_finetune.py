"""数据构造、指标、提示词一致性测试（CPU，无需 torch）。

    python -m unittest discover -s tests -t .
"""
from __future__ import annotations

import re
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from finetune.common import REFUSAL, SYSTEM_PROMPT, format_refs, to_messages  # noqa: E402
from finetune.datagen import Builder  # noqa: E402
from finetune.metrics import aggregate, is_refusal, score_one, unsupported_tokens  # noqa: E402

RISK_BACKEND = ROOT.parent / "lanhong_小盟_电商风控" / "backend"


class DatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.evals = Builder(20261122).build()

    def test_sizes_and_mix(self):
        self.assertGreaterEqual(len(self.train), 950)
        self.assertEqual(len(self.evals), 100)
        c = Counter(s["type"] for s in self.train)
        for t in ("faithful", "multi", "refuse", "corrective"):
            self.assertGreater(c[t], 100, t)
        self.assertEqual(set(Counter(s["type"] for s in self.evals)), {"faithful", "multi", "refuse"})

    def test_no_leak_between_train_and_eval(self):
        key = lambda s: tuple(r["doc_code"] for r in s["refs"])  # noqa: E731
        for e in self.evals:
            for t in self.train:
                if key(t) == key(e):
                    self.assertNotIn(e["question"], t["question"], (e["id"], t["id"]))

    def test_deterministic(self):
        t2, e2 = Builder(20261122).build()
        self.assertEqual([s["answer"] for s in e2], [s["answer"] for s in self.evals])

    def test_sample_shape(self):
        for s in self.train + self.evals:
            self.assertTrue(s["refs"], s["id"])
            if s["answerable"]:
                self.assertTrue(set(s["support"]) <= set(range(1, len(s["refs"]) + 1)), s["id"])
            elif s["type"] == "refuse":
                self.assertIn(REFUSAL, s["answer"])

    def test_gold_scores_full_marks(self):
        g = aggregate([score_one(s, s["answer"]) for s in self.evals])
        self.assertEqual((g["不编造率"], g["引用正确率"], g["正确拒答率"], g["误拒率"]), (100.0, 100.0, 100.0, 0.0))


class MetricsTest(unittest.TestCase):
    S = {"id": "x", "type": "faithful", "answerable": True, "support": [1], "allowed": [],
         "question": "R-001 命中扣多少分？",
         "refs": [{"doc_code": "R-001", "title": "新设备高频下单", "content": "1 小时内下单 5 次以上，加 30 分。"},
                  {"doc_code": "P-003", "title": "处置规范", "content": "高风险订单进入人工审核。"}]}

    def test_faithful_answer(self):
        r = score_one(self.S, "命中后加 30 分 [1]。")
        self.assertFalse(r["fabricated"])
        self.assertTrue(r["cite_ok"])

    def test_fabricated_number_and_code(self):
        self.assertEqual(unsupported_tokens(self.S, "加 40 分 [1]，同时触发 R-099。"), ["40", "R-099"])

    def test_wrong_or_missing_citation(self):
        self.assertFalse(score_one(self.S, "加 30 分 [2]。")["cite_ok"])
        self.assertFalse(score_one(self.S, "加 30 分。")["cite_ok"])

    def test_reversed_or_misattributed_score(self):
        for answer in ("减 30 分 [1]。", "加 5 分 [1]。", "", "无法确定，但加 40 分 [1]。"):
            r = score_one(self.S, answer)
            self.assertTrue(r["fabricated"], answer)
            self.assertFalse(r["cite_ok"], answer)

    def test_refusal(self):
        self.assertTrue(is_refusal("根据现有资料无法确定。"))
        refuse = dict(self.S, type="refuse", answerable=False, support=[])
        g = aggregate([score_one(refuse, "根据现有资料无法确定。"), score_one(self.S, "根据现有资料无法确定。")])
        self.assertEqual(g["正确拒答率"], 100.0)
        self.assertEqual(g["误拒率"], 100.0)

    def test_verified_total_variants(self):
        sample = dict(self.S, answer="命中规则，加 40 分 [1]。")
        for text in ("本次风险分 40 分", "累计 40 分", "加 40 分，风险分 40 分", "总分为40分"):
            result = score_one(sample, text + " [1]")
            self.assertFalse(result["fabricated"], result)
            self.assertTrue(result["cite_ok"])
        self.assertTrue(score_one(sample, "风险分 30 分 [1]")["fabricated"])

    def test_total_counts_equal_scoring_rules_separately(self):
        sample = dict(self.S, answer="第一项加 40 分；第二项加 40 分 [1]。")
        self.assertFalse(score_one(sample, "合计 80 分 [1]")["fabricated"])
        self.assertTrue(score_one(sample, "合计 40 分 [1]")["fabricated"])

    def test_total_clamps_and_subtracts(self):
        for gold, total in (("加 80 分，加 40 分", 100), ("加 40 分，减 10 分", 30),
                            ("加 10 分，减 40 分", 0)):
            sample = dict(self.S, answer=gold)
            self.assertFalse(score_one(sample, f"累计 {total} 分 [1]")["fabricated"])
        sample = dict(self.S, answer="加 40 分，减 10 分")
        self.assertTrue(score_one(sample, "累计 50 分 [1]")["fabricated"])

    def test_retrieved_candidates_do_not_establish_total(self):
        self.assertTrue(score_one(self.S, "总分 30 分 [1]")["fabricated"])
        sample = dict(self.S, allowed=["40"])
        self.assertFalse(score_one(sample, "总分 40 分 [1]")["fabricated"])

    def test_signed_scores_preserve_sign(self):
        sample = dict(self.S, answer="减 30 分 [1]")
        for answer in ("本次风险分 -30 分 [1]", "累计 -30.0 分 [1]",
                       "减 -30 分 [1]", "加 -30 分 [1]"):
            with self.subTest(answer=answer):
                result = score_one(sample, answer)
                self.assertTrue(result["fabricated"], result)
                self.assertFalse(result["cite_ok"], result)
        for answer in ("风险分 0 分 [1]", "减 +30 分 [1]"):
            self.assertFalse(score_one(sample, answer)["fabricated"])
        sample = dict(self.S, answer="加 30 分 [1]")
        self.assertFalse(score_one(sample, "风险分 +30.0 分 [1]")["fabricated"])
        self.assertTrue(score_one(sample, "风险分 +5 分 [1]")["fabricated"])

    def test_unicode_and_chinese_score_signs(self):
        sample = dict(self.S, answer="加 30 分 [1]")
        for text in ("风险分 −30分", "风险分 －30分", "风险分 ＋5分", "风险分为负30分",
                     "加负30分", "减＋30分"):
            with self.subTest(text=text):
                result = score_one(sample, text + " [1]")
                self.assertTrue(result["fabricated"], result)
                self.assertFalse(result["cite_ok"])
        for text in ("风险分＋30分", "加正30分"):
            self.assertFalse(score_one(sample, text + " [1]")["fabricated"])
        for gold in ("加＋40分，减正10分", "加40分，加负10分"):
            self.assertFalse(score_one(dict(self.S, answer=gold), "累计30分 [1]")["fabricated"])

    def test_score_signs_allow_whitespace(self):
        sample = dict(self.S, answer="加 30 分 [1]")
        for sign in ("-", "−", "负", "+", "＋", "正"):
            for gap in (" ", "\t", "\u00a0", "\u3000"):
                value = 30 if sign in ("-", "−", "负") else 5
                text = f"风险分为{sign}{gap}{value}分 [1]"
                with self.subTest(text=text):
                    result = score_one(sample, text)
                    self.assertTrue(result["fabricated"], result)
                    self.assertFalse(result["cite_ok"])
        for text in ("风险分 + 30.0 分", "加正\t30分", "风险分＋\u300030分"):
            with self.subTest(text=text):
                self.assertTrue(score_one(sample, text + " [1]")["cite_ok"])
        for gold, total in (("加＋ 40分，减正\t10分", 30), ("加40分，加负 10分", 30),
                            ("加＋ 40分，加正 40分", 80)):
            with self.subTest(gold=gold):
                self.assertTrue(score_one(dict(self.S, answer=gold), f"累计{total}分 [1]")["cite_ok"])
        evidence = dict(self.S, refs=[dict(self.S["refs"][0], content="命中加正 30分。")])
        self.assertTrue(score_one(evidence, "加30分 [1]")["cite_ok"])

    def test_additional_unicode_score_signs(self):
        sample = dict(self.S, answer="加 30 分 [1]")
        for sign in ("﹣", "‐", "‑", "‒", "–", "—"):
            for gap in ("", " "):
                with self.subTest(sign=sign, gap=gap):
                    result = score_one(sample, f"风险分 {sign}{gap}30分 [1]")
                    self.assertTrue(result["fabricated"], result)
                    self.assertFalse(result["cite_ok"])
                    gold = f"加40分，加{sign}{gap}10分"
                    self.assertTrue(score_one(dict(self.S, answer=gold), "累计30分 [1]")["cite_ok"])
        self.assertTrue(score_one(sample, "风险分﹢ 5分 [1]")["fabricated"])
        self.assertTrue(score_one(sample, "风险分﹢ 30分 [1]")["cite_ok"])

    def test_score_aliases_preserve_value_and_sign(self):
        sample = dict(self.S, answer="加 30 分 [1]")
        for label in ("本次得分", "本次评分", "风险分值", "本次分数", "本次成绩", "风险分数"):
            for value in ("-30", "− 30", "负 30", "+5", "＋ 5"):
                with self.subTest(label=label, value=value):
                    result = score_one(sample, f"{label}为{value}分 [1]")
                    self.assertTrue(result["fabricated"], result)
                    self.assertFalse(result["cite_ok"])
            for value in ("30", "+ 30.0", "正 30"):
                with self.subTest(label=label, value=value):
                    result = score_one(sample, f"{label}是{value}分 [1]")
                    self.assertFalse(result["fabricated"], result)
                    self.assertTrue(result["cite_ok"])

    def test_score_aliases_share_total_evidence_rules(self):
        for label in ("得分", "评分", "分值", "分数", "成绩"):
            with self.subTest(label=label):
                # 检索到的候选加分不能独自证明本次总分。
                result = score_one(self.S, f"{label}30分 [1]")
                self.assertTrue(result["fabricated"], result)
                self.assertFalse(result["cite_ok"])
                sample = dict(self.S, answer="命中加40分 [1]")
                self.assertTrue(score_one(sample, f"{label}30分 [1]")["fabricated"])
                self.assertTrue(score_one(sample, f"{label}40分 [1]")["cite_ok"])
                # 标准答案与支撑片段中的明确分值使用同一套识别规则。
                sample = dict(self.S, answer=f"加30分，{label}为40分 [1]")
                self.assertTrue(score_one(sample, "总分40分 [1]")["cite_ok"])
                self.assertTrue(score_one(sample, "总分30分 [1]")["fabricated"])
                sample = dict(self.S, refs=[dict(self.S["refs"][0], content=f"{label}为40分。")])
                self.assertTrue(score_one(sample, "总分40分 [1]")["cite_ok"])
                self.assertTrue(score_one(sample, "总分-40分 [1]")["fabricated"])

    def test_parenthesized_score_claims(self):
        sample = dict(self.S, answer="加30分 [1]")
        for label in ("得分", "评分", "分值", "分数", "成绩", "风险分数", "总分"):
            for left, right in (("(", ")"), ("（", "）")):
                for value, valid in (("-30", False), ("负 30", False), ("＋ 5", False),
                                     ("30", True), ("正 30.0", True)):
                    text = f"{label}{left} {value} 分{right} [1]"
                    with self.subTest(text=text):
                        result = score_one(sample, text)
                        self.assertEqual(result["fabricated"], not valid, result)
                        self.assertEqual(result["cite_ok"], valid, result)

    def test_parenthesized_score_evidence_and_totals(self):
        for left, right in (("(", ")"), ("（", "）")):
            for label in ("得分", "评分", "分值", "分数", "成绩"):
                text = f"{label}为{left}正 40 分{right}"
                for source in ("gold", "evidence"):
                    with self.subTest(text=text, source=source):
                        sample = (dict(self.S, answer=text) if source == "gold" else
                                  dict(self.S, refs=[dict(self.S["refs"][0], content=text)]))
                        self.assertTrue(score_one(sample, "总分40分 [1]")["cite_ok"])
                        self.assertTrue(score_one(sample, "总分-40分 [1]")["fabricated"])
            for gold, total in ((f"加{left}40分{right}，减{left}10分{right}", 30),
                                (f"加{left}40分{right}，加{left}40分{right}", 80),
                                (f"加{left}10分{right}，加{left}负 10分{right}", 0)):
                with self.subTest(gold=gold):
                    sample = dict(self.S, answer=gold)
                    self.assertTrue(score_one(sample, f"总分{total}分 [1]")["cite_ok"])
                    self.assertTrue(score_one(sample, "总分40分 [1]")["fabricated"])

    def test_zero_total_requires_evidence(self):
        # 有依据的 0 合法；没有加减项不代表实际总分为 0。
        for sample in (dict(self.S, allowed=["0"]), dict(self.S, answer="总分0分 [1]"),
                       dict(self.S, answer="加30分，减30分 [1]"),
                       dict(self.S, refs=[dict(self.S["refs"][0], content="本次总分0分。")])):
            with self.subTest(sample=sample):
                result = score_one(sample, "本次得分0分 [1]")
                self.assertFalse(result["fabricated"], result)
                self.assertTrue(result["cite_ok"], result)
                self.assertTrue(score_one(sample, "本次得分30分 [1]")["fabricated"])
        for sample in (self.S, dict(self.S, answer="正常用户，处置为放行 [1]"),
                       dict(self.S, answer="不触发该规则 [1]"),
                       dict(self.S, answer="加30分 [1]"),
                       dict(self.S, answer="总分30分 [1]"),
                       dict(self.S, answer="根据现有资料无法确定。", answerable=False)):
            with self.subTest(sample=sample):
                self.assertTrue(score_one(sample, "本次得分0分 [1]")["fabricated"])

    def test_bare_scores_preserve_value_and_sign(self):
        sample = dict(self.S, answer="命中 R-001，60 分 [1]")
        for sign in ("-", "−", "－", "﹣", "‐", "‑", "‒", "–", "—", "负"):
            for gap in ("", " ", "\t", "\u00a0", "\u3000"):
                with self.subTest(sign=sign, gap=gap):
                    result = score_one(sample, f"命中 R-001，{sign}{gap}60 分 [1]")
                    self.assertTrue(result["fabricated"], result)
                    self.assertFalse(result["cite_ok"])
        for value, valid in (("60", True), ("+60.0", True), ("正 60", True),
                             ("＋ 60", True), ("﹢60", True), ("+30", False),
                             ("30", False), ("0", False)):
            with self.subTest(value=value):
                result = score_one(sample, f"命中 R-001，（{value} 分）[1]")
                self.assertEqual(result["fabricated"], not valid, result)
                self.assertEqual(result["cite_ok"], valid)

    def test_bare_scores_share_evidence_without_double_counting(self):
        for text in ("命中 R-001，60 分", "命中 R-001，+60 分"):
            for source in ("gold", "evidence"):
                with self.subTest(text=text, source=source):
                    sample = (dict(self.S, answer=text) if source == "gold" else
                              dict(self.S, refs=[dict(self.S["refs"][0], content=text)]))
                    self.assertTrue(score_one(sample, "总分60分 [1]")["cite_ok"])
                    self.assertTrue(score_one(sample, "-60分 [1]")["fabricated"])
        for gold, total in (("加30分，加30分", 60), ("加40分，减10分", 30),
                            ("加30分，减30分", 0)):
            with self.subTest(gold=gold):
                sample = dict(self.S, answer=gold)
                self.assertTrue(score_one(sample, f"+{total}分 [1]")["cite_ok"])
                self.assertTrue(score_one(sample, "+5分 [1]")["fabricated"])
        self.assertTrue(score_one(self.S, "+30分 [1]")["fabricated"])
        for text in ("等待60分钟，比例为百分之60。", "累计60 分钟", "增加60 分钱"):
            with self.subTest(text=text):
                sample = dict(self.S, answer=text)
                self.assertTrue(score_one(sample, "+60分 [1]")["fabricated"])
                self.assertTrue(score_one(sample, text + " [1]")["cite_ok"])

    def test_chinese_score_numbers(self):
        sample = dict(self.S, answer="命中 R-001，60 分 [1]")
        for value, valid in (("六十", True), ("六〇", True), ("正六十", True),
                             ("负六十", False), ("负 六十", False),
                             ("−六十", False), ("三十", False), ("零", False)):
            for label in ("风险分为", "命中 R-001，"):
                with self.subTest(value=value, label=label):
                    result = score_one(sample, f"{label}{value}分 [1]")
                    self.assertEqual(result["fabricated"], not valid, result)
                    self.assertEqual(result["cite_ok"], valid)
        for text, total in (("六十分", "60"), ("得分（一百分）", "100"),
                            ("加四十分，减十分", "30"), ("加两分，加三点五分", "5.5"),
                            ("加四十分，加负十分", "30")):
            with self.subTest(text=text):
                sample = dict(self.S, answer=text)
                self.assertTrue(score_one(sample, f"总分{total}分 [1]")["cite_ok"])
        sample = dict(self.S, refs=[dict(self.S["refs"][0], content="本次六十分。")])
        self.assertTrue(score_one(sample, "总分60分 [1]")["cite_ok"])
        self.assertTrue(score_one(sample, "风险分为负六十分 [1]")["fabricated"])

    def test_unsupported_scores_use_plain_decimal_notation(self):
        sample = dict(self.S, answer="加30分 [1]")
        for value, expected in (("-60", "total:-60"), ("60.00", "total:60"),
                                ("0.000001", "total:0.000001")):
            with self.subTest(value=value):
                self.assertIn(expected, score_one(sample, f"总分{value}分 [1]")["unsupported"])

    def test_comparison_requires_current_metric_version(self):
        current = aggregate([])["metric_version"]
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "before.json", Path(tmp) / "after.json"
            out = Path(tmp) / "对比.md"
            summary = dict(aggregate([]), backend="hf", model="model", dataset_sha256="same", lora=None)
            for versions in (("2-score-claims", "2-score-claims"), ("4-signed-scores", "4-signed-scores"),
                             ("5-unicode-signed-scores", "5-unicode-signed-scores"),
                             ("6-flexible-score-signs", "6-flexible-score-signs"),
                             ("7-score-aliases", "7-score-aliases"),
                             ("8-parenthesized-score-aliases", "8-parenthesized-score-aliases"), (None, None),
                             (current, "old"), ("old", current), (current, current)):
                with self.subTest(versions=versions):
                    out.write_text("unchanged", encoding="utf-8")
                    for path, version in zip((a, b), versions):
                        path.write_text(json.dumps({"summary": dict(summary, metric_version=version)}), encoding="utf-8")
                    result = subprocess.run([sys.executable, str(ROOT / "scripts/compare.py"), str(a), str(b)],
                                            capture_output=True, encoding="utf-8",
                                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
                    if versions == (current, current):
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertIn("微调前后对比", out.read_text(encoding="utf-8"))
                    else:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn("拒绝比较", result.stderr)
                        self.assertEqual(out.read_text(encoding="utf-8"), "unchanged")

    def test_gold_cannot_be_presented_as_training_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "gold.json"
            report.write_text(json.dumps({"summary": {"backend": "gold", "model": ""}}), encoding="utf-8")
            result = subprocess.run([sys.executable, str(ROOT / "scripts/compare.py"), str(report), str(report)],
                                    capture_output=True, encoding="utf-8",
                                    env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((Path(tmp) / "对比.md").exists())


@unittest.skipUnless((RISK_BACKEND / "app" / "services" / "rag.py").exists(), "风控项目不在同级目录")
class OnlineConsistencyTest(unittest.TestCase):
    """训练格式必须和风控系统线上调用一模一样。"""

    def test_system_prompt_and_ref_format(self):
        src = (RISK_BACKEND / "app" / "services" / "rag.py").read_text(encoding="utf-8")
        body = re.search(r"SYSTEM_PROMPT\s*=\s*\((.*?)\)\n", src, re.S).group(1)
        online = "".join(re.findall(r'"([^"]*)"', body))
        self.assertEqual(online, SYSTEM_PROMPT)
        self.assertIn("[{i}]《{", src)
        self.assertIn("【参考资料】", src)
        self.assertIn("【问题】", src)
        msgs = to_messages(MetricsTest.S, with_answer=False)
        self.assertTrue(msgs[1]["content"].startswith("【参考资料】\n" + format_refs(MetricsTest.S["refs"])[:10]))


if __name__ == "__main__":
    unittest.main()
