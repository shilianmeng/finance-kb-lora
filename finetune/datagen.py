"""训练数据构造：约 1100 条「问题 + 带编号检索片段 → 带 [n] 引用的标准答案」。

| 类型       | 数量 | 说明                                                   |
|------------|------|--------------------------------------------------------|
| faithful   | 300  | 忠实引用正例：答案只复述片段事实，引用准确             |
| corrective | 200  | 忠实引用反例：给出编造数字 / 引错出处 / 资料外结论的   |
|            |      | 草稿答案，模型要指出问题并给出更正答案                 |
| multi      | 300  | 多片段整合：规则分值求和 + 分级标准、判例 + 政策 + 处置|
|            |      | 动作、按条件区分适用范围                               |
| refuse     | 300  | 拒答：无召回 / 召回无关 / 片段相互冲突                 |

语料来自 data/corpus/kb.json（与风控系统线上 RAG 是同一份），另外按 7 类风控场景
随机生成 240 篇带具体数字的合成判例（C-1001 起），让「不编造数字」可以被量化检查。

每条样本字段：
  id, type, question, refs[{doc_code,title,content}], answer,
  support  —— 答案应引用的片段编号（1 起），拒答样本可为空
  answerable —— False 表示标准答案是拒答
  allowed  —— 答案里允许出现、但片段里没有的数字（如规则分值求和结果）
全部由固定随机种子生成，可复现；只用标准库，CPU 上秒级完成。
"""
from __future__ import annotations

import random
import re
from typing import Any, Callable

from finetune.common import REFUSAL, load_corpus

COUNTS = {"faithful": 300, "corrective": 200, "multi": 300, "refuse": 300}
EVAL_COUNTS = {"faithful": 34, "multi": 33, "refuse": 33}      # 留出集 100 条，不含 corrective

DECISION_CN = {"REJECT": "拒绝", "REVIEW": "人工审核", "PASS": "通过"}
ACTION_CN = {"block_order": "拦截订单", "blacklist_user": "账号拉黑", "ban_device": "设备封禁", "release": "放行"}


def decide(score: int) -> str:
    return "REJECT" if score >= 80 else "REVIEW" if score >= 60 else "PASS"


def range_of(decision: str) -> str:
    return {"REJECT": "80-100 分", "REVIEW": "60-79 分", "PASS": "0-59 分"}[decision]


# ====================================================================== 合成判例
SCENARIOS: dict[str, dict[str, Any]] = {
    "wool": {"event": "领券", "policy": "P-005", "fraud": True,
             "rule_sets": [["R-001", "R-002", "R-003"], ["R-001", "R-002"], ["R-002", "R-003"]],
             "actions": ["block_order", "ban_device", "blacklist_user"], "title": "新账号设备农场领券"},
    "scalper": {"event": "下单", "policy": "P-007", "fraud": True,
                "rule_sets": [["R-004", "R-005"], ["R-004", "R-005", "R-003"]],
                "actions": ["block_order", "blacklist_user"], "title": "同地址多账号抢购限购商品"},
    "brushing": {"event": "支付", "policy": "P-009", "fraud": True,
                 "rule_sets": [["R-007", "R-003"], ["R-007", "R-010", "R-003"]],
                 "actions": ["block_order", "blacklist_user"], "title": "同 IP 集中支付刷单"},
    "refund": {"event": "售后退款申请", "policy": "P-006", "fraud": True,
               "rule_sets": [["R-008", "R-009"], ["R-008", "R-011"], ["R-008"]],
               "actions": ["block_order", "blacklist_user"], "title": "多次未收到货恶意退款"},
    "campus": {"event": "支付", "policy": "P-009", "fraud": False,
               "rule_sets": [["R-007", "R-003"], ["R-007"]],
               "actions": ["release"], "title": "公共网络 IP 关联误判"},
    "family": {"event": "登录", "policy": "P-009", "fraud": False,
               "rule_sets": [["R-012"]], "actions": ["release"], "title": "家庭共用设备多账号"},
    "newbig": {"event": "下单", "policy": "P-001", "fraud": False,
               "rule_sets": [["R-006"]], "actions": ["release"], "title": "新账号大额首单"},
}


def _facts_text(sc: str, rng: random.Random) -> tuple[str, list[str]]:
    """返回场景事实描述和其中的关键数字（数字都写进判例正文，答案复述时才算有依据）。"""
    r = rng.randint
    if sc == "wool":
        a, h, d, p = r(6, 20), r(1, 4), r(2, 4), r(1, 3)
        return f"{a} 个注册不满 7 天的账号在 {h} 小时内用 {d} 台设备轮流领取同一张优惠券，共用 {p} 个收货手机号", [a, h, d, p]
    if sc == "scalper":
        a, k = r(4, 9), r(6, 12)
        return f"{a} 个账号 7 天内把限购商品寄往同一收货地址，其中一台设备 1 小时内下单 {k} 次", [a, k]
    if sc == "brushing":
        a, h, lo = r(5, 30), r(1, 6), r(50, 300)
        return f"{a} 个账号在 {h} 小时内经同一 IP 支付同一店铺商品，金额集中在 {lo}-{lo + r(10, 60)} 元，核查发现设备指纹高度重合", [a, h, lo]
    if sc == "refund":
        k, rate, amt = r(3, 7), r(60, 95), r(300, 5000)
        return f"一个账号 30 天内 {k} 次以未收到货申请退款，单笔最高 {amt} 元，物流均显示本人签收，同收货手机号退款率 {rate}%", [k, amt, rate]
    if sc == "campus":
        a = r(8, 40)
        return f"某高校同一校园网出口 IP 24 小时内出现 {a} 个账号支付，核查设备与收货信息均不重合，订单为正常小额消费", [a]
    if sc == "family":
        k, y = r(5, 8), r(2, 8)
        return f"一台平板 24 小时内登录 {k} 个账号，核查为一家人共用，账号注册时间跨越 {y} 年，均为本人正常消费", [k, y]
    d, amt = r(1, 6), r(3000, 9999)
    return f"注册 {d} 天的新账号首单购买 {amt} 元商品，后续按时签收、无其他异常事件", [d, amt]


def make_cases(rules: dict[str, dict], rng: random.Random, n: int = 240) -> list[dict[str, Any]]:
    out = []
    names = list(SCENARIOS)
    for i in range(n):
        sc = names[i % len(names)]
        cfg = SCENARIOS[sc]
        hit = rng.choice(cfg["rule_sets"])
        raw = sum(rules[c]["score"] for c in hit)
        score = min(100, raw)
        dec = decide(score)
        facts, nums = _facts_text(sc, rng)
        concl = "确认欺诈" if cfg["fraud"] else "正常用户"
        acts = "、".join(ACTION_CN[a] for a in cfg["actions"])
        trunc = f"（合计 {raw} 分，截断为 100 分）" if raw > 100 else ""
        code = f"C-{1001 + i}"
        content = (f"{facts}。命中 {'、'.join(hit)}，{score} 分{trunc}，引擎判定{DECISION_CN[dec]}。"
                   f"审核结论：{concl}。处置：{acts}。")
        out.append({"doc_code": code, "category": "case", "title": f"判例：{cfg['title']}", "content": content,
                    "meta": {"scenario": sc, "rules": hit, "score": score, "decision": dec, "conclusion": concl,
                             "actions": acts, "facts": facts, "nums": nums}})
    return out


# ====================================================================== 工具
def first_sentences(text: str, n: int = 2) -> str:
    parts = [p for p in re.split(r"(?<=。)", text) if p.strip()]
    return "".join(parts[:n]).rstrip("。")


def _plain(d: dict) -> dict:
    return {"doc_code": d["doc_code"], "title": d["title"], "content": d["content"]}


class Builder:
    def __init__(self, seed: int = 20261122):
        self.rng = random.Random(seed)
        corpus = load_corpus()
        self.rules = {d["doc_code"]: d for d in corpus if d["category"] == "rule"}
        self.base = {d["doc_code"]: d for d in corpus}
        self.cases = make_cases(self.rules, self.rng)
        self.pool = list(self.base.values()) + self.cases
        self.by_sc: dict[str, list[dict]] = {}
        for c in self.cases:
            self.by_sc.setdefault(c["meta"]["scenario"], []).append(c)
        self.n = 0

    # -------------------------------------------------------------- 片段组装
    def assemble(self, needed: list[dict], k_distract: tuple[int, int] = (1, 3),
                 exclude: set[str] | None = None) -> tuple[list[dict], Callable[[dict], str], list[int]]:
        codes = {d["doc_code"] for d in needed} | (exclude or set())
        cand = [d for d in self.pool if d["doc_code"] not in codes]
        refs = needed + self.rng.sample(cand, self.rng.randint(*k_distract))
        self.rng.shuffle(refs)
        idx = {d["doc_code"]: i + 1 for i, d in enumerate(refs)}
        support = sorted(idx[d["doc_code"]] for d in needed)
        return [_plain(d) for d in refs], (lambda d: f"[{idx[d['doc_code']]}]"), support

    def sample(self, typ: str, question: str, refs: list[dict], answer: str, support: list[int],
               answerable: bool = True, allowed: list[str] | None = None) -> dict[str, Any]:
        self.n += 1
        return {"id": f"{typ}-{self.n:05d}", "type": typ, "question": question, "refs": refs, "answer": answer,
                "support": support, "answerable": answerable, "allowed": allowed or []}

    def rule_line(self, code: str) -> str:
        r = self.rules[code]
        desc = r["content"].split("：", 1)[1].split("。")[0]
        return f"{code}「{r['title'].split(' ', 2)[-1]}」{desc}，命中加 {r['score']} 分"

    # ============================================================== faithful
    POLICY_Q = {
        "P-001": "风险分数是怎么分级的？各分数段分别怎么处理？", "P-002": "案件审核的流程是怎样的？",
        "P-003": "处置动作有哪几种，分别是什么意思？", "P-004": "名单管理有什么规范？",
        "P-005": "优惠券反薅羊毛的政策是什么？", "P-006": "未收到货的退款申请应该怎么处理？",
        "P-007": "限购商品怎么认定黄牛？", "P-008": "用户申诉被误判时怎么处理？",
        "P-009": "使用图谱关联证据时要注意什么？", "P-010": "新增或调整规则之前要做什么？",
    }
    PREFIX = ["", "请问", "审核员想确认：", "麻烦说明一下，"]

    def faithful(self) -> dict:
        kind = self.rng.choice(["why", "why", "rule", "policy", "case"])
        if kind == "why":
            case = self.rng.choice([c for c in self.cases if c["meta"]["decision"] != "PASS"])
            m = case["meta"]
            code = self.rng.choice(m["rules"])
            need = [self.rules[code], case]
            refs, cite, sup = self.assemble(need)
            ev = SCENARIOS[m["scenario"]]["event"]
            q = f"当前案件是一笔{ev}事件，命中了 {code}。为什么被判为高风险？有没有相似判例？"
            a = (f"该事件命中规则 {self.rule_line(code)}{cite(self.rules[code])}。"
                 f"相似判例 {case['doc_code']}：{m['facts']}，引擎判定{DECISION_CN[m['decision']]}，"
                 f"审核结论为{m['conclusion']}，处置为{m['actions']}{cite(case)}。可参照该判例核查后处置。")
            return self.sample("faithful", q, refs, a, sup)
        if kind == "rule":
            code = self.rng.choice(list(self.rules))
            r = self.rules[code]
            refs, cite, sup = self.assemble([r])
            ev = r["content"].split("适用事件：")[-1].rstrip("。")
            q = f"{self.rng.choice(self.PREFIX)}规则 {code} 是什么意思？命中加多少分？"
            a = f"规则 {self.rule_line(code)}，适用事件为 {ev}{cite(r)}。"
            return self.sample("faithful", q, refs, a, sup)
        if kind == "policy":
            code = self.rng.choice(list(self.POLICY_Q) + [c for c in self.base if c.startswith("F-")])
            d = self.base[code]
            q = self.POLICY_Q.get(code) or d["title"].replace("FAQ：", "") + "？"
            refs, cite, sup = self.assemble([d])
            a = f"根据《{d['title']}》：{first_sentences(d['content'])}{cite(d)}。"
            return self.sample("faithful", f"{self.rng.choice(self.PREFIX)}{q}", refs, a, sup)
        case = self.rng.choice(self.cases)
        m = case["meta"]
        refs, cite, sup = self.assemble([case])
        q = f"判例 {case['doc_code']} 命中了哪些规则？最后怎么定性和处置的？"
        a = (f"判例 {case['doc_code']} 命中 {'、'.join(m['rules'])}，{m['score']} 分，引擎判定{DECISION_CN[m['decision']]}；"
             f"审核结论为{m['conclusion']}，处置为{m['actions']}{cite(case)}。")
        return self.sample("faithful", q, refs, a, sup)

    # ============================================================== multi
    def multi(self) -> dict:
        kind = self.rng.choice(["score", "score", "dispose", "refund", "ip"])
        p001 = self.base["P-001"]
        if kind == "score":
            # 只组合同一事件类型上可能同时命中的规则（R-003 等不限事件的规则可与任何事件组合）
            ev = self.rng.choice(["coupon_receive", "order_create", "order_pay", "after_sale_apply", "login"])
            cand = [c for c, r in self.rules.items() if ev in r["tags"][2:] or len(r["tags"]) == 2]
            codes = self.rng.sample(cand, min(len(cand), self.rng.choice([2, 2, 3])))
            grey = self.rng.random() < 0.25
            need = [self.rules[c] for c in codes] + [p001]
            refs, cite, sup = self.assemble(need, (1, 2))
            raw = sum(self.rules[c]["score"] for c in codes) + (20 if grey else 0)
            total = min(100, raw)
            dec = decide(total)
            q = f"某事件同时命中 {'、'.join(codes)}{'，且收货地址在灰名单上' if grey else ''}，总分多少？会怎么判定？"
            parts = [f"{c} 命中加 {self.rules[c]['score']} 分{cite(self.rules[c])}" for c in codes]
            if grey:
                parts.append(f"灰名单命中额外加 20 分{cite(p001)}")
            trunc = f"合计 {raw} 分，总分截断在 0-100 之间，按 100 分计" if raw > 100 else f"合计 {total} 分"
            a = (f"{'；'.join(parts)}。{trunc}{cite(p001)}。按分级标准，{range_of(dec)}判定为"
                 f"{DECISION_CN[dec]}（{dec}）{cite(p001)}。")
            return self.sample("multi", q, refs, a, sup, allowed=[str(raw), str(total)])
        if kind == "dispose":
            case = self.rng.choice(self.cases)
            m = case["meta"]
            pol = self.base[SCENARIOS[m["scenario"]]["policy"]]
            p003 = self.base["P-003"]
            need = [case, pol] + ([p003] if pol is not p003 else [])
            refs, cite, sup = self.assemble(need, (0, 2))
            ev = SCENARIOS[m["scenario"]]["event"]
            q = f"一笔{ev}事件的情况与判例 {case['doc_code']} 类似，结合判例和相关政策，应该怎么处置？"
            a = (f"相似判例 {case['doc_code']} 的审核结论为{m['conclusion']}，处置为{m['actions']}{cite(case)}。"
                 f"相关政策《{pol['title']}》规定：{first_sentences(pol['content'], 1)}{cite(pol)}。"
                 f"处置动作的含义见《{p003['title']}》：{first_sentences(p003['content'], 1)}{cite(p003)}。"
                 f"建议核实与判例情况一致后，参照判例处置。")
            return self.sample("multi", q, refs, a, sup)
        if kind == "refund":
            p006, r008 = self.base["P-006"], self.rules["R-008"]
            proof = self.rng.random() < 0.5
            k = self.rng.randint(1, 6)
            refs, cite, sup = self.assemble([p006, r008], (1, 2))
            q = (f"一笔未收到货退款，物流显示已签收{'且有签收凭证' if proof else '但没有签收凭证'}，"
                 f"这是该账号 30 天内第 {k} 次以未收到货申请退款，应该怎么处理？")
            first = ("物流显示已签收且有签收凭证的，不予退款并引导用户联系快递" if proof
                     else "物流无签收凭证的，正常退款")
            a = f"按退款赔付政策，{first}{cite(p006)}。"
            if k >= 3:
                a += (f"另外，30 天内第 {k} 次以未收到货申请，达到 R-008 的 ≥3 次阈值，命中加 60 分{cite(r008)}；"
                      f"政策要求多次以未收到货为由申请退款的转人工审核，审核期间暂停自动退款{cite(p006)}。")
            else:
                a += f"30 天内第 {k} 次申请，未达到 R-008 要求的 ≥3 次，不触发该规则{cite(r008)}。"
            return self.sample("multi", q, refs, a, sup)
        p009, r007 = self.base["P-009"], self.rules["R-007"]
        case = self.rng.choice(self.by_sc["campus"])
        refs, cite, sup = self.assemble([p009, r007, case], (0, 2))
        q = "一批账号同 IP 支付命中了 R-007，但设备和收货信息都不重合，能直接定性为刷单吗？"
        a = (f"不能直接定性。R-007 只看同一 IP 24 小时内 ≥5 个账号，命中加 30 分{cite(r007)}；"
             f"图谱使用原则指出公共 IP 会造成大量无关联系，{first_sentences(p009['content'], 3).split('。')[-1]}{cite(p009)}。"
             f"相似判例 {case['doc_code']}：{case['meta']['facts']}，审核结论为正常用户，处置为放行{cite(case)}。"
             f"建议核查设备与收货信息后再判断。")
        return self.sample("multi", q, refs, a, sup)

    # ============================================================== refuse
    OUT_OF_SCOPE = [
        ("商户的结算周期是多少天？", "商户结算周期"), ("跨境支付的汇率按哪天的牌价计算？", "跨境支付汇率"),
        ("信用卡分期的手续费率是多少？", "信用卡分期手续费"), ("这个账号持有人的真实姓名是什么？", "账号实名信息"),
        ("快递丢件按什么标准赔付给商家？", "物流丢件赔付标准"), ("这次活动的优惠券面额是多少？", "优惠券面额"),
        ("规则 R-013 的触发条件是什么？", "规则 R-013"), ("判例 C-9999 的审核结论是什么？", "判例 C-9999"),
        ("风控模型用的是 XGBoost 还是 LightGBM？", "机器学习模型选型"), ("审核员的绩效考核标准是什么？", "审核员绩效考核"),
        ("海外仓发货的订单适用哪些规则？", "海外仓订单"), ("直播间下单有单独的风控策略吗？", "直播间风控策略"),
        ("拉黑的账号多久自动解封？", "黑名单自动解封期限"), ("退款到账一般需要几个工作日？", "退款到账时效"),
        ("虚拟商品（充值卡）的风控阈值是多少？", "虚拟商品风控阈值"), ("这个 IP 属于哪个城市？", "IP 归属地"),
        ("上个月拦截了多少笔订单？", "上月拦截统计"), ("风控系统的 QPS 上限是多少？", "系统性能指标"),
        ("会员等级会影响风险分吗？具体减多少分？", "会员等级减分规则"), ("银行卡被盗刷怎么向银行报案？", "盗刷报案流程"),
    ]
    MISSING_ATTR = [("是哪一年上线的", "上线时间"), ("是哪个策略师负责维护的", "负责人"),
                    ("上个月命中了多少次", "命中次数统计"), ("误伤率是多少", "误伤率数据")]

    def refuse(self) -> dict:
        kind = self.rng.choice(["none", "none", "attr", "conflict"])
        if kind == "none":
            q, topic = self.rng.choice(self.OUT_OF_SCOPE)
            excl = {t for t in re.findall(r"[RC]-\d{3,4}", q)}
            refs, _, _ = self.assemble([], (2, 4), exclude=excl)
            a = f"{REFUSAL}。参考资料中没有关于{topic}的内容，建议补充相关资料或人工确认。"
            return self.sample("refuse", f"{self.rng.choice(self.PREFIX)}{q}", refs, a, [], answerable=False)
        if kind == "attr":
            code = self.rng.choice(list(self.rules))
            ask, what = self.rng.choice(self.MISSING_ATTR)
            refs, cite, sup = self.assemble([self.rules[code]])
            q = f"规则 {code} {ask}？"
            a = f"{REFUSAL}。{cite(self.rules[code])} 只说明了 {code} 的触发条件和分值，没有{what}的信息。"
            return self.sample("refuse", q, refs, a, sup, answerable=False)
        code = self.rng.choice(["R-001", "R-004", "R-005", "R-007", "R-008", "R-012"])
        r = self.rules[code]
        m = re.search(r"(≥|超过 ?)(\d+)", r["content"])
        real = int(m.group(2))
        fake = real + self.rng.choice([1, 2, 3, -1]) if real > 2 else real + 2
        alt = dict(r, doc_code=f"M-{code[2:]}", title=f"风控规则手册摘录 {code}",
                   content=r["content"].replace(m.group(0), f"{m.group(1)}{fake}", 1).replace(f"规则 {code}", f"{code}", 1))
        refs, cite, sup = self.assemble([r, alt], (0, 2))
        q = f"{code} 的阈值是多少？达到多少就会命中？"
        a = (f"{REFUSAL}。{cite(r)} 写的是 {m.group(1)}{real}，{cite(alt)} 写的是 {m.group(1)}{fake}，两份资料相互冲突，"
             f"请以规则管理页当前生效的版本为准并人工确认。")
        return self.sample("refuse", q, refs, a, sup, answerable=False)

    # ============================================================== corrective
    def corrective(self) -> dict:
        while True:
            base = self.faithful() if self.rng.random() < 0.5 else self.multi()
            self.n -= 1
            bad, issue = self._corrupt(base)
            if bad:
                break
        q = (f"{base['question']}\n请核查下面的答案是否忠于参考资料，指出问题并给出更正后的答案。\n"
             f"【待核查答案】{bad}")
        a = f"该答案存在问题：{issue}。\n更正后的答案：{base['answer']}"
        return self.sample("corrective", q, base["refs"], a, base["support"], allowed=base["allowed"])

    def _corrupt(self, s: dict) -> tuple[str | None, str]:
        ans = s["answer"]
        kind = self.rng.choice(["number", "cite", "extra"])
        ctx = " ".join(r["content"] for r in s["refs"]) + s["question"] + " ".join(s["allowed"])
        if kind == "number":
            nums = [m for m in re.finditer(r"(?<![\[\d.-])\d+(?![\]\d])", ans)]
            if nums:
                m = self.rng.choice(nums)
                for _ in range(20):
                    fake = str(int(m.group()) + self.rng.choice([1, 2, 5, 10, -1, -2]))
                    if fake not in re.findall(r"\d+", ctx) and int(fake) > 0:
                        bad = ans[:m.start()] + fake + ans[m.end():]
                        return bad, f"数字「{fake}」在参考资料中没有依据，资料中对应的是「{m.group()}」"
        if kind == "cite" and len(s["refs"]) > len(s["support"]):
            cites = list(re.finditer(r"\[(\d+)\]", ans))
            wrong = [i for i in range(1, len(s["refs"]) + 1) if i not in s["support"]]
            if cites and wrong:
                m, w = self.rng.choice(cites), self.rng.choice(wrong)
                bad = ans[:m.start()] + f"[{w}]" + ans[m.end():]
                return bad, f"引用 [{w}] 错误，[{w}] 中没有这条结论，应引用 [{m.group(1)}]"
        extra = self.rng.choice(["该用户已被公安机关立案调查", "该账号此前已被封禁 3 次",
                                 "同类团伙通常会在 48 小时内再次作案", "平台将对该用户罚款 500 元"])
        cite = self.rng.randint(1, len(s["refs"]))
        return ans + f"另外，{extra}[{cite}]。", f"「{extra}」在参考资料中没有任何依据，属于编造，应删除"

    # ============================================================== 汇总
    def build(self) -> tuple[list[dict], list[dict]]:
        gen = {"faithful": self.faithful, "multi": self.multi, "refuse": self.refuse, "corrective": self.corrective}
        train, evals = [], []
        for typ, n in COUNTS.items():
            rows = [gen[typ]() for _ in range(n)]
            k = EVAL_COUNTS.get(typ, 0)
            evals += rows[:k]
            train += rows[k:]
        # 去泄漏：留出集的问题（同一组参考资料）不得出现在训练集里，含 corrective 中嵌入的原问题
        held: dict[tuple, set[str]] = {}
        for s in evals:
            held.setdefault(tuple(r["doc_code"] for r in s["refs"]), set()).add(s["question"])

        def leaked(s: dict) -> bool:
            qs = held.get(tuple(r["doc_code"] for r in s["refs"]), ())
            return any(q == s["question"] or q in s["question"] for q in qs)

        train = [s for s in train if not leaked(s)]
        self.rng.shuffle(train)
        return train, evals
