# 金融知识库微调（Qwen2.5 + LoRA）

电商风控系统 RAG 审核助手的生成引擎。目标：只依据检索片段作答、每个结论带 [n] 引用、资料不足时拒答。

## 目录

| 路径 | 说明 |
| --- | --- |
| `finetune/common.py` | 系统提示词与参考资料拼接格式，**与风控后端 `app/services/rag.py` 完全一致**（测试会校验） |
| `finetune/datagen.py` | 指令数据构造：faithful 忠实作答 / multi 多片段整合 / refuse 拒答 / corrective 纠错反例 |
| `finetune/metrics.py` | 规则判定的三项指标：不编造率、引用正确率、正确拒答率（另给引用召回、误拒率） |
| `scripts/sync_corpus.py` | 从风控系统同步知识库与 12 条规则 → `data/corpus/kb.json` |
| `scripts/build_dataset.py` | 生成 `data/sft/train.jsonl`（991 条）、`eval.jsonl`（100 条留出集）、`train_chat.jsonl`（messages 格式） |
| `scripts/train_lora.py` | transformers + peft SFT，只对 assistant 部分算 loss |
| `scripts/evaluate.py` | 留出集评估，后端 `hf` / `openai`(vLLM) / `gold` |
| `scripts/compare.py` | 微调前后对比 → `reports/对比.md` |
| `run_autodl.sh` | 云 GPU 一键全流程 |
| `serve_vllm.sh` | vLLM 部署（基座 + LoRA 热挂载，模型名 `qwen-risk-lora`） |

## 数据

| 类型 | 训练 | 留出 | 内容 |
| --- | --- | --- | --- |
| faithful | 266 | 34 | 规则含义 / 阈值 / 处置政策 / 判例，单片段作答带引用 |
| multi | 266 | 33 | 同事件多规则分值累加与灰名单加分、处置流程、退款举证、IP 聚集定性 |
| refuse | 259 | 33 | 超出范围、资料中没有的属性、资料互相矛盾 → “根据现有资料无法确定” |
| corrective | 200 | – | 给出带错误的草稿（改数字 / 错引用 / 编造句），模型指出问题并给更正答案 |

留出集与训练集按“问题 + 参考资料组合”去重（含 corrective 中嵌入的原问题）。标准答案自检三项指标均 100%。

## 本地（CPU）

```bash
python scripts/sync_corpus.py          # 风控规则有改动时再跑
python scripts/build_dataset.py
python scripts/evaluate.py --backend gold   # 验证评估流程，应满分
python -m unittest discover -s tests -t .
```

## 云 GPU（AutoDL，24G 显存即可）

```bash
bash run_autodl.sh                                   # 默认 Qwen2.5-1.5B-Instruct
MODEL_ID=Qwen/Qwen2.5-3B-Instruct bash run_autodl.sh # 换 3B
```

流程：装依赖 → ModelScope 下载基座 → 造数据 → 微调前评估 → LoRA 训练（r=16, alpha=32, lr 2e-4, 3 epoch）→ 微调后评估 → 对比表。
1.5B 在 4090 上训练约 10 分钟。

## 接入风控系统

1. 云端 `bash serve_vllm.sh`（监听 127.0.0.1:8001）。
2. 本机开隧道：`ssh -CNg -L 8093:127.0.0.1:8001 root@<host> -p <port>`。
3. 风控后端 `app/core/config.py`：`RAG_LLM_URL = "http://127.0.0.1:8093/v1"`，`RAG_LLM_MODEL = "qwen-risk-lora"`。

### 本机 transformers 部署（不用云端，4GB 显存可跑 1.5B 半精度）

```powershell
python scripts/download_model.py          # 下载基座到 models/Qwen2.5-1.5B-Instruct（约 3GB）
python scripts/serve_hf.py                # 监听 127.0.0.1:8093，OpenAI 兼容接口
python scripts/serve_hf.py --lora outputs/qwen-risk-lora   # 训练完挂适配器
```

- 风控后端默认 `RAG_LLM_URL` 就是 8093，起了这个服务「生成审核建议」就走大模型。本机服务和云端隧道都用 8093，二选一。
- VS Code：风控项目里选「电商风控 + 本地模型」一键启动三个进程。
- 没挂 LoRA 时请求 `qwen-risk-lora` 会用基座回答，响应里 `model` 字段如实写 `qwen-base`——这时看到的是**微调前**的效果。
- 本机生成较慢（几十秒），风控后端 `RAG_TIMEOUT_SECONDS` 已调到 90，前端这个请求超时 120 秒。
- 也可用它跑基线：`python scripts/evaluate.py --backend openai --url http://127.0.0.1:8093/v1 --model qwen-base --tag before`。

审核助手是旁路服务：大模型超时或报错时自动降级为检索模式（模板答案 + 规则引用），不影响风控决策链路。
模型输出的 `[n]` 会被后端换成对应文档编号 `[R-001]` 等。

## 2026-11-22 评估修复

指标版本为 `9-bare-and-chinese-scores`：分值与加减方向绑定检查；“得分／评分／分值／分数／成绩”（含“风险分值／风险分数”）与“总分／风险分”统一按总分核验，支持 `得分（-30 分）`、`成绩为(+30分)` 等中英文括号写法；答案、标准答案和支撑资料使用同一识别规则。总分依据标准答案中的明确总分、逐项加减求和（限制 0–100）或样本 allowed 标注验证。相同分值的两条命中项分别计入；不把所有检索候选规则相加。答案、证据和标准答案求和统一识别 ASCII 正负号、`−－﹣‐‑‒–—`、`＋﹢`、中文“正／负”，允许符号与数字间有空格、制表符、不换行空格或全角空格。问题中的数字不自动成为事实依据；错误事实、空答案和拒答不能获得引用正确分。规则检查仍不能替代完整的语义忠实度人工复核。

没有关键词但以“分”为单位的数值也按总分核验（如 `命中 R-007、R-003，-60 分`）；标准答案或支撑资料写 `60 分` 时，正确的 `+60 分` 可通过。加减项整段匹配，不能重复当作总分；分钟、百分之和分钱不按风险分识别。支持常用中文数词（零／〇、一至九、两、十／百／千／万及“点”小数），如 `风险分为负六十分`、`加四十分，减十分`，并与阿拉伯数字统一比较；这不是任意中文数字、财务大写或完整自然语言的解析器。报告分值使用普通小数格式（如 `total:-60`），不显示科学计数法。

零分仍须有依据：标准答案或支撑资料明确总分为 0、逐项加减计算得到 0，或样本 `allowed` 明确标注 `"0"`，均可通过。不能把“标准答案没写加减项”自动当作 0；当前 100 条留出集中符合该描述的 51 条包含 33 条拒答，其余还含政策问答及实际为 25／60／80 分的判例，正常用户或处置放行也不代表风险分为 0。对确实为零但未记录分值的样本，须由标注确认后补充标准答案或 `allowed`，不能靠指标猜测。

`evaluate.py` 记录评估集 SHA-256 和指标版本。`compare.py` 拒绝 gold 标准答案自检、缺少真实模型名称、评估集指纹缺失或不一致的报告；两份报告的指标版本都必须等于当前代码版本，即使两份旧报告版本相同也拒绝，前后模型需按当前版本重新评估。`reports/对比.md` 是按当前指标版本在云 GPU 上真实训练、前后模型分别评估得到的结果（Qwen2.5-1.5B-Instruct + LoRA，留出集 100 条）：引用正确率 0%→91%、引用召回 0%→95.8%、正确拒答率 75.8%→100%、误拒率 56.7%→0%。

验证：`python -m unittest discover -s tests -t .`，30 条通过，包含总分不同说法、同分值多项、加减抵消、封顶、错误总分、候选资料不能证明总分、特殊符号及符号后空白、“得分／评分／分值／分数／成绩”的错误符号与数值、中英文括号及括号内加减求和、标准答案与支撑资料的同义分值识别、有依据的零分通过而未知分数不归零、无关键词分数的正负号与错误数值、中文分值及求和、分钟与百分比排除、报告数字格式、当前指标版本校验（含拒绝两份版本 8 报告）。Windows 默认 GBK 环境与 UTF-8 模式均可直接运行；测试显式约定子进程按 UTF-8 输出并按 UTF-8 解码，无需用户设置 `PYTHONUTF8`。`python scripts/evaluate.py --backend gold --tag gold` 仅为标准答案流程自检，不是模型效果。
