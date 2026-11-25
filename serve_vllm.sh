#!/usr/bin/env bash
# 用 vLLM 起 OpenAI 兼容服务：基座 + LoRA 热挂载，模型名 qwen-risk-lora
#   bash serve_vllm.sh
# 风控后端 app/core/config.py 里改：
#   RAG_LLM_URL   = "http://127.0.0.1:8093/v1"   （本机经 SSH 隧道转发到云端 8001）
#   RAG_LLM_MODEL = "qwen-risk-lora"
# 隧道示例（在本机执行，端口按 AutoDL 控制台填）：
#   ssh -CNg -L 8093:127.0.0.1:8001 root@<host> -p <port>
set -euo pipefail
cd "$(dirname "$0")"

MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}"
LORA_OUT="${LORA_OUT:-outputs/qwen-risk-lora}"
PORT="${PORT:-8001}"

pip install "vllm>=0.6.3" -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" --served-model-name qwen-base \
  --enable-lora --lora-modules qwen-risk-lora="$LORA_OUT" --max-lora-rank 16 \
  --host 127.0.0.1 --port "$PORT" --max-model-len 4096 --gpu-memory-utilization 0.85
