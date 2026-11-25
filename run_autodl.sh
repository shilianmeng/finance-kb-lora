#!/usr/bin/env bash
# AutoDL / 任意 Linux 云 GPU 一键：装依赖 → 下载基座 → 造数据 → 微调前评估 → LoRA 训练 → 微调后评估 → 对比
#   bash run_autodl.sh                       # 默认 Qwen2.5-1.5B-Instruct
#   MODEL_ID=Qwen/Qwen2.5-3B-Instruct bash run_autodl.sh
set -euo pipefail
cd "$(dirname "$0")"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/$(basename "$MODEL_ID")}"
LORA_OUT="${LORA_OUT:-outputs/qwen-risk-lora}"

# AutoDL 学术加速（没有则忽略）
[ -f /etc/network_turbo ] && source /etc/network_turbo || true

pip install -r requirements-train.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

if [ ! -f "$MODEL_DIR/config.json" ]; then
  python -c "from modelscope import snapshot_download; snapshot_download('$MODEL_ID', local_dir='$MODEL_DIR')"
fi

python scripts/build_dataset.py
python scripts/evaluate.py --backend hf --model "$MODEL_DIR" --tag before
python scripts/train_lora.py --model "$MODEL_DIR" --out "$LORA_OUT"
python scripts/evaluate.py --backend hf --model "$MODEL_DIR" --lora "$LORA_OUT" --tag after
python scripts/compare.py reports/before.json reports/after.json

echo "完成：适配器在 $LORA_OUT，对比表在 reports/对比.md。部署见 serve_vllm.sh"
