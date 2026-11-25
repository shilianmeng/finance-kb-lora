"""从 ModelScope 下载基座模型到 models/ 目录（国内源，不走 HuggingFace）。

    python scripts/download_model.py                              # Qwen2.5-1.5B-Instruct，约 3GB
    python scripts/download_model.py --id Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = ap.parse_args()
    from modelscope import snapshot_download

    target = ROOT / "models" / args.id.split("/")[-1]
    path = snapshot_download(args.id, local_dir=str(target))
    print(f"已下载到 {path}")


if __name__ == "__main__":
    main()
