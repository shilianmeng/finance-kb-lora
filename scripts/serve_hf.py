"""本机用 transformers 部署模型，提供 OpenAI 兼容接口（替代云端 vLLM，适合 4GB 显存的小卡）。

    python scripts/serve_hf.py                       # 只挂基座，默认 models/Qwen2.5-1.5B-Instruct，端口 8093
    python scripts/serve_hf.py --lora outputs/qwen-risk-lora   # 训练完后挂 LoRA

接口：
  GET  /v1/models
  POST /v1/chat/completions   {"model", "messages", "temperature", "max_tokens"}（不支持 stream）
  GET  /health

模型名：
  qwen-base       基座
  qwen-risk-lora  挂了 --lora 时用适配器；没挂时退回基座，响应里的 "model" 如实写 qwen-base，
                  这样风控后端 config.RAG_LLM_MODEL 不用改，也不会把基座的回答当成微调效果。

风控后端 config.RAG_LLM_URL 默认就是 http://127.0.0.1:8093/v1，起了本服务即可用。
8093 原本留给云端隧道；本机服务和隧道二选一，不能同时开。
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
import uuid
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-1.5B-Instruct"
BASE_NAME, LORA_NAME = "qwen-base", "qwen-risk-lora"

log = logging.getLogger("serve_hf")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = BASE_NAME
    messages: list[Message]
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    stream: bool = False


class Engine:
    def __init__(self, model_dir: str, lora: str, max_input_tokens: int):
        self.tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            device_map = "cuda"
        else:
            dtype, device_map = torch.float32, "cpu"
            log.warning("没有可用的 CUDA，退回 CPU，会很慢")
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype, device_map=device_map,
                                                     trust_remote_code=True)
        self.has_lora = bool(lora)
        if lora:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora)
        self.model = model.eval()
        self.max_input_tokens = max_input_tokens
        self.lock = threading.Lock()   # 单卡单请求，避免并发把显存打爆
        log.info("模型已加载：%s dtype=%s device=%s lora=%s", model_dir, dtype, self.model.device, lora or "-")

    def served_names(self) -> list[str]:
        return [BASE_NAME, LORA_NAME] if self.has_lora else [BASE_NAME]

    def generate(self, req: ChatRequest) -> tuple[str, str, int, int]:
        use_lora = req.model == LORA_NAME and self.has_lora
        actual = LORA_NAME if use_lora else BASE_NAME
        text = self.tok.apply_chat_template([m.model_dump() for m in req.messages], tokenize=False,
                                            add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False)
        n_in = enc["input_ids"].shape[1]
        if n_in > self.max_input_tokens:
            raise HTTPException(400, f"输入 {n_in} tokens，超过上限 {self.max_input_tokens}")
        enc = enc.to(self.model.device)
        kw = {"max_new_tokens": max(1, min(req.max_tokens, 2048)), "pad_token_id": self.tok.eos_token_id}
        if req.temperature > 0:
            kw.update(do_sample=True, temperature=req.temperature, top_p=req.top_p)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        with self.lock, torch.no_grad():
            if self.has_lora and not use_lora:
                with self.model.disable_adapter():
                    out = self.model.generate(**enc, **kw)
            else:
                out = self.model.generate(**enc, **kw)
        gen = out[0][n_in:]
        return self.tok.decode(gen, skip_special_tokens=True).strip(), actual, n_in, int(gen.shape[0])


def build_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="本地 transformers 推理服务")

    @app.get("/health")
    def health():
        return {"status": "ok", "models": engine.served_names(), "cuda": torch.cuda.is_available()}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": n, "object": "model", "owned_by": "local"}
                                           for n in engine.served_names()]}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        if req.stream:
            raise HTTPException(400, "不支持 stream")
        if req.model not in (BASE_NAME, LORA_NAME):
            raise HTTPException(404, f"未知模型 {req.model}，可用：{engine.served_names()}")
        if req.model == LORA_NAME and not engine.has_lora:
            log.warning("请求 %s 但未挂 LoRA，用基座回答", LORA_NAME)
        t0 = time.time()
        content, actual, n_in, n_out = engine.generate(req)
        log.info("%s 输入 %d 输出 %d tokens，用时 %.1fs", actual, n_in, n_out, time.time() - t0)
        return {"id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion", "created": int(time.time()),
                "model": actual,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                             "finish_reason": "length" if n_out >= req.max_tokens else "stop"}],
                "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out}}

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="基座目录")
    ap.add_argument("--lora", default="", help="LoRA 适配器目录（可选）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--max-input-tokens", type=int, default=3072)
    args = ap.parse_args()
    if args.port == 8000:
        ap.error("8000 被本机其他服务占用，不要用")
    if not Path(args.model).is_dir():
        ap.error(f"找不到模型目录 {args.model}，先运行 python scripts/download_model.py")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(build_app(Engine(args.model, args.lora, args.max_input_tokens)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
