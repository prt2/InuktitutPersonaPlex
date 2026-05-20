"""
backend_server.py — Inuktitut Q&A Backend
==========================================
Run this to expose a local REST API that Streamlit/Gradio UI can call.

Usage:
    pip install fastapi uvicorn transformers torch peft bitsandbytes
    pip install -r requirements-rag.txt          # optional, enables /generate_rag
    python backend_server.py

Then point the UI at http://localhost:8000

Endpoints:
    POST /generate      — generate from base or adapted model
    POST /generate_rag  — retrieve top-k chunks from dataset/DataSet, then generate
    GET  /health        — health check (also reports rag enabled/disabled)
    GET  /models        — list available models
"""

import torch
from typing import List, Literal, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# ── Config ────────────────────────────────────────────────────────────────────

BASE_MODEL_NAME  = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH     = "./inuktitut_lora_adapter"   # folder saved from notebook Step 11
LOAD_IN_4BIT     = True                          # set False if you have ≥16 GB VRAM
MAX_NEW_TOKENS   = 300
TEMPERATURE      = 0.1

SYSTEM_PROMPT = (
    "You are a helpful assistant specialized in Inuktitut culture and communities."
)

RAG_TOP_K_DEFAULT = 3

# ── FastAPI App ────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Inuktitut Q&A Backend",
    description = "Serves base and LoRA-adapted Qwen2.5-3B-Instruct responses",
    version     = "1.0.0",
)

# Global model references — loaded once at startup
_base_model      = None
_adapted_model   = None
_tokenizer       = None
_rag_store       = None    # FAISS index (None if RAG disabled)
_rag_retrieve    = None    # rag_eval.retrieve
_rag_build_q     = None    # rag_eval.build_rag_question


# ── Request / Response Schemas ─────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    question  : str
    context   : str = "general"
    model_type: Literal["base", "adapted"] = "adapted"

class GenerateResponse(BaseModel):
    question   : str
    context    : str
    model_type : str
    answer     : str

class RagGenerateRequest(BaseModel):
    question  : str
    context   : str = "general"
    model_type: Literal["base", "adapted"] = "adapted"
    k         : int = RAG_TOP_K_DEFAULT

class RagGenerateResponse(BaseModel):
    question   : str
    context    : str
    model_type : str
    answer     : str
    retrieved  : List[str]


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_prompt(instruction: str, context: str) -> str:
    """Format a question into the Qwen2.5 chat template (inference mode)."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}\n\nContext: {context}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def load_models():
    """Load base model and LoRA-adapted model into memory."""
    global _base_model, _adapted_model, _tokenizer

    print("Loading tokenizer and base model...")
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(load_in_4bit=LOAD_IN_4BIT) if LOAD_IN_4BIT else None

    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    _base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config = quantization_config,
        device_map          = "auto",
        torch_dtype         = torch.float16,
    )
    _base_model.eval()
    print("✓ Base model loaded.")

    print("Loading LoRA adapter on top of base model...")
    from peft import PeftModel
    _adapted_model = PeftModel.from_pretrained(_base_model, ADAPTER_PATH)
    _adapted_model.eval()
    print("✓ Adapted model loaded.")


def load_rag():
    """Best-effort RAG enable. Silently disabled if langchain isn't installed
    or neither the cached index nor the corpus directory is present."""
    global _rag_store, _rag_retrieve, _rag_build_q
    try:
        from rag_eval import (
            build_or_load_index, retrieve, build_rag_question,
            CORPUS_DIR, INDEX_DIR,
        )
    except ImportError as e:
        print(f"RAG disabled (langchain deps not installed): {e}")
        return
    if not (INDEX_DIR.exists() or CORPUS_DIR.exists()):
        print(f"RAG disabled (no index at {INDEX_DIR}, no corpus at {CORPUS_DIR})")
        return
    _rag_store    = build_or_load_index()
    _rag_retrieve = retrieve
    _rag_build_q  = build_rag_question
    print("✓ RAG store loaded.")


def generate(model, prompt: str) -> str:
    """Run inference and return the generated text."""
    inputs = _tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens = MAX_NEW_TOKENS,
            temperature    = TEMPERATURE,
            do_sample      = True,
            pad_token_id   = _tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return _tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── Startup ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    load_models()
    load_rag()


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "gpu":    torch.cuda.is_available(),
        "rag":    _rag_store is not None,
    }


@app.get("/models")
def list_models():
    return {
        "available": ["base", "adapted"],
        "base"     : BASE_MODEL_NAME,
        "adapted"  : ADAPTER_PATH,
        "rag"      : _rag_store is not None,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(req: GenerateRequest):
    if _tokenizer is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    prompt = build_prompt(req.question, req.context)

    if req.model_type == "base":
        answer = generate(_base_model, prompt)
    elif req.model_type == "adapted":
        answer = generate(_adapted_model, prompt)
    else:
        raise HTTPException(status_code=400, detail="model_type must be 'base' or 'adapted'")

    return GenerateResponse(
        question   = req.question,
        context    = req.context,
        model_type = req.model_type,
        answer     = answer,
    )


@app.post("/generate_rag", response_model=RagGenerateResponse)
def generate_rag_endpoint(req: RagGenerateRequest):
    if _tokenizer is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")
    if _rag_store is None:
        raise HTTPException(
            status_code=503,
            detail=("RAG not enabled. Install requirements-rag.txt and ensure "
                    "dataset/DataSet/ or .rag_index/ is present next to backend_server.py."),
        )

    chunks       = _rag_retrieve(_rag_store, req.question, req.context, k=req.k)
    rag_question = _rag_build_q(req.question, chunks)
    prompt       = build_prompt(rag_question, req.context)

    model = _adapted_model if req.model_type == "adapted" else _base_model
    answer = generate(model, prompt)

    return RagGenerateResponse(
        question   = req.question,
        context    = req.context,
        model_type = req.model_type,
        answer     = answer,
        retrieved  = chunks,
    )


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
