# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Inuktitut Q&A fine-tuning project (RBC Borealis "Let's Solve It" Spring 2026). LoRA-adapts `Qwen/Qwen2.5-3B-Instruct` on a small Inuktitut culture/language Q&A dataset, evaluates base vs. adapted side-by-side, and serves both via a local FastAPI backend.

The pipeline has two halves that live in different runtimes — keep them aligned:
- **Training** (`Inuktitut_FineTuning.ipynb`) is built for Google Colab + T4 GPU using **Unsloth** on top of `unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit`.
- **Serving** (`backend_server.py`) runs locally using plain `transformers` + `peft` against the upstream `Qwen/Qwen2.5-3B-Instruct` and stacks the LoRA adapter on top.

The adapter `base_model_name_or_path` (Unsloth bnb-4bit) does **not** match what the backend loads (upstream Qwen). This is intentional — the LoRA targets are the same Qwen2 modules, but if you change quantization, dtype, or target_modules you must verify both paths still produce sane outputs.

## Repo layout

- `Inuktitut_FineTuning.ipynb` — end-to-end pipeline: load base, baseline-eval 48 test questions, apply LoRA, train 5 epochs on 194 samples, re-eval, score, save adapter, save comparison report. Designed to run top-to-bottom in Colab.
- `backend_server.py` — FastAPI service exposing `/generate`, `/generate_rag`, `/health`, `/models`. Loads both base and adapted models at startup. RAG support is best-effort: if `requirements-rag.txt` is installed and either `.rag_index/` or `dataset/DataSet/` is present, `load_rag()` enables the FAISS index and `/generate_rag`. Otherwise that endpoint returns 503 and the rest of the server runs normally.
- `rag_eval.py` — standalone LangChain RAG evaluator. Builds a FAISS index over `dataset/DataSet/**/*.txt` with `sentence-transformers/all-MiniLM-L6-v2` embeddings, retrieves top-k chunks per test question, then calls the running backend in 4 variants (base / adapted / base+RAG / adapted+RAG) and writes `comparison_rag.txt` using the same keyword-overlap metric as the notebook.
- `requirements-rag.txt` — deps for `rag_eval.py` only (langchain, faiss-cpu, sentence-transformers). Index runs on CPU; the script just hits the backend over HTTP, so a Mac can drive a remote-GPU backend.
- `1st_iteration/` and `2nd/` — saved artifacts from two training runs. Each contains an `inuktitut_lora_adapter/` directory and a `comparison_base_vs_adapted.txt` report. The two runs differ in LoRA hyperparameters:
  - 1st: `r=16`, `lora_alpha=16`, targets `q_proj`, `v_proj` only-ish (smaller adapter)
  - 2nd: `r=32`, `lora_alpha=32`, all 7 attention + MLP projections (current preferred config; matches the notebook's Step 7)
- `dataset/train_qa_pairs.jsonl` / `dataset/test_qa_pairs.jsonl` — 194/48 Q&A pairs. Records have `instruction`, `context` (topic: culture/food/history/language/location/professions), `response`. Notebook Step 2 still uses `google.colab.files.upload()`, so in Colab you upload these manually.
- `dataset/DataSet/` — source documents the Q&A pairs were derived from, organized into 6 topic folders (`01_geography_communities`, `02_food_daily_life`, `03_culture_traditions`, `04_language_basics`, `05_history_identity`, `06_professions_contributions`) plus `SOURCES.md`. The folder→topic mapping is hardcoded in `rag_eval.py` as `TOPIC_BY_DIR` and aligns 1:1 with the `context` values in the JSONL. Used as the RAG corpus.

## Commands

There is no build system, lint config, or test suite. Two main entry points:

```bash
# Serve both models locally (expects ./inuktitut_lora_adapter/ next to the script — symlink or copy from 2nd/ first)
pip install fastapi uvicorn transformers torch peft bitsandbytes
python backend_server.py   # listens on 0.0.0.0:8000

# Probe
curl http://localhost:8000/health
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"question":"What does Iqalummiut mean?","context":"language","model_type":"adapted"}'

# RAG: install on the backend host so /generate_rag works
pip install -r requirements-rag.txt
# (the backend's load_rag() will then build .rag_index/ from dataset/DataSet/ on first start)

curl -X POST http://localhost:8000/generate_rag \
  -H 'Content-Type: application/json' \
  -d '{"question":"What was the James Bay and Northern Quebec Agreement?","context":"history","model_type":"adapted","k":3}'

# RAG eval (CPU-only, runs anywhere, points at the backend over HTTP)
python rag_eval.py                                  # 4-way 48-question eval -> comparison_rag.txt
python rag_eval.py --single "What does Iqalummiut mean?" --topic language
python rag_eval.py --backend http://remote-gpu:8000 --k 4
```

The training notebook is meant to be executed in Colab — local execution requires CUDA and the Unsloth install (Step 1) which only supports Linux + NVIDIA. Don't try to run the notebook on this Mac. `rag_eval.py` is fine on Mac (CPU embeddings + HTTP), but it needs a reachable backend for the actual generations.

## Critical contracts

**Prompt format must match across train and serve.** Both `format_prompt` (notebook Step 4) and `build_prompt` (backend) emit:
```
<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{instruction}\n\nContext: {context}<|im_end|>\n<|im_start|>assistant\n
```
The literal `\n\nContext: {context}` suffix on the user turn is part of the trained format — removing it will degrade the adapter. `SYSTEM_PROMPT` is also identical in both files; keep them in sync.

**Adapter path.** `backend_server.py` hardcodes `ADAPTER_PATH = "./inuktitut_lora_adapter"`. To use the 2nd-iteration adapter, either copy/symlink `2nd/inuktitut_lora_adapter` into the repo root or edit the constant.

**Generation defaults.** `temperature=0.1`, `do_sample=True`, `max_new_tokens=300` (backend) / `256` (notebook). Low temperature is intentional — outputs are evaluated by keyword overlap against reference answers (notebook Step 11), so determinism matters more than diversity.

**RAG injection point.** `rag_eval.py` injects retrieved chunks into the *user instruction text* (the `instruction` field of the request body), not into the `Context:` slot. The `Context:` field stays as the topic word (`language`, `food`, etc.) it was during training, so the trained `\n\nContext: {topic}` suffix on the user turn is preserved verbatim. Don't move retrieval into the `context` field — that field was never trained on free text and the adapter will degrade.

## Evaluation

Scoring is keyword-set overlap (notebook Step 11): tokenize reference answer, drop a small stopword set, compute `|ref ∩ model| / |ref|`. Aggregated overall and per-topic. The 2nd-iteration report (`2nd/comparison_base_vs_adapted.txt`) shows base 42.65% → adapted 57.44%, with `professions` being the one topic that regressed slightly. If you re-train, replicate this scoring rather than swapping in BLEU/ROUGE — comparisons across runs assume the same metric.
