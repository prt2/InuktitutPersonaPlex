# InuktitutPersonaPlex

A Q&A prototype for Inuktitut language, culture, history, food, and communities — built for the RBC Borealis Let's Solve It Spring 2026 challenge.

The Streamlit UI supports:

- model selection from a dropdown
- multiple chats with persistent history
- topic/context filtering per query
- retrieved-context display for RAG responses
- a backend-agnostic contract so the model layer can be swapped independently

---

## Backends

There are two backends. Use the one that matches your environment.

### `server.py` — lightweight, no GPU required (use this for demos and local development)

Runs entirely on CPU. Serves answers from `sample_qa.jsonl` using keyword matching (Original baseline) and TF-IDF retrieval via LangChain (Our LangChain RAG). No model download needed.

### `backend_server.py` — full ML backend, requires CUDA GPU

Loads `Qwen/Qwen2.5-3B-Instruct` base model and a LoRA adapter (`./inuktitut_lora_adapter`). Requires `torch`, `transformers`, `peft`, `bitsandbytes`, and a CUDA-capable GPU with ≥8 GB VRAM. Intended for Colab or a remote GPU host.

---

## Quick start (Mac / CPU — demo setup)

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the backend (Terminal 1):

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

4. Start the UI (Terminal 2):

```bash
streamlit run app.py
```

5. Open `http://localhost:8501`

---

## Full ML backend (GPU host)

```bash
pip install fastapi uvicorn transformers torch peft bitsandbytes
pip install -r requirements-rag.txt   # optional — enables /generate_rag
python backend_server.py              # listens on 0.0.0.0:8000
```

The adapter must be at `./inuktitut_lora_adapter` next to the script. Symlink or copy from `2nd/inuktitut_lora_adapter` (preferred) or `1st_iteration/inuktitut_lora_adapter`.

---

## API

Both backends expose the same endpoints the UI calls.

### `GET /health`

```json
{ "status": "ok" }
```

### `GET /models`

`server.py` returns:

```json
{
  "models": [
    { "id": "original",     "label": "Original baseline" },
    { "id": "langchain-rag","label": "Our LangChain RAG" }
  ]
}
```

`backend_server.py` returns:

```json
{
  "available": ["base", "adapted"],
  "base": "Qwen/Qwen2.5-3B-Instruct",
  "adapted": "./inuktitut_lora_adapter",
  "rag": true
}
```

The UI handles both formats automatically.

### `POST /generate`

Request:

```json
{
  "question": "What is Inuit Nunangat?",
  "context": "geography",
  "model_type": "langchain-rag"
}
```

Response:

```json
{
  "response": "Inuit Nunangat refers to the Inuit homeland in Canada...",
  "model_label": "Our LangChain RAG",
  "sources": [
    {
      "instruction": "What is Inuit Nunangat?",
      "context": "geography",
      "response": "Inuit Nunangat refers to the Inuit homeland in Canada..."
    }
  ]
}
```

### `POST /generate_rag` (`backend_server.py` only)

Request:

```json
{
  "question": "What is Inuit Nunangat?",
  "context": "geography",
  "model_type": "adapted",
  "k": 3
}
```

---

## Topics

The `context` field accepts: `general`, `geography`, `food`, `daily life`, `culture`, `language`, `history`, `identity`, `professions`.

---

## Chat history

Stored in `chat_history.json` at the project root by default. Override with the `INUKTITUT_CHAT_STORE` environment variable.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `INUKTITUT_BACKEND_URL` | `http://localhost:8000` | Backend base URL |
| `INUKTITUT_MODELS_URL` | `{backend}/models` | Models endpoint override |
| `INUKTITUT_APP_TITLE` | `Chat Interface` | Browser tab title |
| `INUKTITUT_APP_CAPTION` | — | Subtitle shown in the UI |
| `INUKTITUT_ASSISTANT_GREETING` | `Start a conversation.` | First assistant message |
| `INUKTITUT_MODELS` | — | JSON array to hard-code model list |
| `INUKTITUT_CHAT_STORE` | `./chat_history.json` | Path to chat history file |
