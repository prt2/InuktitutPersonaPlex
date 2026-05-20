# InuktitutPersonaPlex

The Streamlit UI supports:

- model selection from a dropdown
- multiple chats
- chat history in the sidebar
- starting a new chat
- persistent chat history on disk
- a backend-agnostic request/response contract so another team can plug in the real model layer later

The current UI is wired for `backend_server.py` and its endpoints:

- `GET /health`
- `GET /models`
- `POST /generate`
- `POST /generate_rag`

The shell text is configurable through environment variables:

- `INUKTITUT_APP_TITLE`
- `INUKTITUT_APP_CAPTION`
- `INUKTITUT_ASSISTANT_GREETING`
- `INUKTITUT_MODELS`
- `INUKTITUT_MODELS_URL`
- `INUKTITUT_BACKEND_URL`
- `INUKTITUT_CHAT_STORE`

## Start the app

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

3. Start the backend:

```bash
python3 backend_server.py
```

4. In a second terminal, start the UI:

```bash
source .venv/bin/activate
python3 -m streamlit run app.py
```

5. Open `http://localhost:8501`

Chat history is stored in `chat_history.json` at the project root by default. You can override that path with `INUKTITUT_CHAT_STORE`.

## UI integration contract

The UI is intentionally flexible, but it now directly supports the current backend in `backend_server.py`.

### Models endpoint

The app tries to fetch available models from `GET /models` by default. `backend_server.py` returns:

```json
{
  "available": ["base", "adapted"],
  "base": "Qwen/Qwen2.5-3B-Instruct",
  "adapted": "./inuktitut_lora_adapter",
  "rag": true
}
```

The UI converts that into model options automatically:

- `Base model`
- `Adapted model`
- `Base model + RAG` when `rag` is `true`
- `Adapted model + RAG` when `rag` is `true`

### Generate endpoints

For non-RAG models, the UI sends `POST /generate` with this shape:

```json
{
  "question": "What is Inuit Nunangat?",
  "context": "geography",
  "model_type": "adapted"
}
```

For RAG models, the UI sends `POST /generate_rag` with:

```json
{
  "question": "What is Inuit Nunangat?",
  "context": "geography",
  "model_type": "adapted",
  "k": 3
}
```

The UI accepts these backend answer fields:

- `response`
- `answer`
- `message`

Optional response fields:

- `model_label`
- `sources`
- `retrieved`

Example response:

```json
{
  "response": "Inuit Nunangat refers to the Inuit homeland in Canada.",
  "model_label": "Our model",
  "sources": [
    {
      "instruction": "What is Inuit Nunangat?",
      "context": "geography",
      "response": "Inuit Nunangat refers to the Inuit homeland in Canada."
    }
  ]
}
```
