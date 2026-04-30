# InuktitutPersonaPlex

The Streamlit UI supports:

- model selection from a dropdown
- multiple chats
- chat history in the sidebar
- starting a new chat
- persistent chat history on disk
- a backend-agnostic request/response contract so another team can plug in the real model layer later

The shell text is configurable through environment variables:

- `INUKTITUT_APP_TITLE`
- `INUKTITUT_APP_CAPTION`
- `INUKTITUT_ASSISTANT_GREETING`
- `INUKTITUT_MODELS`
- `INUKTITUT_MODELS_URL`
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
python3 -m uvicorn server:app --reload
```

4. In a second terminal, start the UI:

```bash
source .venv/bin/activate
python3 -m streamlit run app.py
```

5. Open `http://localhost:8501`

Chat history is stored in `chat_history.json` at the project root by default. You can override that path with `INUKTITUT_CHAT_STORE`.

## UI integration contract

The UI is intentionally flexible so the backend can be replaced later without rewriting the frontend.

### Models endpoint

The app tries to fetch available models from `GET /models` by default. Expected response:

```json
{
  "models": [
    { "id": "original", "label": "Original model" },
    { "id": "ours", "label": "Our model" }
  ]
}
```

If that endpoint does not exist yet, the UI falls back to:

- `Original model`
- `Our model`

You can also override the fallback list with `INUKTITUT_MODELS` as a JSON array.

### Generate endpoint

The UI sends a `POST` request to the configured API URL with this shape:

```json
{
  "question": "What is Inuit Nunangat?",
  "model": "ours",
  "chat_id": "0d7d46bb-7a31-4f89-8f89-18fca1c7f4f6",
  "messages": [
    { "role": "assistant", "content": "Ask me something about Inuktitut language, Inuit communities, food, culture, or history." },
    { "role": "user", "content": "What is Inuit Nunangat?" }
  ]
}
```

The UI accepts any of these answer fields in the response:

- `response`
- `answer`
- `message`

Optional response fields:

- `model_label`
- `sources`

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

## Current sample backend endpoints

- `GET /health`
- `GET /models`
- `POST /generate`
