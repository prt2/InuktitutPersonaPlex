# InuktitutPersonaPlex

## Start the web app

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

4. In a second terminal, activate the same environment and start the UI:

```bash
source .venv/bin/activate
python3 -m streamlit run app.py
```

5. Open:

```text
http://localhost:8501
```
