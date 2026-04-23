import json
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


APP_ROOT = Path(__file__).resolve().parent
DATA_PATH = APP_ROOT / "sample_qa.jsonl"
TOKEN_PATTERN = re.compile(r"[a-zA-Z']+")


class GenerateRequest(BaseModel):
    question: str
    model: str | None = None


class SampleEntry(BaseModel):
    instruction: str
    context: str
    response: str


def load_sample_entries() -> list[SampleEntry]:
    entries: list[SampleEntry] = []
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Sample QA file not found: {DATA_PATH}")

    for line in DATA_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        entries.append(SampleEntry(**json.loads(stripped)))
    return entries


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(text)}


def score_entry(question_tokens: set[str], entry: SampleEntry) -> int:
    instruction_tokens = tokenize(entry.instruction)
    context_tokens = tokenize(entry.context)
    overlap = len(question_tokens & instruction_tokens)
    overlap += len(question_tokens & context_tokens)
    return overlap


def choose_response(question: str, entries: list[SampleEntry]) -> str:
    question_tokens = tokenize(question)
    if not question_tokens:
        return "Please ask a non-empty question about Inuktitut communities, language, culture, food, or history."

    ranked = sorted(
        entries,
        key=lambda entry: score_entry(question_tokens, entry),
        reverse=True,
    )
    best = ranked[0]
    if score_entry(question_tokens, best) == 0:
        return (
            "I only have a small sample knowledge base right now. Try asking about geography, food, "
            "daily life, culture, language, history, identity, or professions."
        )
    return best.response


sample_entries = load_sample_entries()
app = FastAPI(title="InuktitutPersonaPlex Sample Backend")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate")
def generate(request: GenerateRequest) -> dict[str, Any]:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    response = choose_response(question, sample_entries)
    return {
        "response": response,
        "model": request.model or "sample-backend",
        "source": "sample_qa.jsonl",
    }
