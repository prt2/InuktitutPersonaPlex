import json
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from langchain_core.documents import Document
from langchain_community.retrievers import TFIDFRetriever
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


class ChatSource(BaseModel):
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


def choose_response(question: str, entries: list[SampleEntry]) -> tuple[str, list[ChatSource]]:
    question_tokens = tokenize(question)
    if not question_tokens:
        return (
            "Please ask a non-empty question about Inuktitut communities, language, culture, food, or history.",
            [],
        )

    ranked = sorted(
        entries,
        key=lambda entry: score_entry(question_tokens, entry),
        reverse=True,
    )
    best = ranked[0]
    if score_entry(question_tokens, best) == 0:
        return (
            "I only have a small sample knowledge base right now. Try asking about geography, food, "
            "daily life, culture, language, history, identity, or professions.",
            [],
        )
    return best.response, [ChatSource(**best.model_dump())]


def build_langchain_documents(entries: list[SampleEntry]) -> list[Document]:
    documents: list[Document] = []
    for entry in entries:
        page_content = "\n".join(
            [
                f"Question: {entry.instruction}",
                f"Topic: {entry.context}",
                f"Answer: {entry.response}",
            ]
        )
        documents.append(
            Document(
                page_content=page_content,
                metadata=entry.model_dump(),
            )
        )
    return documents


def choose_rag_response(question: str, retriever: TFIDFRetriever) -> tuple[str, list[ChatSource]]:
    if not tokenize(question):
        return (
            "Please ask a non-empty question about Inuktitut communities, language, culture, food, or history.",
            [],
        )

    retrieved_docs = retriever.invoke(question)
    if not retrieved_docs:
        return (
            "I could not find a relevant passage in the current knowledge base. Try asking about geography, food, "
            "daily life, culture, language, history, identity, or professions.",
            [],
        )

    sources = [
        ChatSource(
            instruction=str(doc.metadata.get("instruction", "")),
            context=str(doc.metadata.get("context", "")),
            response=str(doc.metadata.get("response", "")),
        )
        for doc in retrieved_docs
        if doc.metadata
    ]
    sources = [source for source in sources if source.response]
    if not sources:
        return (
            "I found related material, but it was incomplete. Try rephrasing your question.",
            [],
        )

    primary = sources[0]
    related_topics = [source.context for source in sources[1:] if source.context and source.context != primary.context]
    seen_topics: list[str] = []
    for topic in related_topics:
        if topic not in seen_topics:
            seen_topics.append(topic)

    answer = primary.response
    if seen_topics:
        answer += "\n\nRetrieved related topics: " + ", ".join(seen_topics[:2]) + "."
    answer += "\n\nThis answer came from the LangChain retrieval path over the local knowledge base."
    return answer, sources[:3]


def normalize_model_name(model: str | None) -> str:
    normalized = (model or "").strip().lower()
    if normalized in {"langchain-rag", "rag", "langchain rag", "ours", "our model"}:
        return "langchain-rag"
    return "original"


sample_entries = load_sample_entries()
retriever = TFIDFRetriever.from_documents(build_langchain_documents(sample_entries), k=3)
app = FastAPI(title="InuktitutPersonaPlex Sample Backend")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def models() -> dict[str, list[dict[str, str]]]:
    return {
        "models": [
            {"id": "original", "label": "Original baseline"},
            {"id": "langchain-rag", "label": "Our LangChain RAG"},
        ]
    }


@app.post("/generate")
def generate(request: GenerateRequest) -> dict[str, Any]:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    selected_model = normalize_model_name(request.model)
    if selected_model == "langchain-rag":
        response, sources = choose_rag_response(question, retriever)
        model_label = "Our LangChain RAG"
    else:
        response, sources = choose_response(question, sample_entries)
        model_label = "Original baseline"

    return {
        "response": response,
        "model": selected_model,
        "model_label": model_label,
        "source": "sample_qa.jsonl",
        "sources": [source.model_dump() for source in sources],
    }
