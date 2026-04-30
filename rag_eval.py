"""
rag_eval.py — RAG evaluation against the FastAPI backend
=========================================================
Builds a FAISS index over `dataset/DataSet/**/*.txt` (the source documents the
Q&A pairs were derived from) using sentence-transformers embeddings, then for
each test question retrieves top-k chunks and asks the backend to answer with
and without retrieval. Produces a 4-way comparison (base / adapted / base+RAG /
adapted+RAG) scored with the same keyword-overlap metric as the notebook.

Usage:
    pip install -r requirements-rag.txt

    # In a separate process (needs GPU): python backend_server.py

    # Then run the eval (CPU-only, no GPU needed for retrieval):
    python rag_eval.py                                          # full 48-Q eval
    python rag_eval.py --single "What does Iqalummiut mean?" --topic language
    python rag_eval.py --backend http://remote-gpu:8000 --k 4
"""

import argparse
import json
import textwrap
from pathlib import Path
from typing import List, Optional

import requests
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Config ────────────────────────────────────────────────────────────────────

REPO_ROOT     = Path(__file__).resolve().parent
CORPUS_DIR    = REPO_ROOT / "dataset" / "DataSet"
TEST_PATH     = REPO_ROOT / "dataset" / "test_qa_pairs.jsonl"
INDEX_DIR     = REPO_ROOT / ".rag_index"
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 80
TOP_K         = 3
BACKEND_URL   = "http://localhost:8000"

# Folder name -> `context` value used in the JSONL Q&A pairs.
TOPIC_BY_DIR = {
    "01_geography_communities":     "location",
    "02_food_daily_life":           "food",
    "03_culture_traditions":        "culture",
    "04_language_basics":           "language",
    "05_history_identity":          "history",
    "06_professions_contributions": "professions",
}

# ── Index ─────────────────────────────────────────────────────────────────────

def build_or_load_index(rebuild: bool = False) -> FAISS:
    """Build a FAISS index over corpus .txt files (or load from disk if cached)."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if INDEX_DIR.exists() and not rebuild:
        print(f"Loading cached FAISS index from {INDEX_DIR}")
        return FAISS.load_local(
            str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )

    print(f"Building FAISS index from {CORPUS_DIR}...")
    docs = []
    for topic_dir, topic in TOPIC_BY_DIR.items():
        loader = DirectoryLoader(
            str(CORPUS_DIR / topic_dir),
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        for d in loader.load():
            d.metadata["topic"] = topic
            d.metadata["source_file"] = Path(d.metadata.get("source", "")).name
            docs.append(d)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  {len(docs)} files -> {len(chunks)} chunks")

    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(INDEX_DIR))
    return store


def retrieve(store: FAISS, question: str, topic: Optional[str], k: int) -> List[str]:
    """Top-k chunks; falls back to global retrieval if topic-scoped returns < k."""
    if topic and topic != "general":
        scoped = store.similarity_search(question, k=k, filter={"topic": topic})
        if len(scoped) >= k:
            return [d.page_content for d in scoped]
    return [d.page_content for d in store.similarity_search(question, k=k)]


# ── Backend client ────────────────────────────────────────────────────────────

def call_backend(question: str, context: str, model_type: str, backend: str) -> str:
    r = requests.post(
        f"{backend}/generate",
        json={"question": question, "context": context, "model_type": model_type},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["answer"]


def build_rag_question(question: str, snippets: List[str]) -> str:
    """Inject retrieved snippets into the user `instruction`. The trained
    `\\n\\nContext: {topic}` suffix on the user turn is preserved by the
    backend's prompt builder, so we only modify the question text itself."""
    refs = "\n\n".join(f"[{i+1}] {s.strip()}" for i, s in enumerate(snippets))
    return (
        "Use the following reference passages to answer the question. "
        "If the answer is not contained in the passages, answer from your own "
        "knowledge but stay concise.\n\n"
        f"Reference passages:\n{refs}\n\n"
        f"Question: {question}"
    )


# ── Scoring (matches notebook Step 11) ────────────────────────────────────────

STOPWORDS = {
    "the", "a", "an", "is", "in", "of", "and", "to", "it", "for", "or",
    "that", "this", "was", "are", "as", "by", "with", "its", "at", "be",
    "from", "has", "have", "on",
}

def keyword_score(model_answer: str, reference_answer: str) -> float:
    ref = {w.lower().strip(".,;:") for w in reference_answer.split()} - STOPWORDS
    out = {w.lower().strip(".,;:") for w in model_answer.split()}
    return len(ref & out) / len(ref) if ref else 0.0


# ── Full eval ─────────────────────────────────────────────────────────────────

VARIANTS = [
    # (label,        score_key,    answer_key,   model_type, use_rag)
    ("base",         "s_base",     "a_base",     "base",     False),
    ("adapted",      "s_adapted",  "a_adapted",  "adapted",  False),
    ("base+RAG",     "s_base_rag", "a_base_rag", "base",     True),
    ("adapted+RAG",  "s_adapt_rag","a_adapt_rag","adapted",  True),
]

def run_eval(backend: str, k: int, rebuild: bool, out_path: Path):
    store = build_or_load_index(rebuild=rebuild)

    with open(TEST_PATH, encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f if line.strip()]

    health = requests.get(f"{backend}/health", timeout=10).json()
    print(f"Backend: {backend}  {health}\n")

    rows = []
    for i, q in enumerate(test_data):
        snippets = retrieve(store, q["instruction"], q["context"], k=k)
        rag_q    = build_rag_question(q["instruction"], snippets)

        row = {
            "q":        q["instruction"],
            "topic":    q["context"],
            "ref":      q["response"],
            "snippets": snippets,
        }
        for label, s_key, a_key, mtype, use_rag in VARIANTS:
            user_q = rag_q if use_rag else q["instruction"]
            ans    = call_backend(user_q, q["context"], mtype, backend)
            row[a_key] = ans
            row[s_key] = keyword_score(ans, q["response"])
        rows.append(row)

        scores = "  ".join(f"{lbl}={row[s]:.0%}" for lbl, s, *_ in VARIANTS)
        print(f"[{i+1}/{len(test_data)}] {q['instruction'][:55]}\n  {scores}")

    write_report(rows, out_path)


def write_report(rows, out_path: Path):
    SEP, DSEP = "=" * 80, "-" * 80
    avgs = {s: sum(r[s] for r in rows) / len(rows) for _, s, *_ in VARIANTS}

    lines = [SEP, "INUKTITUT Q&A — 4-WAY COMPARISON (base / adapted / +RAG)", SEP, "",
             "OVERALL", DSEP]
    for label, s_key, *_ in VARIANTS:
        lines.append(f"  {label:<14} {avgs[s_key]:.2%}")
    lines += ["", "BY TOPIC", DSEP]
    for t in sorted({r["topic"] for r in rows}):
        tr = [r for r in rows if r["topic"] == t]
        bits = "  ".join(
            f"{lbl}={sum(r[s] for r in tr)/len(tr):.2%}"
            for lbl, s, *_ in VARIANTS
        )
        lines.append(f"  {t:<14} {bits}  (n={len(tr)})")
    lines += ["", SEP, ""]

    for i, r in enumerate(rows):
        lines += [
            f"Q{i+1} [{r['topic'].upper()}]: {r['q']}", DSEP,
            "REFERENCE:", textwrap.fill(r["ref"], 78), "",
            "RETRIEVED CHUNKS:",
        ]
        for j, s in enumerate(r["snippets"]):
            lines.append(textwrap.fill(
                f"  [{j+1}] {s}", 78, subsequent_indent="      "
            ))
        for label, s_key, a_key, *_ in VARIANTS:
            lines += ["", f"{label} ({r[s_key]:.2%}):",
                      textwrap.fill(r[a_key], 78)]
        lines += ["", SEP, ""]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_path}  ({len(rows)} questions)")
    print("Overall: " + "  ".join(
        f"{lbl}={avgs[s]:.2%}" for lbl, s, *_ in VARIANTS
    ))


# ── Single-question mode ──────────────────────────────────────────────────────

def run_single(question: str, topic: str, backend: str, k: int, rebuild: bool):
    store    = build_or_load_index(rebuild=rebuild)
    snippets = retrieve(store, question, topic, k=k)
    rag_q    = build_rag_question(question, snippets)

    print(f"\nQuestion: {question}\nTopic:    {topic}\n\nRetrieved:")
    for i, s in enumerate(snippets):
        print(f"  [{i+1}] {s[:160]}{'...' if len(s) > 160 else ''}")
    print()

    for label, _, _, mtype, use_rag in VARIANTS:
        user_q = rag_q if use_rag else question
        print(f"--- {label} ---\n{call_backend(user_q, topic, mtype, backend)}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default=BACKEND_URL)
    p.add_argument("--k", type=int, default=TOP_K)
    p.add_argument("--rebuild-index", action="store_true")
    p.add_argument("--single", help="Run one ad-hoc question instead of the eval")
    p.add_argument("--topic", default="general", help="Topic for --single mode")
    p.add_argument("--out", default="comparison_rag.txt")
    args = p.parse_args()

    if args.single:
        run_single(args.single, args.topic, args.backend, args.k, args.rebuild_index)
    else:
        run_eval(args.backend, args.k, args.rebuild_index, REPO_ROOT / args.out)


if __name__ == "__main__":
    main()
