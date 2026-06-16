"""
Retrieval module — TF-IDF and BM25 search over the job-posting dataset.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from utils.preprocessing import preprocess_text, tokenize_query

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


# ---------------------------------------------------------------------------
# Lazy-loaded singletons (loaded once per process)
# ---------------------------------------------------------------------------
_tfidf_vectorizer = None
_tfidf_matrix = None
_bm25_model = None
_df_jobs: pd.DataFrame | None = None


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _get_resources():
    """Return all models / dataframe, loading from disk the first time."""
    global _tfidf_vectorizer, _tfidf_matrix, _bm25_model, _df_jobs

    if _df_jobs is None:
        _tfidf_vectorizer = _load_pickle(MODELS_DIR / "tfidf_vectorizer.pkl")
        _tfidf_matrix = _load_pickle(MODELS_DIR / "tfidf_matrix.pkl")
        _bm25_model = _load_pickle(MODELS_DIR / "bm25_model.pkl")
        _df_jobs = _load_pickle(MODELS_DIR / "df_jobs.pkl")

    return _tfidf_vectorizer, _tfidf_matrix, _bm25_model, _df_jobs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_str(val) -> str:
    """Return a clean string representation of a potentially NaN value."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return str(val).strip()


def _build_result(row: pd.Series, rank: int, score: float, model: str) -> dict:
    """Build a standardised result dictionary from a DataFrame row."""
    title = _safe_str(row.get("Job Opening Title", ""))
    category = _safe_str(row.get("Category", ""))
    keywords_raw = _safe_str(row.get("Keywords", ""))
    description = _safe_str(row.get("Description", ""))

    # Parse keywords — they are sometimes pipe-separated
    keywords: list[str] = []
    if keywords_raw:
        for sep in ("|", ",", ";"):
            if sep in keywords_raw:
                keywords = [k.strip() for k in keywords_raw.split(sep) if k.strip()]
                break
        if not keywords:
            keywords = [keywords_raw]

    # Truncate description for preview
    description_preview = (description[:300] + "…") if len(description) > 300 else description

    return {
        "rank": rank,
        "score": round(float(score), 4),
        "title": title or "Untitled Position",
        "category": category or "General",
        "keywords": keywords[:8],  # max 8 tags
        "description": description_preview,
        "model": model,
    }


# ---------------------------------------------------------------------------
# TF-IDF search
# ---------------------------------------------------------------------------

def search_tfidf(query: str, top_k: int = 10) -> list[dict]:
    """
    Search job postings using TF-IDF + cosine similarity.

    Args:
        query:  Raw user query.
        top_k:  Number of top results to return.

    Returns:
        List of result dicts (rank, score, title, category, keywords, description, model).
    """
    vectorizer, matrix, _, df = _get_resources()

    processed = preprocess_text(query)
    if not processed:
        return []

    query_vec = vectorizer.transform([processed])
    scores = cosine_similarity(query_vec, matrix).flatten()

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        score = scores[idx]
        if score <= 0:
            break
        results.append(_build_result(df.iloc[idx], rank, score, "TF-IDF"))

    return results


# ---------------------------------------------------------------------------
# BM25 search
# ---------------------------------------------------------------------------

def search_bm25(query: str, top_k: int = 10) -> list[dict]:
    """
    Search job postings using BM25 ranking.

    Args:
        query:  Raw user query.
        top_k:  Number of top results to return.

    Returns:
        List of result dicts (rank, score, title, category, keywords, description, model).
    """
    _, _, bm25, df = _get_resources()

    tokens = tokenize_query(query)
    if not tokens:
        return []

    scores = bm25.get_scores(tokens)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        score = scores[idx]
        if score <= 0:
            break
        results.append(_build_result(df.iloc[idx], rank, score, "BM25"))

    return results
