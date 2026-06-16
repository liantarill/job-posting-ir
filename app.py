"""
Flask application entry point for the Job Search IR System.
"""
from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from utils.retrieval import search_bm25, search_tfidf, _get_resources

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


# ---------------------------------------------------------------------------
# Dashboard stats (computed once at startup)
# ---------------------------------------------------------------------------
def _compute_stats() -> dict:
    """Read basic statistics about the dataset for the dashboard."""
    try:
        _, _, _, df = _get_resources()
        total_jobs = len(df)
        categories = df["Category"].dropna().nunique() if "Category" in df.columns else 0
        top_categories = (
            df["Category"]
            .dropna()
            .value_counts()
            .head(5)
            .to_dict()
            if "Category" in df.columns
            else {}
        )
        return {
            "total_jobs": total_jobs,
            "total_categories": categories,
            "top_categories": top_categories,
        }
    except Exception as exc:  # pylint: disable=broad-except
        app.logger.warning("Could not compute stats: %s", exc)
        return {"total_jobs": 0, "total_categories": 0, "top_categories": {}}


# Pre-load models at startup so first request is fast
with app.app_context():
    try:
        STATS = _compute_stats()
        app.logger.info("Models loaded. Dataset: %d jobs.", STATS["total_jobs"])
    except Exception as exc:  # pylint: disable=broad-except
        app.logger.error("Failed to pre-load models: %s", exc)
        STATS = {"total_jobs": 0, "total_categories": 0, "top_categories": {}}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Render the main search page."""
    return render_template("index.html", stats=STATS)


@app.route("/search", methods=["POST"])
def search():
    """
    Handle search requests.

    Expected JSON body:
        { "query": "<string>", "model": "tfidf" | "bm25", "top_k": <int> }

    Returns JSON:
        { "results": [...], "query": "...", "model": "...", "count": <int> }
    """
    data = request.get_json(silent=True) or {}

    query: str = (data.get("query") or "").strip()
    model: str = (data.get("model") or "tfidf").lower()
    top_k: int = int(data.get("top_k", 10))

    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    if model == "bm25":
        results = search_bm25(query, top_k=top_k)
    else:
        results = search_tfidf(query, top_k=top_k)

    return jsonify(
        {
            "results": results,
            "query": query,
            "model": model.upper(),
            "count": len(results),
        }
    )


@app.route("/stats")
def stats():
    """Return dataset statistics as JSON."""
    return jsonify(STATS)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
