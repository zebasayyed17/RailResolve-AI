
from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INDEX_DIR = PROJECT_ROOT / "artifacts" / "vector_index"
FAISS_PATH = INDEX_DIR / "policy.faiss"
CHUNKS_PATH = INDEX_DIR / "chunks.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Reciprocal Rank Fusion parameter.
RRF_K = 60

# Retrieve extra candidates before applying diversity rules.
DEFAULT_CANDIDATE_MULTIPLIER = 5

# Near-duplicate thresholds. These are heuristic filters,
# not learned relevance probabilities.
DUPLICATE_CONTAINMENT_THRESHOLD = 0.82
DUPLICATE_JACCARD_THRESHOLD = 0.72


# ============================================================
# 2. LOAD THE EXISTING INDEX AND DOCUMENT CHUNKS
# ============================================================

@lru_cache(maxsize=1)
def load_chunks() -> tuple[dict[str, Any], ...]:
    """Load policy chunks once and validate their basic structure."""

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Policy chunks not found: {CHUNKS_PATH}\n"
            "Run the existing document-indexing pipeline first."
        )

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("chunks.json must contain a JSON list.")

    chunks = []

    for position, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(
                f"Chunk at position {position} is not a JSON object."
            )

        if not str(item.get("text", "")).strip():
            continue

        chunk = dict(item)

        # Supply a stable fallback ID if the original lacks one.
        chunk.setdefault("chunk_id", f"CHUNK_{position:05d}")
        chunk.setdefault("source", "Unknown source")
        chunk.setdefault("page", None)

        chunks.append(chunk)

    if not chunks:
        raise ValueError("No usable policy text was found in chunks.json.")

    return tuple(chunks)


@lru_cache(maxsize=1)
def load_faiss_index():
    """Load the existing FAISS index once."""

    if not FAISS_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {FAISS_PATH}\n"
            "Run the existing document-indexing pipeline first."
        )

    index = faiss.read_index(str(FAISS_PATH))
    chunks = load_chunks()

    if index.ntotal != len(chunks):
        raise ValueError(
            "The FAISS index and chunks.json contain different numbers "
            f"of records: index={index.ntotal}, chunks={len(chunks)}. "
            "Rebuild the index and chunks together before searching."
        )

    return index


@lru_cache(maxsize=1)
def load_embedding_model():
    """Load the embedding model once per Python process."""

    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def load_bm25():
    """Build the BM25 keyword index once per Python process."""

    chunks = load_chunks()

    tokenized_documents = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_documents)


# ============================================================
# 3. TEXT NORMALIZATION AND DUPLICATE DETECTION
# ============================================================

def tokenize(text: str) -> list[str]:
    """Normalize text for keyword search and overlap comparisons."""

    return re.findall(r"[a-z0-9]+", str(text).lower())


def normalized_text(text: str) -> str:
    """Return a normalized version for exact-duplicate detection."""

    return " ".join(tokenize(text))


def text_similarity(text_a: str, text_b: str) -> tuple[float, float]:
    """
    Compare two passages using token-set overlap.

    Returns:
        containment: shared tokens / size of the smaller token set
        jaccard: shared tokens / size of the union
    """

    tokens_a = set(tokenize(text_a))
    tokens_b = set(tokenize(text_b))

    if not tokens_a or not tokens_b:
        return 0.0, 0.0

    intersection = len(tokens_a & tokens_b)
    smaller_size = min(len(tokens_a), len(tokens_b))
    union_size = len(tokens_a | tokens_b)

    containment = intersection / smaller_size
    jaccard = intersection / union_size

    return containment, jaccard


def are_near_duplicates(
    candidate: dict[str, Any],
    selected: dict[str, Any],
) -> bool:
    """Identify exact or highly overlapping passages."""

    text_a = normalized_text(candidate.get("text", ""))
    text_b = normalized_text(selected.get("text", ""))

    if not text_a or not text_b:
        return False

    if text_a == text_b:
        return True

    containment, jaccard = text_similarity(text_a, text_b)

    # High containment catches a passage that mostly repeats
    # another, even when one passage is longer.
    if containment >= DUPLICATE_CONTAINMENT_THRESHOLD:
        return True

    if jaccard >= DUPLICATE_JACCARD_THRESHOLD:
        return True

    return False


# ============================================================
# 4. OPTIONAL SOURCE FILTER
# ============================================================

def normalize_source_filter(source_filter):
    """Accept a filename string, an iterable of filenames, or None."""

    if source_filter is None:
        return None

    if isinstance(source_filter, str):
        return {source_filter.casefold()}

    return {
        str(source).casefold()
        for source in source_filter
    }


# ============================================================
# 5. HYBRID RETRIEVAL
# ============================================================

def hybrid_search(
    query: str,
    top_k: int = 3,
    source_filter=None,
    candidate_k: int | None = None,
    max_per_source_page: int = 1,
) -> list[dict[str, Any]]:
    """
    Retrieve relevant railway policy passages using:

      1. Dense retrieval through the existing FAISS index.
      2. BM25 keyword retrieval.
      3. Reciprocal Rank Fusion.
      4. Duplicate suppression.
      5. Source/page diversification.

    Returns dictionaries containing the original chunk fields and
    dense_similarity, bm25_score, and hybrid_score.

    The source_filter argument accepts one filename or a collection
    of filenames.
    """

    if not isinstance(query, str) or not query.strip():
        return []

    if top_k < 1:
        return []

    if max_per_source_page < 1:
        raise ValueError("max_per_source_page must be at least 1.")

    query = query.strip()

    chunks = load_chunks()
    index = load_faiss_index()
    model = load_embedding_model()
    bm25 = load_bm25()

    if len(chunks) != index.ntotal:
        raise ValueError(
            "Policy index/chunk count mismatch. Rebuild the index."
        )

    # --------------------------------------------------------
    # A. Dense semantic retrieval
    # --------------------------------------------------------

    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    query_vector = np.asarray(query_vector, dtype=np.float32)

    dense_scores, dense_positions = index.search(
        query_vector,
        index.ntotal,
    )

    dense_order = [
        int(position)
        for position in dense_positions[0]
        if 0 <= int(position) < len(chunks)
    ]

    dense_rank = {
        chunk_position: rank
        for rank, chunk_position in enumerate(dense_order, start=1)
    }

    dense_score_by_position = {
        int(position): float(score)
        for position, score in zip(
            dense_positions[0],
            dense_scores[0],
        )
        if 0 <= int(position) < len(chunks)
    }

    # --------------------------------------------------------
    # B. BM25 keyword retrieval
    # --------------------------------------------------------

    query_tokens = tokenize(query)
    bm25_scores = np.asarray(
        bm25.get_scores(query_tokens),
        dtype=np.float64,
    )

    bm25_order = sorted(
        range(len(chunks)),
        key=lambda position: float(bm25_scores[position]),
        reverse=True,
    )

    bm25_rank = {
        chunk_position: rank
        for rank, chunk_position in enumerate(bm25_order, start=1)
    }

    # --------------------------------------------------------
    # C. Reciprocal Rank Fusion
    # --------------------------------------------------------

    allowed_sources = normalize_source_filter(source_filter)

    candidates = []

    for position, chunk in enumerate(chunks):
        if allowed_sources is not None:
            source_name = str(chunk.get("source", "")).casefold()

            if source_name not in allowed_sources:
                continue

        semantic_rank = dense_rank.get(position, len(chunks) + 1)
        keyword_rank = bm25_rank.get(position, len(chunks) + 1)

        hybrid_score = (
            1.0 / (RRF_K + semantic_rank)
            + 1.0 / (RRF_K + keyword_rank)
        )

        item = dict(chunk)

        raw_dense_score = dense_score_by_position.get(position, 0.0)

        # IndexFlatIP returns an inner-product score. For an L2 index,
        # this is a monotonic distance transform, not cosine similarity.
        if index.metric_type == faiss.METRIC_INNER_PRODUCT:
            dense_similarity = raw_dense_score
        else:
            dense_similarity = 1.0 / (1.0 + max(raw_dense_score, 0.0))

        item["dense_similarity"] = float(dense_similarity)
        item["bm25_score"] = float(bm25_scores[position])
        item["dense_rank"] = int(semantic_rank)
        item["bm25_rank"] = int(keyword_rank)
        item["hybrid_score"] = float(hybrid_score)

        candidates.append(item)

    candidates.sort(
        key=lambda item: item["hybrid_score"],
        reverse=True,
    )

    # Retrieve a larger candidate pool, then diversify.
    if candidate_k is None:
        candidate_k = max(
            top_k * DEFAULT_CANDIDATE_MULTIPLIER,
            top_k,
        )

    candidate_k = max(top_k, candidate_k)

    candidates = candidates[:min(candidate_k, len(candidates))]

    # --------------------------------------------------------
    # D. Diversify source pages and remove near duplicates
    # --------------------------------------------------------

    selected = []
    selected_page_counts = Counter()
    deferred = []

    for candidate in candidates:
        source_name = str(candidate.get("source", "Unknown source"))
        page_number = candidate.get("page")

        page_key = (source_name.casefold(), str(page_number))

        # Do not let a single page dominate the first pass.
        if selected_page_counts[page_key] >= max_per_source_page:
            deferred.append(candidate)
            continue

        if any(
            are_near_duplicates(candidate, existing)
            for existing in selected
        ):
            continue

        selected.append(candidate)
        selected_page_counts[page_key] += 1

        if len(selected) >= top_k:
            return selected

    # --------------------------------------------------------
    # E. Relax page diversity if there are too few results
    # --------------------------------------------------------

    # We still suppress near-duplicates, but allow a second distinct
    # passage from a page when the first pass did not return enough.
    for candidate in deferred:
        if any(
            are_near_duplicates(candidate, existing)
            for existing in selected
        ):
            continue

        selected.append(candidate)

        if len(selected) >= top_k:
            break

    return selected


# ============================================================
# 6. COMMAND-LINE SMOKE TEST
# ============================================================

if __name__ == "__main__":
    test_query = (
        "My confirmed e-ticket refund has not reached "
        "my bank account. What should I check?"
    )

    print("\nRailResolve AI — Hybrid Retrieval Test")
    print("=" * 55)

    results = hybrid_search(
        query=test_query,
        top_k=3,
        source_filter="refund_rules.pdf",
    )

    if not results:
        print("No results returned. Check the source filter and index.")
    else:
        for rank, result in enumerate(results, start=1):
            print(f"\nReference {rank}")
            print(f"Source: {result.get('source')}")
            print(f"Page: {result.get('page')}")
            print(f"Chunk ID: {result.get('chunk_id')}")
            print(f"Dense score: {result.get('dense_similarity'):.4f}")
            print(f"BM25 score: {result.get('bm25_score'):.4f}")
            print(f"Hybrid score: {result.get('hybrid_score'):.6f}")
            print(f"Text: {result.get('text', '')[:700]}")
