
from pathlib import Path
import argparse
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Project configuration
ARTIFACT_DIR = Path("artifacts/vector_index")

CHUNKS_FILE = ARTIFACT_DIR / "chunks.json"
INDEX_FILE = ARTIFACT_DIR / "policy.faiss"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks():
    """Load extracted policy passages."""

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            "chunks.json not found. Run document_processor first."
        )

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        raise ValueError("No text chunks found.")

    return chunks


def load_model():
    """Load the sentence embedding model."""

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded successfully.")

    return model


def build_index():
    """Convert text chunks into embeddings and save FAISS index."""

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks()
    model = load_model()

    texts = [chunk["text"] for chunk in chunks]

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    # Create FAISS index using inner product.
    # Because vectors are normalized, this is cosine similarity.
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_FILE))

    print("\n========== INDEX SUMMARY ==========")
    print("Total chunks:", len(chunks))
    print("Embedding dimensions:", dimension)
    print("Indexed passages:", index.ntotal)
    print("Saved index:", INDEX_FILE)
    print("===================================")


def search_policy(question: str, top_k: int = 5):
    """Retrieve the most relevant policy passages."""

    if not question or not question.strip():
        raise ValueError("Please enter a valid search question.")

    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            "FAISS index not found. Run retriever --build first."
        )

    chunks = load_chunks()

    index = faiss.read_index(str(INDEX_FILE))

    if index.ntotal != len(chunks):
        raise ValueError(
            "The index and chunks have different sizes. "
            "Rebuild the index."
        )

    model = load_model()

    print("\nSearching policy documents...")

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    k = min(max(1, top_k), len(chunks))

    scores, indices = index.search(query_embedding, k)

    results = []

    for score, chunk_index in zip(scores[0], indices[0]):

        if chunk_index < 0:
            continue

        chunk = chunks[int(chunk_index)]

        results.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "similarity": float(score)
        })

    return results


def main():

    parser = argparse.ArgumentParser(
        description="RailResolve AI Policy Retriever"
    )

    parser.add_argument(
        "--build",
        action="store_true",
        help="Create FAISS vector index"
    )

    parser.add_argument(
        "--query",
        type=str,
        help="Search the railway policy documents"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of passages to retrieve"
    )

    args = parser.parse_args()

    if args.build:
        build_index()

    elif args.query:
        results = search_policy(
            args.query,
            top_k=args.top_k
        )

        print("\n========== SEARCH RESULTS ==========")

        for rank, result in enumerate(results, start=1):

            print(f"\nResult {rank}")
            print("Chunk ID:", result["chunk_id"])
            print("Source:", result["source"])
            print("Page:", result["page"])
            print(
                "Similarity:",
                round(result["similarity"], 4)
            )
            print("Passage:")
            print(result["text"])

        print("\n====================================")

    else:
        parser.error("Please provide --build or --query.")


if __name__ == "__main__":
    main()
