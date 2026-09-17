"""
evaluate_rag.py — RAG retrieval evaluation harness

test_main.py deliberately never touches real embeddings, FAISS, or the
LLM — it's fast, network-free, and answers "is the logic correct?". It
can't answer the more important question: "does retrieval actually find
the right transcript chunk?" That's what this script is for.

This DOES need network access (to download the embedding model from
Hugging Face the first time) and takes a few seconds to run — that's
exactly why it's kept separate from the unit test suite instead of mixed
into it.

Usage:
    python evaluate_rag.py
    python evaluate_rag.py --k 5 --threshold 0.4
    python evaluate_rag.py --chunk-size 700 --chunk-overlap 150

Use the flags to compare configurations empirically — this is the tool
--show-scores was built to complement, for exactly the "don't change
CHUNK_SIZE/k/threshold blindly, evaluate first" calibration this project
has been flagging as outstanding.
"""

import argparse
import sys

import main as rag


class EvalSnippet:
    """Minimal stand-in for a youtube_transcript_api snippet."""
    def __init__(self, text, start, duration=5.0):
        self.text = text
        self.start = start
        self.duration = duration


# A small, clean, hand-checkable transcript with clearly separated topics.
# Real transcripts are messier than this — the point here isn't to judge
# embedding quality on realistic captions, it's to verify retrieval
# MECHANICS (does the right topic surface for the right question, does an
# unrelated question get rejected) with ground truth you can eyeball.
SAMPLE_TRANSCRIPT = [
    EvalSnippet("Python is a general purpose programming language created by Guido van Rossum.", start=0),
    EvalSnippet("It is known for readable syntax and a large standard library.", start=8),
    EvalSnippet("Python is widely used in data science and machine learning.", start=18),
    EvalSnippet("Popular machine learning libraries in Python include TensorFlow and PyTorch.", start=30),
    EvalSnippet("TensorFlow was originally developed by the Google Brain team.", start=42),
    EvalSnippet("Clustering is an unsupervised learning technique that groups similar data points.", start=55),
    EvalSnippet("K-means is one of the most common clustering algorithms.", start=68),
    EvalSnippet("Clustering is often used for customer segmentation in marketing.", start=80),
]

# Each case: (question, expected_timestamp_range_or_None, description).
# A (low, high) range means: at least one retrieved chunk must start
# within that window. None means retrieval is expected to return NOTHING
# — this is what actually exercises the relevance threshold, not just the
# top-k ranking.
EVAL_CASES = [
    ("What is Python used for?", (18, 42), "on-topic — should hit the ML-in-Python chunk(s)"),
    ("Who created TensorFlow?", (30, 55), "on-topic — should hit the TensorFlow chunk"),
    ("What is clustering used for?", (55, 90), "on-topic — should hit a clustering chunk"),
    ("What is the capital of France?", None, "off-topic — should retrieve nothing"),
    ("What's the weather like tomorrow?", None, "off-topic — should retrieve nothing"),
]


def build_vector_store(chunk_size: int, chunk_overlap: int):
    # create_chunks_from_snippets reads CHUNK_SIZE/CHUNK_OVERLAP as module
    # globals, so overriding them here lets this script compare
    # configurations without needing a second code path in main.py.
    rag.CHUNK_SIZE = chunk_size
    rag.CHUNK_OVERLAP = chunk_overlap

    chunks = rag.create_chunks_from_snippets(SAMPLE_TRANSCRIPT, video_id="eval", language="en")
    embeddings = rag.get_embeddings()
    return rag.FAISS.from_documents(chunks, embeddings, distance_strategy=rag.DistanceStrategy.COSINE)


def run_evaluation(k: int, threshold: float, chunk_size: int, chunk_overlap: int) -> bool:
    print(f"Config: chunk_size={chunk_size} chunk_overlap={chunk_overlap} k={k} threshold={threshold}\n")

    vector_store = build_vector_store(chunk_size, chunk_overlap)
    # Use the exact same retriever construction the real app uses, so this
    # evaluation reflects production behavior, not a hand-rolled variant.
    retriever = rag.create_retriever(vector_store, k=k, threshold=threshold)

    passed = 0
    failed = 0

    for question, expected_range, description in EVAL_CASES:
        retrieved_docs = retriever.invoke(question)
        all_scored = vector_store.similarity_search_with_relevance_scores(question, k=k)

        if expected_range is None:
            ok = len(retrieved_docs) == 0
        else:
            low, high = expected_range
            ok = any(low <= doc.metadata["start_seconds"] <= high for doc in retrieved_docs)

        status = "PASS" if ok else "FAIL"
        passed += ok
        failed += not ok

        print(f"[{status}] {question!r} — {description}")
        for doc, score in all_scored:
            ts = doc.metadata.get("timestamp", "??:??")
            preview = " ".join(doc.page_content.split())
            if len(preview) > 70:
                preview = preview[:67] + "..."
            kept = "kept   " if score >= threshold else "dropped"
            print(f"        {score:.3f} ({kept})  [{ts}]  {preview}")
        print()

    total = passed + failed
    print(f"Result: {passed}/{total} passed")
    return failed == 0


def main():
    p = argparse.ArgumentParser(description="RAG retrieval evaluation harness")
    p.add_argument("--k", type=int, default=rag.RETRIEVER_K)
    p.add_argument("--threshold", type=float, default=rag.RELEVANCE_SCORE_THRESHOLD)
    p.add_argument("--chunk-size", type=int, default=rag.CHUNK_SIZE)
    p.add_argument("--chunk-overlap", type=int, default=rag.CHUNK_OVERLAP)
    args = p.parse_args()

    rag.validate_config(
        k=args.k, threshold=args.threshold,
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
    )

    ok = run_evaluation(args.k, args.threshold, args.chunk_size, args.chunk_overlap)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
