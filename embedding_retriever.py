"""Real CPU embeddings with visible vectors and inspectable cosine retrieval."""

import argparse
from importlib.metadata import version
from pathlib import Path
import platform

import numpy as np
from sentence_transformers import SentenceTransformer

from keyword_retriever import load_chunks, positive_int


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOLERANCE = 1e-6


def check_vectors(vectors, expected_shape):
    if vectors.shape != expected_shape or not np.isfinite(vectors).all():
        raise ValueError(f"Invalid embedding array: {vectors.shape}")
    lengths = np.linalg.norm(vectors, axis=-1)
    if not np.allclose(lengths, 1.0, atol=TOLERANCE, rtol=0):
        raise ValueError(f"Embedding lengths are not approximately 1: {lengths}")
    return lengths


def rank_scores(scores, chunks):
    """Rank every passage, including zero/negative scores; use ID for ties."""
    return sorted(range(len(chunks)), key=lambda i: (-float(scores[i]), chunks[i].id))


class EmbeddingRetriever:
    def __init__(self):
        self.chunks = load_chunks()
        self.model = SentenceTransformer(MODEL_NAME, device="cpu")
        if self.model.device.type != "cpu":
            raise RuntimeError(f"Expected CPU, got {self.model.device}")
        # Encode only complete passage text, once per retriever/run.
        encoded = self.model.encode_document(
            [chunk.text for chunk in self.chunks],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        self.model_output_dtype = str(encoded.dtype)
        # Widen model coordinates without rounding for stable inspection sums.
        self.document_vectors = np.asarray(encoded, dtype=np.float64)
        self.document_lengths = check_vectors(self.document_vectors, (6, 384))

    def score_question(self, question):
        encoded = self.model.encode_query(
            question,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        query_vector = np.asarray(encoded, dtype=np.float64)
        query_length = check_vectors(query_vector, (384,))
        scores = np.array([np.dot(query_vector, doc) for doc in self.document_vectors])
        cosine = scores / (self.document_lengths * query_length)
        np.testing.assert_allclose(scores, cosine, rtol=0, atol=TOLERANCE)
        return query_vector, scores, cosine


def coordinates(vector):
    return "[" + ", ".join(f"{value:.4f}" for value in vector[:12]) + "]"


def print_similarity_breakdown(query_vector, document_vector, chunk_id, score):
    products = query_vector * document_vector
    first = float(np.sum(products[:12]))
    remaining = float(np.sum(products[12:]))
    np.testing.assert_allclose(first + remaining, score, rtol=0, atol=1e-12)
    print(f"\nCoordinate products for the top passage {chunk_id} (1-based coordinates):")
    print("Coordinate\tQuery (4 d.p.)\tDocument (4 d.p.)\tProduct (unrounded inputs)")
    for i in range(12):
        print(f"{i + 1}\t{query_vector[i]:.4f}\t{document_vector[i]:.4f}\t{products[i]:.10f}")
    print(f"First 12 products subtotal: {first:.10f}")
    print(f"Remaining 372 products subtotal: {remaining:.10f}")
    print(f"Full 384-coordinate dot product: {score:.10f}")
    cosine = float(np.dot(query_vector, document_vector) /
                   (np.linalg.norm(query_vector) * np.linalg.norm(document_vector)))
    print(f"Cosine calculated directly from vector lengths: {cosine:.10f}")
    print(f"Absolute dot/cosine difference: {abs(float(score) - cosine):.12g}")
    print(f"Subtotal reconstruction: PASS (absolute tolerance 1e-12)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", nargs="+", help="One or more separately quoted questions")
    parser.add_argument("-k", "--k", type=positive_int, default=2)
    parser.add_argument("--save-vectors", type=Path,
                        help="Optional .npz archive of all actual coordinates and metadata")
    args = parser.parse_args()
    if args.save_vectors and args.save_vectors.suffix != ".npz":
        parser.error("--save-vectors must end in .npz")

    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"Python: {platform.python_version()}", flush=True)
    for package in ("sentence-transformers", "numpy", "torch", "transformers", "huggingface-hub"):
        print(f"{package}: {version(package)}", flush=True)
    print("Loading model on CPU...", flush=True)
    retriever = EmbeddingRetriever()
    docs = retriever.document_vectors
    print(f"Device: {retriever.model.device}")
    print(f"Model output dtype: {retriever.model_output_dtype}; calculation dtype: {docs.dtype}")
    print(f"Document matrix shape: {docs.shape}")
    print("Document encoding calls: 1; this same matrix is reused across questions.")
    print("Normalization requested for documents and questions: normalize_embeddings=True")
    print("All 384 coordinates are used in retrieval, not just the displayed first 12.")
    print("Only passage text is encoded; IDs and filenames remain separate metadata.")
    print("Displayed coordinates are rounded to four decimals; calculations use unrounded coordinates.")
    print("Individual coordinates have no assigned semantic names.")
    print("\nDocument vectors (first 12 coordinates):")
    for i, chunk in enumerate(retriever.chunks):
        print(f"{chunk.id} ({chunk.filename}): {coordinates(docs[i])}")
        print(f"  Length: {retriever.document_lengths[i]:.10f}; approximately 1: PASS")

    query_vectors = []
    score_rows = []
    for number, question in enumerate(args.questions, 1):
        vector, scores, cosine = retriever.score_question(question)
        query_vectors.append(vector)
        score_rows.append(scores)
        order = rank_scores(scores, retriever.chunks)
        selected = order[:args.k]
        print(f"\nQuestion {number}: {question}")
        print(f"Query vector shape: {vector.shape}")
        print(f"Query first 12 coordinates: {coordinates(vector)}")
        print(f"Query length: {np.linalg.norm(vector):.10f}; approximately 1: PASS")
        print(f"k: {args.k}; always select the top k available passages; no relevance threshold.")
        print("\nAll scores (descending score, then ascending chunk ID):")
        print("Rank\tID\tFilename\tDot product\tDirect cosine")
        for rank, i in enumerate(order, 1):
            chunk = retriever.chunks[i]
            print(f"{rank}\t{chunk.id}\t{chunk.filename}\t{scores[i]:.8f}\t{cosine[i]:.8f}")
        print(f"All six dot/cosine comparisons: PASS (absolute tolerance {TOLERANCE:g})")
        print("Selected IDs: " + ", ".join(retriever.chunks[i].id for i in selected))
        for i in selected:
            chunk = retriever.chunks[i]
            print(f"\nID: {chunk.id}\nFilename: {chunk.filename}\nScore: {scores[i]:.8f}")
            print(f"Text: {chunk.text}")
        top = order[0]
        print_similarity_breakdown(vector, docs[top], retriever.chunks[top].id, scores[top])

    displayed = np.vstack((docs[:, :12], np.stack(query_vectors)[:, :12]))
    labels = [chunk.id for chunk in retriever.chunks] + [f"Question {i}" for i in range(1, len(query_vectors) + 1)]
    print("\nExtrema among displayed coordinates (not across all 384 coordinates):")
    for label, flat_index in (("Smallest", np.argmin(displayed)), ("Largest", np.argmax(displayed))):
        row, col = np.unravel_index(flat_index, displayed.shape)
        print(f"{label}: {displayed[row, col]:.4f} ({labels[row]}, coordinate {col + 1})")
    print("Neither extreme alone explains a passage; similarity uses the whole vector.")

    if args.save_vectors:
        args.save_vectors.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_vectors,
            model_name=np.array(MODEL_NAME),
            document_ids=np.array([chunk.id for chunk in retriever.chunks]),
            document_filenames=np.array([chunk.filename for chunk in retriever.chunks]),
            document_texts=np.array([chunk.text for chunk in retriever.chunks]),
            document_vectors=docs,
            questions=np.array(args.questions),
            query_vectors=np.stack(query_vectors),
            scores=np.stack(score_rows),
        )
        print(f"\nSaved full unrounded vectors and scores: {args.save_vectors}")
    print("Retrieval inspection only; no answer generated.")


if __name__ == "__main__":
    main()
