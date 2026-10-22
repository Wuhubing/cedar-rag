"""Independently audit saved real vectors using math.fsum; no model is loaded."""

import math
from pathlib import Path

import numpy as np

from keyword_retriever import load_chunks


def main():
    root = Path(__file__).resolve().parent
    archive = root / "evidence/embedding-vectors-q1.npz"
    with np.load(archive, allow_pickle=False) as saved:
        docs = saved["document_vectors"]
        queries = saved["query_vectors"]
        scores = saved["scores"]
        ids = saved["document_ids"].tolist()
        chunks = load_chunks()
        assert saved["model_name"].item() == "sentence-transformers/all-MiniLM-L6-v2"
        assert ids == [chunk.id for chunk in chunks]
        assert saved["document_filenames"].tolist() == [chunk.filename for chunk in chunks]
        assert saved["document_texts"].tolist() == [chunk.text for chunk in chunks]
        assert saved["questions"].tolist() == [
            "May a trained graduate student operate the laser alone at 19:00?"
        ]
    assert docs.shape == (6, 384) and queries.shape == (1, 384)
    assert scores.shape == (1, 6)
    assert np.isfinite(docs).all() and np.isfinite(queries).all()
    # Check that widening to float64 preserved actual float32 model coordinates.
    assert np.array_equal(docs, docs.astype(np.float32).astype(np.float64))
    assert np.array_equal(queries, queries.astype(np.float32).astype(np.float64))
    vector = queries[0]
    norm = lambda row: math.sqrt(math.fsum(float(x) ** 2 for x in row))
    query_norm = norm(vector)
    lengths = [norm(row) for row in docs]
    assert all(abs(length - 1) <= 1e-6 for length in lengths + [query_norm])
    # These scalar sums do not call the retriever's np.dot scoring operation.
    products = [[float(q) * float(d) for q, d in zip(vector, row)] for row in docs]
    dots = [math.fsum(row) for row in products]
    cosines = [dot / (query_norm * length) for dot, length in zip(dots, lengths)]
    np.testing.assert_allclose(scores[0], dots, rtol=0, atol=1e-12)
    np.testing.assert_allclose(dots, cosines, rtol=0, atol=1e-6)
    order = sorted(range(6), key=lambda i: (-dots[i], ids[i]))
    selected = [ids[i] for i in order[:2]]
    top = order[0]
    first = math.fsum(products[top][:12])
    remaining = math.fsum(products[top][12:])
    assert math.isclose(first + remaining, dots[top], rel_tol=0, abs_tol=1e-12)
    transcript = (root / "evidence/embedding-inspection.txt").read_text(encoding="utf-8")
    assert "Device: cpu" in transcript
    assert "Selected IDs: " + ", ".join(selected) in transcript
    for rank, i in enumerate(order, 1):
        assert f"{rank}\t{ids[i]}\t{chunks[i].filename}\t{dots[i]:.8f}\t{cosines[i]:.8f}" in transcript
    for label, value in (("First 12 products subtotal", first),
                         ("Remaining 372 products subtotal", remaining),
                         ("Full 384-coordinate dot product", dots[top])):
        assert f"{label}: {value:.10f}" in transcript

    print("Independent audit of the actual saved Q1 embedding run")
    print("Method: scalar coordinate products and math.fsum; no model rerun or fabricated vectors.")
    print("PASS: archive metadata matches the six complete source texts; labels and filenames are separate.")
    print("PASS: document matrix (6, 384), question matrix (1, 384), score matrix (1, 6).")
    print("PASS: all full vectors are finite and retain the unrounded float32 model coordinates.")
    print("PASS: transcript records CPU execution.")
    for chunk_id, length in zip(ids, lengths):
        print(f"PASS: {chunk_id} vector length = {length:.10f} (within 1e-6 of 1).")
    print(f"PASS: Q1 vector length = {query_norm:.10f} (within 1e-6 of 1).")
    print("PASS: all six saved dot scores agree with independent scalar sums within 1e-12.")
    print("PASS: all six dot scores agree with direct cosine within 1e-6.")
    print("PASS: the transcript's complete ranking and scores match the saved vectors.")
    print(f"Top passage: {ids[top]}")
    print(f"First 12 products subtotal: {first:.10f}")
    print(f"Remaining 372 products subtotal: {remaining:.10f}")
    print(f"Full similarity: {dots[top]:.10f}")
    print(f"Direct cosine: {cosines[top]:.10f}")
    print(f"Absolute difference: {abs(dots[top] - cosines[top]):.12g}")
    print("PASS: the two subtotals reconstruct the full dot score within 1e-12.")
    print("Selected IDs (k=2): " + ", ".join(selected))
    required = {"A1", "A2"}
    retrieved = required & set(selected)
    missing = required - set(selected)
    print("Q1 required IDs: A1, A2")
    print(f"Required evidence coverage: {len(retrieved)}/2 = {len(retrieved) / 2:.1f}")
    print("Missing required IDs: " + (", ".join(sorted(missing)) or "none"))
    print("Evidence completeness: " + ("complete" if not missing else "partial"))
    if missing:
        print("A2 supplies the operating condition; A1's numeric access hours are missing from the selected set.")
    print("T1 supplies training requirements; it does not replace A1's access-hour evidence.")
    print("Keyword Q1 selected A1, A2; embeddings at k=2 did not improve required evidence coverage in this run.")
    print("Similarity scores and keyword overlap counts use different scales and must not be compared numerically.")
    displayed = np.vstack((docs[:, :12], vector[:12]))
    labels = ids + ["Q1"]
    for label, index in (("Smallest", np.argmin(displayed)), ("Largest", np.argmax(displayed))):
        row, col = np.unravel_index(index, displayed.shape)
        print(f"{label} displayed coordinate: {displayed[row, col]:.4f} ({labels[row]}, coordinate {col + 1}).")
    print("Coordinates have no reliable individual semantic names; the vector as a whole represents text relationships.")
    print("This is a retrieval audit, not a generated answer or a personal reflection.")


if __name__ == "__main__":
    main()
