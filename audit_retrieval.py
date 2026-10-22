"""Audit actual Task 3 records, then fill the requested comparison CSV."""

import csv
import json
import math
from pathlib import Path

import numpy as np

from compare_retrieval import COLUMNS, QUESTIONS, REQUIRED, write_csv
from keyword_retriever import load_chunks


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"


def observation(qid, record):
    summaries = []
    for method in ("keyword", "embedding"):
        top = record[f"{method}_selected"]
        items = ", ".join(f'{r["id"]} ({r["score"] if method == "keyword" else format(r["score"], ".8f")})' for r in top)
        summaries.append(f"{method.capitalize()} selects {items}.")
        if qid != "Q4":
            target = set(REQUIRED[qid])
            ids = {r["id"] for r in top}
            missing = ", ".join(sorted(target - ids)) or "none"
            summaries.append(f"Required evidence coverage {len(target & ids)}/{len(target)}; missing {missing}.")
    if qid == "Q1":
        a1 = next(r for r in record["embedding_ranking"] if r["id"] == "A1")
        summaries.append(f'A1 embedding rank is {a1["rank"]} ({a1["score"]:.8f}). T1 does not supply access hours.')
    elif qid in ("Q2", "Q3"):
        target = REQUIRED[qid][0]
        keyword = next(r for r in record["keyword_ranking"] if r["id"] == target)
        summaries.append(f'{target} shared keyword words: {", ".join(keyword["matched_words"])}. The paraphrase is judged by whether its required passage is selected.')
    else:
        summaries.append("Wavelength is absent from all six sources. Both completeness fields are false. Nearest passages and positive scores do not establish answerability. Coverage ratio is not applicable.")
    return " ".join(summaries)


def verify_archive(path, questions, rankings):
    """Independently reconstruct every similarity with scalar coordinate sums."""
    chunks = load_chunks()
    with np.load(path, allow_pickle=False) as saved:
        assert saved["model_name"].item() == "sentence-transformers/all-MiniLM-L6-v2"
        assert saved["document_ids"].tolist() == [c.id for c in chunks]
        assert saved["document_filenames"].tolist() == [c.filename for c in chunks]
        assert saved["document_texts"].tolist() == [c.text for c in chunks]
        assert saved["questions"].tolist() == questions
        docs, queries, scores = saved["document_vectors"], saved["query_vectors"], saved["scores"]
    assert docs.shape == (6, 384)
    assert queries.shape == (len(questions), 384)
    assert scores.shape == (len(questions), 6)
    assert np.isfinite(docs).all() and np.isfinite(queries).all() and np.isfinite(scores).all()
    for vector in [*docs, *queries]:
        length = math.sqrt(math.fsum(float(x) ** 2 for x in vector))
        assert abs(length - 1) < 1e-6
    for q, ranking in zip(queries, rankings):
        reconstructed = [math.fsum(float(x) * float(y) for x, y in zip(q, d)) for d in docs]
        order = sorted(range(6), key=lambda i: (-reconstructed[i], chunks[i].id))
        assert [chunks[i].id for i in order] == [r["id"] for r in ranking]
        for i, row in zip(order, ranking):
            assert abs(reconstructed[i] - row["score"]) <= 5.1e-9
    np.testing.assert_allclose(queries @ docs.T, scores, rtol=0, atol=1e-12)
    return docs, queries, scores


def main():
    records = json.loads((EVIDENCE / "retrieval-records.json").read_text(encoding="utf-8"))
    experiments = json.loads((EVIDENCE / "k-records.json").read_text(encoding="utf-8"))
    assert list(records) == list(QUESTIONS)
    chunks = load_chunks()
    source_by_id = {chunk.id: chunk for chunk in chunks}
    with (EVIDENCE / "results-unreviewed.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == COLUMNS
        rows = list(reader)
    assert [row["question_id"] for row in rows] == list(QUESTIONS)
    assert all(row[key] == "" for row in rows for key in COLUMNS[-3:])
    docs, queries, scores = verify_archive(
        EVIDENCE / "embedding-vectors-q1-q4.npz", list(QUESTIONS.values()),
        [record["embedding_ranking"] for record in records.values()])

    report = ["Task 3 retrieval audit (Codex-assisted, after recording actual rankings)",
              "Raw rankings: retrieval-runs.txt; blank judgments: results-unreviewed.csv.",
              "PASS: four actual queries, six source vectors, CPU run, normalized full vectors.",
              "PASS: all embedding rankings and printed scores independently reconstructed from saved vectors.",
              "Judgment convention: true = all required evidence is present; false = incomplete or absent.",
              "For Q4, required_ids=absent, both judgments=false, and coverage=N/A (no answer in corpus).", ""]
    raw_output = (EVIDENCE / "retrieval-runs.txt").read_text(encoding="utf-8")
    assert "Device: cpu" in raw_output
    for row, (qid, record) in zip(rows, records.items()):
        assert record["question"] == QUESTIONS[qid]
        assert record["required_ids"] == REQUIRED[qid]
        selected = {}
        for method in ("keyword", "embedding"):
            ranking = record[f"{method}_ranking"]
            assert len(ranking) == 6 and {r["id"] for r in ranking} == set(source_by_id)
            assert ranking == sorted(ranking, key=lambda r: (-r["score"], r["id"]))
            candidates = [r for r in ranking if r["score"] > 0] if method == "keyword" else ranking
            assert record[f"{method}_selected"] == candidates[:2]
            top = record[f"{method}_selected"]
            selected[method] = {r["id"] for r in top}
            assert row[f"{method}_top_ids"] == ";".join(r["id"] for r in top)
            formatted = [str(r["score"]) if method == "keyword" else f'{r["score"]:.8f}' for r in top]
            assert row[f"{method}_scores"] == ";".join(formatted)
            complete = qid != "Q4" and set(REQUIRED[qid]) <= selected[method]
            row[f"{method}_evidence_complete"] = str(complete).lower()
            for item in top:
                assert source_by_id[item["id"]].text in raw_output
        row["observation"] = observation(qid, record)
        report += [f"{qid}: {QUESTIONS[qid]}", row["observation"]]
        for method in ("keyword", "embedding"):
            if qid == "Q4":
                report.append(f"  {method}: evidence_complete=false; coverage=N/A; missing wavelength information.")
            else:
                required = set(REQUIRED[qid])
                found = len(required & selected[method])
                missing = ", ".join(sorted(required - selected[method])) or "none"
                report.append(f"  {method}: evidence_complete={row[method + '_evidence_complete']}; coverage={found}/{len(required)}; missing={missing}.")
        report.append("")
    report += ["Q4 missing-information check: inspected all six original passages.",
               "A1 covers access hours; A2 covers two-person operation; T1 covers training renewal;",
               "T2 covers visitors; B1 covers bookings; B2 covers cancellation. None specifies a wavelength.",
               "Embedding retrieval returns the top k relative similarities even when the corpus cannot answer.",
               "Keyword retrieval also returns laser-related passages for Q4 because of literal overlap.",
               "Neither method's positive score establishes the presence of wavelength evidence.",
               "No generated answers or personal reflections are included."]

    k_report = ["Q1 k experiment audit (actual runs in k-experiment.txt)",
                "Required IDs: A1, A2. Model, question, corpus, and scoring stayed the same.", ""]
    assert [e["k"] for e in experiments] == [1, 3]
    for experiment in experiments:
        k = experiment["k"]
        k_docs, k_queries, k_scores = verify_archive(
            EVIDENCE / f"embedding-vectors-q1-k{k}.npz", [QUESTIONS["Q1"]], [experiment["ranking"]])
        np.testing.assert_allclose(k_docs, docs, rtol=0, atol=1e-7)
        np.testing.assert_allclose(k_queries, queries[:1], rtol=0, atol=1e-7)
        np.testing.assert_allclose(k_scores, scores[:1], rtol=0, atol=1e-7)
        assert experiment["selected"] == experiment["ranking"][:k]
    descriptions = {
        "A1": "numeric access hours and written approval outside those hours",
        "A2": "the two-trained-person operating condition",
        "T1": "training renewal and a current training record",
        "T2": "the visitor accompaniment condition",
        "B1": "the two-hour booking limit and equipment-calendar reservation",
        "B2": "the four-hour cancellation deadline",
    }
    previous = set()
    for k in (1, 2, 3):
        selected = records["Q1"]["embedding_ranking"][:k]
        ids = [r["id"] for r in selected]
        found = len(set(ids) & {"A1", "A2"})
        missing = ", ".join(sorted({"A1", "A2"} - set(ids))) or "none"
        k_report.append(f"k={k}: selected={', '.join(ids)}; scores={', '.join(format(r['score'], '.8f') for r in selected)}; coverage={found}/2; evidence_complete={str(found == 2).lower()}; missing={missing}.")
        for chunk_id in ids:
            if chunk_id not in previous:
                k_report.append(f"  Newly selected {chunk_id}: {descriptions[chunk_id]}.")
        previous = set(ids)
    a1 = next(r for r in records["Q1"]["embedding_ranking"] if r["id"] == "A1")
    k_report += ["", "Training and booking details do not replace the numeric access-hour evidence in A1.",
                 f'A1 rank: {a1["rank"]}; similarity: {a1["score"]:.8f}.',
                 "PASS: independently reconstructed rankings/scores match the saved raw k runs.",
                 "PASS: document/query vectors and scores agree across runs within 1e-7.",
                 "No additional k value was tested, and no ranking was adjusted."]
    # Preserve the raw CSV with blank judgments, then fill the requested final table.
    write_csv(ROOT / "results.csv", rows)
    (EVIDENCE / "reviewed-results.json").write_text(
        json.dumps({"columns": COLUMNS, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    (EVIDENCE / "retrieval-audit.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    (EVIDENCE / "k-experiment-audit.txt").write_text("\n".join(k_report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print("\n".join(k_report))
    print("Saved results.csv with all nine required columns, four rows and audited judgments.")


if __name__ == "__main__":
    main()
