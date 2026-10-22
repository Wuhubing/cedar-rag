"""Run the existing retrievers unchanged and save actual Task 3 evidence."""

import csv
import json
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
QUESTIONS = {
    "Q1": "May a trained graduate student operate the laser alone at 19:00?",
    "Q2": "Can I use the laser without another person present?",
    "Q3": "How frequently must I refresh my laser qualification?",
    "Q4": "What is the laser wavelength?",
}
REQUIRED = {"Q1": ["A1", "A2"], "Q2": ["A2"], "Q3": ["T1"], "Q4": "absent"}
COLUMNS = [
    "question_id", "keyword_top_ids", "keyword_scores", "embedding_top_ids",
    "embedding_scores", "required_ids", "keyword_evidence_complete",
    "embedding_evidence_complete", "observation",
]


def run(script, arguments):
    command = [sys.executable, str(ROOT / script), *arguments]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"{shlex.join(command)}\n{completed.stderr}\n{completed.stdout}")
    return shlex.join(command), completed.stdout, completed.stderr


def keyword_records(output):
    rows = []
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) == 4 and fields[0] in {"A1", "A2", "T1", "T2", "B1", "B2"}:
            chunk_id, filename, score, matched = fields
            rows.append({"id": chunk_id, "filename": filename, "score": int(score),
                         "matched_words": [] if matched == "(none)" else matched.split(", ")})
    if len(rows) != 6:
        raise ValueError("Expected all six actual keyword scores")
    return rows


def embedding_records(output):
    rows = []
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) == 5 and fields[0].isdigit():
            rank, chunk_id, filename, dot, cosine = fields
            rows.append({"rank": int(rank), "id": chunk_id, "filename": filename,
                         "score": float(dot), "cosine": float(cosine)})
    if len(rows) != 6:
        raise ValueError("Expected all six actual embedding scores")
    return rows


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    EVIDENCE.mkdir(exist_ok=True)
    transcript = ["Actual retrieval runs: Q1-Q4; keyword and embedding; k=2.",
                  "Rankings were recorded before evidence-completeness judgments.", ""]
    records = {}
    for question_id, question in QUESTIONS.items():
        print(f"Running {question_id}: keyword k=2", flush=True)
        command, output, log = run("keyword_retriever.py", [question, "--k", "2"])
        transcript.extend([f"=== {question_id}: keyword ===", f"Command: {command}", output])
        if log:
            transcript.extend(["stderr:", log])
        rows = keyword_records(output)
        records[question_id] = {"question": question, "required_ids": REQUIRED[question_id],
                                "keyword_ranking": rows,
                                "keyword_selected": [r for r in rows if r["score"] > 0][:2]}

    print("Running Q1-Q4: real CPU embeddings k=2 (one document encoding)", flush=True)
    command, output, log = run("embedding_retriever.py", [
        *QUESTIONS.values(), "--k", "2", "--save-vectors", "evidence/embedding-vectors-q1-q4.npz"
    ])
    (EVIDENCE / "comparison-model-log.txt").write_text(log, encoding="utf-8")
    transcript.extend(["=== Q1-Q4: embeddings ===", f"Command: {command}", output])
    (EVIDENCE / "retrieval-runs.txt").write_text("\n".join(transcript), encoding="utf-8")
    for number, question_id in enumerate(QUESTIONS, 1):
        section = output.split(f"\nQuestion {number}: ", 1)[1]
        if number < len(QUESTIONS):
            section = section.split(f"\nQuestion {number + 1}: ", 1)[0]
        rows = embedding_records(section)
        records[question_id]["embedding_ranking"] = rows
        records[question_id]["embedding_selected"] = rows[:2]

    csv_rows = []
    for question_id, record in records.items():
        keyword = record["keyword_selected"]
        embedding = record["embedding_selected"]
        csv_rows.append({
            "question_id": question_id,
            "keyword_top_ids": ";".join(r["id"] for r in keyword),
            "keyword_scores": ";".join(str(r["score"]) for r in keyword),
            "embedding_top_ids": ";".join(r["id"] for r in embedding),
            "embedding_scores": ";".join(f'{r["score"]:.8f}' for r in embedding),
            "required_ids": "absent" if question_id == "Q4" else ";".join(REQUIRED[question_id]),
            "keyword_evidence_complete": "", "embedding_evidence_complete": "", "observation": "",
        })
    write_csv(EVIDENCE / "results-unreviewed.csv", csv_rows)
    write_csv(ROOT / "results.csv", csv_rows)
    (EVIDENCE / "retrieval-records.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Saved actual rankings and CSV with judgments left blank.", flush=True)

    experiments = []
    k_transcript = ["Actual Q1 embedding runs with k=1 and k=3.",
                    "The question, model, corpus, normalization and scoring rule are unchanged.", ""]
    for k in (1, 3):
        print(f"Running Q1: real CPU embeddings k={k}", flush=True)
        command, k_output, log = run("embedding_retriever.py", [
            QUESTIONS["Q1"], "--k", str(k), "--save-vectors", f"evidence/embedding-vectors-q1-k{k}.npz"
        ])
        (EVIDENCE / f"k{k}-model-log.txt").write_text(log, encoding="utf-8")
        k_transcript.extend([f"=== Q1: embeddings k={k} ===", f"Command: {command}", k_output])
        rows = embedding_records(k_output)
        experiments.append({"k": k, "ranking": rows, "selected": rows[:k]})
    (EVIDENCE / "k-experiment.txt").write_text("\n".join(k_transcript), encoding="utf-8")
    (EVIDENCE / "k-records.json").write_text(
        json.dumps(experiments, indent=2) + "\n", encoding="utf-8")
    print("Saved both actual k experiments. No answer was generated.")


if __name__ == "__main__":
    main()
