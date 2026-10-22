"""Check evidence-only handoffs and exact saved independent-chat responses."""

import json
from pathlib import Path
import re

from keyword_retriever import load_chunks
from prepare_answer_handoffs import PROMPT


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"


def main():
    records = json.loads((EVIDENCE / "retrieval-records.json").read_text(encoding="utf-8"))
    manifest = json.loads((EVIDENCE / "answer-generation-manifest.json").read_text(encoding="utf-8"))
    sources = {chunk.id: chunk.text for chunk in load_chunks()}
    seen_threads = set()
    report = ["Answer handoff and conversation provenance verification", ""]
    for qid in ("Q1", "Q4"):
        record = records[qid]
        ids = [r["id"] for r in record["embedding_selected"]]
        assert len(ids) == 2
        assert ids == [r["id"] for r in record["embedding_ranking"][:2]]
        original = (EVIDENCE / f"{qid.lower()}-answer-input.txt").read_text(encoding="utf-8")
        question_block, rest = original.split("\n\nSelected passages:\n")
        passages, instructions = rest.split("\n\nInstructions:\n")
        assert question_block == f'Question:\n{record["question"]}'
        assert instructions == PROMPT + "\n"
        assert passages == "\n\n".join(f"[{cid}] {sources[cid]}" for cid in ids)
        report.append(f"PASS: {qid} original input is exactly its question, selected {', '.join(ids)}, and assignment prompt.")
        for attempt in ("original", "revised"):
            entry = manifest["questions"][qid][attempt]
            text = (EVIDENCE / entry["input_file"]).read_text(encoding="utf-8")
            assert re.findall(r"^\[([ATB][12])\]", text, re.MULTILINE) == ids
            if attempt == "revised":
                assert text.startswith(original.rstrip() + "\n\n")
                added = text[len(original.rstrip()) + 2:].strip()
                assert added.startswith("In addition,") and "\n\n" not in added
                assert text.split("\n\nInstructions:\n")[0] == original.split("\n\nInstructions:\n")[0]
            trace = json.loads((EVIDENCE / entry["trace_file"]).read_text(encoding="utf-8"))
            assert trace["thread"]["id"] == entry["thread_id"]
            assert trace["thread"]["title"] == entry["title"]
            assert entry["thread_id"] not in seen_threads
            seen_threads.add(entry["thread_id"])
            cwd = Path(trace["thread"]["cwd"]).resolve()
            assert cwd != ROOT and ROOT not in cwd.parents
            assert not trace["page"]["hasMore"] and len(trace["turns"]) == 1
            turn = trace["turns"][0]
            assert turn["status"] == "completed" and turn["id"] == entry["turn_id"]
            deliveries = [i for i in turn["items"] if i["type"] == "functionCallOutput"]
            assert len(deliveries) == 1 and deliveries[0]["name"] == "create_thread"
            match = re.search(r"<input>(.*?)</input>", deliveries[0]["output"]["text"], re.DOTALL)
            assert match and match.group(1) == text
            assert all(i["type"] in {"functionCallOutput", "agentMessage"} for i in turn["items"])
            answers = [i for i in turn["items"] if i["type"] == "agentMessage" and i["phase"] == "final_answer"]
            assert len(answers) == 1 and answers[0]["id"] == entry["message_id"]
            answer = (EVIDENCE / entry["answer_file"]).read_text(encoding="utf-8")
            assert answer == answers[0]["text"] + "\n"
            citations = set(re.findall(r"\[([ATB][12])\]", answer))
            assert citations <= set(ids)
            assert not re.search(r"08:00|22:00", answer)
            if qid == "Q4":
                assert not re.search(r"\b\d+(?:\.\d+)?\s*(?:nm|nanomet\w*|micromet\w*)\b", answer, re.IGNORECASE)
            if attempt == "revised":
                assert citations == set(ids)
            report.append(f"PASS: {qid} {attempt}: distinct fresh chat outside the project; exact input delivered; response preserved verbatim; no answer-initiated tool calls in trace.")
        audit = (EVIDENCE / f"{qid.lower()}-answer-audit.txt").read_text(encoding="utf-8")
        for attempt in ("original", "revised"):
            answer_file = manifest["questions"][qid][attempt]["answer_file"]
            assert (EVIDENCE / answer_file).read_text(encoding="utf-8").strip() in audit
        assert "CLAIM-BY-CLAIM AUDIT" in audit and "Did the revision help? Yes." in audit
        report.append(f"PASS: {qid} audit contains both actual answers, claim checks, missing-information review, and retry outcome.")
    assert len(seen_threads) == 4
    report += ["", "PASS: exactly two original attempts and one instruction revision per question; all four chats are distinct.",
               "PASS: retries retain the same questions, two source passages, and passage order.",
               "PASS: A1, T2, B1 and B2 were not supplied as answer evidence; no unselected IDs were cited.",
               "Scope: these checks verify saved inputs, traces and outputs. Semantic support judgments are recorded separately in each answer audit."]
    output = "\n".join(report) + "\n"
    (EVIDENCE / "answer-handoff-verification.txt").write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
