"""Inspectable word-overlap retrieval using only Python's standard library."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import re


STOPWORDS = frozenset(
    "a an the is are to of in at for and or may can i it be".split()
)
DOCS_DIR = Path(__file__).resolve().parent / "docs"


@dataclass(frozen=True)
class Chunk:
    id: str
    filename: str
    text: str


@dataclass(frozen=True)
class Result:
    chunk: Chunk
    score: int
    matched_words: tuple[str, ...]


def load_chunks(docs_dir=DOCS_DIR):
    """Keep each paragraph intact; separate labels from searchable text."""
    chunks = []
    for filename in ("access.txt", "training.txt", "booking.txt"):
        contents = (Path(docs_dir) / filename).read_text(encoding="utf-8")
        for paragraph in re.split(r"\n\s*\n", contents.strip()):
            match = re.fullmatch(r"\[([A-Z][0-9]+)\] (.+)", paragraph, re.DOTALL)
            if match is None:
                raise ValueError(f"Invalid labeled paragraph in {filename}")
            chunk_id, text = match.groups()
            chunks.append(Chunk(chunk_id, filename, text))
    if len(chunks) != 6 or len({chunk.id for chunk in chunks}) != 6:
        raise ValueError("Expected exactly six chunks with unique IDs")
    return chunks


def words(text):
    """Distinct lowercase alphanumeric words, minus exactly the stopwords."""
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOPWORDS


def rank_chunks(question, chunks):
    query_words = words(question)
    results = []
    for chunk in chunks:
        matched_words = tuple(sorted(query_words & words(chunk.text)))
        results.append(Result(chunk, len(matched_words), matched_words))
    return sorted(results, key=lambda result: (-result.score, result.chunk.id))


def select_chunks(ranked_results, k=2):
    return [result for result in ranked_results if result.score > 0][:k]


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("k must be at least 1")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question to search for (quote the text)")
    parser.add_argument("-k", "--k", type=positive_int, default=2,
                        help="Maximum positive-scoring chunks to select (default: 2)")
    args = parser.parse_args()
    chunks = load_chunks()
    ranked = rank_chunks(args.question, chunks)
    selected = select_chunks(ranked, args.k)

    print(f"Question: {args.question}")
    print("Method: keyword (distinct word overlap)")
    print(f"k: {args.k}")
    print(f"Chunks loaded: {len(chunks)}")
    print("Query words: " + (", ".join(sorted(words(args.question))) or "(none)"))
    print("\nAll scores (descending score, then ascending chunk ID):")
    print("ID\tFilename\tScore\tMatched words")
    for result in ranked:
        print(f"{result.chunk.id}\t{result.chunk.filename}\t{result.score}\t"
              + (", ".join(result.matched_words) or "(none)"))

    if not selected:
        print("\nNo matching passages.")
        return
    print("\nSelected IDs: " + ", ".join(result.chunk.id for result in selected))
    for result in selected:
        print(f"\nID: {result.chunk.id}")
        print(f"Filename: {result.chunk.filename}")
        print(f"Score: {result.score}")
        print(f"Text: {result.chunk.text}")


if __name__ == "__main__":
    main()
