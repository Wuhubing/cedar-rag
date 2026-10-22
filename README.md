# Cedar RAG

A teaching RAG workflow for the fictional Cedar Lab: prepare six source passages,
compare keyword and real embedding retrieval, and hand selected evidence to fresh
answer conversations for cited answers and claim audits.

The technical work is complete, and the author has confirmed **individual work**.
The project author identity is **Wuhubing** (GitHub/Git identity). The author has
approved and finalized the individual reflection. The course signup-sheet
submission remains to be completed.

**GitHub repository / link to submit in the course form:**
[https://github.com/Wuhubing/cedar-rag](https://github.com/Wuhubing/cedar-rag).
The repository is public, its default branch is `main`, and anonymous access was
verified with HTTP 200. Publication checks are recorded in
[evidence/github-publication-check.txt](evidence/github-publication-check.txt).
The course signup-sheet URL was not supplied, so the form itself has not been
submitted.

## Participants and personal work

- Author and individual status are recorded in [PARTICIPANTS.md](PARTICIPANTS.md).
  Confirm the course submission name if it differs from the Git/GitHub identity.
- Final individual reflection: **complete**, 139 words, saved as
  [reflections/participant-1.txt](reflections/participant-1.txt).
- The author explicitly requested an AI-assisted draft, provided separately in
  [reflections/reflection-draft.txt](reflections/reflection-draft.txt) for revision.
  That earlier draft is retained separately; the author-approved final version
  incorporates the author's additional reflection and editorial assistance.
- There is no partner; pair roles and role swaps are not applicable.
- Existing evidence was generated locally or in the recorded independent answer
  chats. No borrowed pair/instructor output is recorded. Label any later borrowed
  output with its source.

The reflection instructions contain the course requirements; the separate draft
is explicitly labeled with its AI assistance and draft status.

## Environment and setup

Tested with **Python 3.13.2** on this Mac, using the CPU. The model is
**`sentence-transformers/all-MiniLM-L6-v2`**, producing 384-dimensional embeddings.

| Package | Recorded and verified installed version |
| --- | --- |
| sentence-transformers | 6.2.0 |
| numpy | 2.5.4 |
| torch | 2.14.1 |
| transformers | 5.20.2 |
| huggingface-hub | 2.2.0 |

All direct and transitive installed dependencies are pinned in
[requirements.txt](requirements.txt). From the project directory, using Python
3.13, create the environment on a new checkout:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python check_environment.py
```

The initial model download requires internet access. No separate model API key
or dedicated GPU is required. Subsequent model loads can use the local Hugging
Face cache. The virtual environment and secrets are excluded from Git.

The initial run is recorded in `evidence/setup-check.txt`. The final environment
check is in `evidence/final-environment-check.txt`: the actual model loaded on
CPU, produced a real `(384,)` vector of length approximately 1, and exposed
`encode_document()` and `encode_query()`.

## Source passages and pre-run prediction

| File | Paragraph IDs |
| --- | --- |
| [docs/access.txt](docs/access.txt) | A1, A2 |
| [docs/training.txt](docs/training.txt) | T1, T2 |
| [docs/booking.txt](docs/booking.txt) | B1, B2 |

The six paragraphs are verbatim from the handout, one labeled paragraph per
chunk with a blank line between paragraphs. IDs and filenames are retained as
metadata but excluded from searched/encoded text. The source check is recorded
in `evidence/source-verification.txt`.

Before retrieval, Q1 was predicted to require **A1 and A2**: the access hours and
the two-trained-person operating condition. The prediction is preserved in
`evidence/q1-pre-run-prediction.txt`.

## Run the retrievers

From the project directory, run the keyword baseline with default `k=2`:

```sh
.venv/bin/python keyword_retriever.py 'May a trained graduate student operate the laser alone at 19:00?'
```

The program accepts a quoted question and `--k`/`-k` as a positive integer.
It prints all six scores and matched words, then up to `k` positive-scoring
passages with IDs, filenames, and complete text. It prints `No matching passages.`
if all scores are zero. It uses only the Python standard library.

Run real CPU embeddings, saving the required inspection and full vectors:

```sh
.venv/bin/python -u embedding_retriever.py 'May a trained graduate student operate the laser alone at 19:00?' --k 2 --save-vectors evidence/embedding-vectors-q1.npz > evidence/embedding-inspection.txt 2> evidence/embedding-model-log.txt
```

Omit the output redirections to see the inspection in the terminal. This program
accepts multiple separately quoted questions; it encodes the six documents once
per run and reuses their matrix across questions. It always returns the top `k`
available passages, including zero/negative scores, with no relevance threshold.
If `k` exceeds six, it returns all six available passages.

### How the code works

The keyword rule lowercases the question and passage text, extracts alphanumeric
words with `[a-z0-9]+`, and removes exactly:

```text
a, an, the, is, are, to, of, in, at, for, and, or, may, can, i, it, be
```

Its score is the number of distinct shared words. Repeats count once; no stemming,
synonyms, or semantic search are added. Sorting is by descending score, then
ascending chunk ID. In `keyword_retriever.py`, loading starts at line 29, word
preparation at line 45, scoring at lines 54–55, ranking at line 56, and positive
top-k selection at lines 59–60.

In `embedding_retriever.py`, line 35 loads the model on CPU. Lines 39–44 call
`encode_document()` with only passage texts. Lines 51–56 use the same model's
`encode_query()` for the question. Both request `normalize_embeddings=True` and
`convert_to_numpy=True`; for this model, the two methods use the same underlying
encoding behavior. The document matrix is **(6, 384)** and a question vector is
**(384,)**. Compatible vector spaces, not merely equal dimensions, are needed
to compare matching coordinates.

The normalization checks start at line 18. Line 59 computes dot products, line
60 computes cosine directly from vector lengths, line 27 sorts scores with IDs
as the tiebreaker, and lines 125–126 select the first `k`. Lines 69–85 inspect the
top passage's coordinate products and subtotals. All 384 unrounded coordinates
are used; only the first 12 are displayed to four decimals. Float32 model outputs
are widened to float64 for calculations without adding information or rounding.

We use a pretrained model, not one trained on these six paragraphs. A coordinate
has no reliable semantic name such as "safety". Encoding selects evidence; it
does not generate an answer. Ordinary RAG changes supplied context, not weights.

## Actual vector inspection

For Q1, the complete embedding ranking was A2 **0.68644575**, T1 **0.58195294**,
B1 **0.53269621**, A1 **0.37424892**, T2 **0.27045221**, B2 **0.12873971**.
The selected set at `k=2` was **A2, T1**, missing A1.

All six document vectors and Q1 have lengths within `1e-6` of 1. For Q1 and its
top passage A2, the actual coordinate calculation was:

```text
First 12 products subtotal:          0.0161114999
Remaining 372 products subtotal:     0.6703342467
Full 384-coordinate dot product:     0.6864457465
Direct cosine from vector lengths:  0.6864456610
Absolute difference:                8.54938960737e-08
```

The subtotals reconstruct the score within `1e-12`; all six dot/cosine comparisons
agree within `1e-6`. Float32 normalization gives approximately unit lengths, which
explains the tiny differences. Multiplying the displayed rounded prefixes may
give slightly different products from calculations using full coordinates.

Among the displayed prefixes, the minimum is **-0.1167** at T1 coordinate 5 and
the maximum is **0.1357** at T1 coordinate 12 (1-based). Neither number alone
explains a passage; the vector as a whole supports text comparisons.

## Four-question comparison and k experiment

Run both methods for Q1–Q4 at `k=2`, then Q1 embeddings at `k=1` and `k=3`:

```sh
.venv/bin/python compare_retrieval.py
.venv/bin/python audit_retrieval.py
```

The first command captures actual CLI output and writes blank judgment columns
to `results.csv`, preserving the original blank table as
`evidence/results-unreviewed.csv`. The second checks saved full vectors and fills
the final judgments. Rerunning the first command replaces run evidence and
resets the final table, so follow it with the audit command. Answer transcripts
are preserved separately; new retrieval selections require a new answer handoff.

| Question | Keyword IDs | Coverage | Embedding IDs | Coverage |
| --- | --- | --- | --- | --- |
| Q1 | A1, A2 | 2/2, complete | A2, T1 | 1/2, partial |
| Q2 | A2, B1 | 1/1, complete | A2, T1 | 1/1, complete |
| Q3 | T1, A2 | 1/1, complete | T1, B1 | 1/1, complete |
| Q4 | A2, B1 | N/A, answer absent | T1, A2 | N/A, answer absent |

For Q1, keyword A1 scores 3 for `00`, `graduate`, `trained`; A2 scores 2 for
`laser`, `trained`. Repeated `00` and `laser` count once. `student` does not match
`students`, and `operate` does not match `operating`. Matching `00` is literal
overlap, not reasoning about time intervals. B1, T1, and T2 each score 1 and
are ordered by ID; B2 scores 0.

Q2 retrieves A2 with keyword score **2** (`laser`, `present`) and embedding
similarity **0.58519334**. Q3 retrieves T1 with keyword score **2** (`laser`,
`must`) and embedding similarity **0.74202748**. Both paraphrases retrieve the
target with both methods. Keyword success still comes from literal shared words.

Q4 has no wavelength information in the six-source corpus. Its embedding results
are T1 **0.34772135** and A2 **0.32082712**. Top-k selection returns relative
neighbors even when no passage answers the question; positive similarity does
not establish answerability. Keyword retrieval also returns laser-related text.
Keyword counts and embedding scores use different scales and are not numerically
comparable. Embeddings need not win this small classroom comparison.

[results.csv](results.csv) has four rows and exactly the nine required columns:

```text
question_id,keyword_top_ids,keyword_scores,embedding_top_ids,embedding_scores,required_ids,keyword_evidence_complete,embedding_evidence_complete,observation
```

Semicolons separate IDs/scores in matching rank order. Completeness is `true`
only when all required evidence is selected. Q4 uses `required_ids=absent`, both
judgments `false`, and coverage N/A rather than 0/0 or 100%. The filled judgments
are Codex-assisted and accompanied by source-based audits.

| Q1 embedding k | Actual selected IDs | Required coverage |
| --- | --- | --- |
| 1 | A2 | 1/2 |
| 2 | A2, T1 | 1/2 |
| 3 | A2, T1, B1 | 1/2 |

Increasing k adds training conditions (T1), then booking limits/calendar
instructions (B1). Neither supplies A1's numeric access hours. A1 remains fourth;
all tested selections are incomplete. The k=2 row comes from the four-question
run; k=1 and k=3 were run separately with identical questions and passages.

## Cited answers and claim audits

Q1 received only A2/T1; Q4 received only T1/A2, in their actual rank order. Each
initial generation used a fresh projectless Codex chat outside this repository,
with only the question, selected paragraphs, and the exact assignment prompt.
No project files, full corpus, earlier answer, or desired conclusion were
provided. Saved traces show no answer-assistant browsing, file access, or other
tool calls under the no-tools instruction. This is observed behavior, not an
OS-enforced filesystem sandbox. The default generator model's identifier was
not exposed by the chat tools; no unverified model name is claimed.

- **Q1:** the original supported the operating/training claims but omitted the
  missing access-hour evidence. One added instruction was tested in another fresh
  chat with identical sources. The revised answer retains the two-trained-person
  condition and states that the passages cannot establish whether 19:00 is within
  permitted access hours. Retrieval still has only 1/2 of required evidence.
- **Q4:** the original correctly said the wavelength was missing, but its answer
  sentence lacked attached chunk references. One instruction revision, again in
  a fresh chat with identical sources, made `[T1] [A2]` explicit as inspected
  passages, not as proof of a numerical wavelength. No value was invented.

Both original answers, the single revisions, raw outputs, delivered inputs and
conversation traces are preserved. The two claim-by-claim audits found no
unsupported factual assertions in the revised responses. Q4's appropriate
missing-information answer does not turn absent answer evidence into retrieval
success. This manual handoff is a teaching workflow, not a deployed application.

## Evidence index

| Required material | Saved files |
| --- | --- |
| Source fidelity and prediction | `evidence/source-verification.txt`, `evidence/q1-pre-run-prediction.txt` |
| Initial/final environment and versions | `evidence/setup-check.txt`, `evidence/final-environment-check.txt`, `requirements.txt` |
| Keyword Q1 output and direct count | `evidence/keyword-q1.txt`, `evidence/keyword-q1-audit.txt` |
| Real vectors, dimensions, norms and products | `evidence/embedding-inspection.txt`, `evidence/embedding-audit.txt` |
| Four-question actual rankings | `evidence/retrieval-runs.txt`, `evidence/retrieval-records.json` |
| Final and original blank results | `results.csv`, `evidence/results-unreviewed.csv`, `evidence/retrieval-audit.txt` |
| Actual k runs and interpretation | `evidence/k-experiment.txt`, `evidence/k-experiment-audit.txt`, `evidence/k-records.json` |
| Q1 answer audit | [evidence/q1-answer-audit.txt](evidence/q1-answer-audit.txt) |
| Q4 answer audit | [evidence/q4-answer-audit.txt](evidence/q4-answer-audit.txt) |
| Original/revised answer provenance | `evidence/answer-generation-manifest.json`, `evidence/q1-answer-*-trace.json`, `evidence/q4-answer-*-trace.json` |
| Handoff verification | `evidence/answer-handoff-verification.txt` |
| Final commands/completeness check | `evidence/final-command-checks.txt`, `evidence/submission-check.txt` (historical Step 7 baseline) |
| GitHub publication and submission URL | [evidence/github-publication-check.txt](evidence/github-publication-check.txt) |

The `.npz` archives contain all unrounded coordinates, scores, questions, source
texts and separate IDs/filenames: `embedding-vectors-q1.npz`,
`embedding-vectors-q1-q4.npz`, `embedding-vectors-q1-k1.npz`, and
`embedding-vectors-q1-k3.npz`, all under `evidence/`. Use
`numpy.load(path, allow_pickle=False)`; array row order is in `document_ids`.
Model stderr is saved separately in the matching model-log files.

## Verify existing evidence

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python verify_embedding_evidence.py
.venv/bin/python prepare_answer_handoffs.py
.venv/bin/python verify_answer_handoffs.py
```

The six keyword tests check exact stopwords, no stemming, duplicate words,
metadata exclusion, real Q1 scores/ties, positive selection, no-match behavior,
and the k argument. The vector checker independently reconstructs saved scores
with scalar products and `math.fsum`. Handoff checks compare exact inputs,
responses, selected evidence and independent-chat traces; they do not call a
model or replace the semantic claim audits.

The final command check ran the comparison/audit and verification commands in a
disposable project copy, reusing the installed environment and model cache.
It preserved the submitted raw evidence. This verifies commands in the existing
environment, not a fresh installation on another machine. A cloned checkout can
be rebuilt using the setup commands and pinned requirements above.

## Before submission

- [x] Three verbatim source files, two retrievers, and explanatory run commands.
- [x] Model/package versions, real CPU vectors, normalization/similarity evidence.
- [x] Four comparison rows with judgments and the k experiment.
- [x] Both answer audits, preserved originals, and single instruction revisions.
- [x] Record individual work and the Git/GitHub author identity in `PARTICIPANTS.md`.
- [x] Save the author-approved final individual reflection (at most 150 words).
- [ ] Confirm the course submission name if it differs from the Git/GitHub identity.
- [ ] Review the Codex-assisted judgments and confirm you can explain the code.
- [x] Push this project to a same-name public GitHub repository and verify anonymous access.
- [ ] Submit the repository link through the course signup sheet.

The earlier AI-assisted reflection draft is preserved. The final reflection was
approved by the author after adding their own ideas and is saved separately.
The course signup-sheet URL was not supplied and has not been guessed.
