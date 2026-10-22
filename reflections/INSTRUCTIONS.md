# Individual reflection — write this yourself

The assignment requires one individual reflection per participant, **no more than
150 words per person**. Do not ask Codex to write your personal explanation.
This file is an instruction sheet, not a completed reflection.

The project is confirmed as individual work, so only `participant-1.txt` is needed.

Current status: `participant-1.txt` contains the author-approved final reflection
(139 words), incorporating the author's additional ideas and editorial assistance.
The obsolete draft has been removed. The final version used AI-assisted editing;
check the course's personally written reflection requirement before submission.

In your own words, cover all three points:

1. Explain how a question passes through retrieval to become a cited answer.
2. Identify one failure or limitation you actually observed in your run.
3. Propose one change you would test next and explain what you would check.

Use your actual results and explain what you understood. No sample personal
answer has been provided.

## Save your work

- Participant 1: `reflections/participant-1.txt`.
- Participant 2, only if working in a pair: `reflections/participant-2.txt`.
- Put your name on the first line and write your reflection below it.
- Keep the reflection body within 150 words. Using English makes the stated
  word limit straightforward to check; use another language only if allowed
  by the course.
- Match your name and work arrangement to `PARTICIPANTS.md`.
- If working individually, only one reflection is required.

You can count an English reflection body from the project directory with:

```sh
.venv/bin/python -c "from pathlib import Path; p=Path('reflections/participant-1.txt'); body='\n'.join(p.read_text(encoding='utf-8').splitlines()[1:]); print(len(body.split()))"
```

For a pair, repeat the count with `participant-2.txt`. Do not submit this
instruction sheet in place of your actual individual reflection.
