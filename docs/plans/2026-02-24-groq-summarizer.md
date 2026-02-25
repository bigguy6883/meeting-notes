# Groq Summarizer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace local Ollama summarization with Groq API to make post-recording processing dramatically faster.

**Architecture:** Swap `ollama.chat` for `groq.chat.completions.create` in `summarizer.py`. Rename the model parameter plumbed through `jobs.py` and `app.py`. Groq is already installed and authenticated via `GROQ_API_KEY`.

**Tech Stack:** Python, `groq` SDK (already in requirements), Flask, pytest

---

### Task 1: Update summarizer.py and its tests

**Files:**
- Modify: `summarizer.py`
- Modify: `tests/test_summarizer.py`

**Step 1: Update test mocks to expect Groq client**

Replace the entire content of `tests/test_summarizer.py`:

```python
from unittest.mock import patch, MagicMock
from summarizer import summarize

def _mock_groq_response(content):
    mock = MagicMock()
    mock.choices[0].message.content = content
    return mock

def test_summarize_returns_string():
    with patch("summarizer._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            "- Discussed roadmap\nAction: Alice to follow up"
        )
        result = summarize("This is a transcript about the Q1 roadmap.")
    assert isinstance(result, str)
    assert len(result) > 0

def test_summarize_sends_transcript_in_prompt():
    transcript = "We talked about deadlines and deliverables."
    with patch("summarizer._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_groq_response("Summary here")
        summarize(transcript)
    call_args = mock_client.chat.completions.create.call_args
    messages = call_args[1]["messages"]
    assert any(transcript in m["content"] for m in messages)

def test_summarize_uses_diarized_prompt_when_diarized():
    with patch("summarizer._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_groq_response("Summary")
        summarize("Speaker_00: Hello", diarized=True)
    messages = mock_client.chat.completions.create.call_args[1]["messages"]
    prompt = " ".join(m["content"] for m in messages)
    assert "SPEAKERS" in prompt
    assert "KEY HIGHLIGHTS" in prompt
    assert "OPEN QUESTIONS" in prompt

def test_summarize_uses_simple_prompt_when_not_diarized():
    with patch("summarizer._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_groq_response("Summary")
        summarize("Hello world", diarized=False)
    messages = mock_client.chat.completions.create.call_args[1]["messages"]
    prompt = " ".join(m["content"] for m in messages)
    assert "SPEAKERS" not in prompt

def test_summarize_defaults_to_simple_prompt():
    with patch("summarizer._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_groq_response("Summary")
        summarize("Hello world")
    messages = mock_client.chat.completions.create.call_args[1]["messages"]
    prompt = " ".join(m["content"] for m in messages)
    assert "SPEAKERS" not in prompt
```

**Step 2: Run tests to verify they fail**

```bash
cd /home/pi/meeting-notes && python -m pytest tests/test_summarizer.py -v
```

Expected: All 5 tests FAIL (wrong mock target).

**Step 3: Rewrite summarizer.py**

Replace entire content of `summarizer.py`:

```python
from groq import Groq

_client = Groq()

PROMPT_SIMPLE = """You are a meeting assistant. Analyze this transcript and provide:

1. SUMMARY (3-5 bullet points of main topics discussed)
2. ACTION ITEMS (person: task, or "None identified" if none)
3. KEY DECISIONS (or "None identified" if none)

Transcript:
{transcript}"""

PROMPT_DIARIZED = """You are a meeting assistant. Analyze this transcript and provide:

1. SUMMARY (3-5 bullets — include specific numbers, dates, names, and commitments)

2. SPEAKERS — for each speaker:
   - Apparent role/name if identifiable
   - Interaction type: decision-maker | facilitator | questioner | contributor | dissenter
   - 1-line characterization of their style

3. KEY HIGHLIGHTS — concrete moments worth noting (specific quotes, commitments,
   surprises, or disagreements — not just topic labels)

4. ACTION ITEMS — person: task, deadline if mentioned (or "None identified")

5. KEY DECISIONS — what was decided and the brief rationale behind it

6. OPEN QUESTIONS — unresolved issues or follow-ups with no clear owner

Transcript:
{transcript}"""


def summarize(transcript, model="llama-3.3-70b-versatile", diarized=False):
    template = PROMPT_DIARIZED if diarized else PROMPT_SIMPLE
    response = _client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": template.format(transcript=transcript)}]
    )
    return response.choices[0].message.content.strip()
```

**Step 4: Run tests to verify they pass**

```bash
cd /home/pi/meeting-notes && python -m pytest tests/test_summarizer.py -v
```

Expected: All 5 tests PASS.

**Step 5: Commit**

```bash
cd /home/pi/meeting-notes && git add summarizer.py tests/test_summarizer.py
git commit -m "feat: switch summarizer from ollama to Groq API"
```

---

### Task 2: Update jobs.py and app.py parameter name

**Files:**
- Modify: `jobs.py:81-82` (method signature)
- Modify: `app.py:63-64` (env var and kwarg)

**Step 1: Update jobs.py — rename `ollama_model` to `summary_model`**

In `jobs.py`, change the `process` method signature and its internal usage:

```python
# Line 81-82: change parameter name
def process(self, job_id, audio_path, transcript_dir, gmail_user,
            gmail_password, to_address, summary_model):
```

```python
# Line 99: change usage
summary = summarize(transcript, model=summary_model, diarized=diarized)
```

**Step 2: Update app.py — rename env var and kwarg**

In `app.py`, change two call sites (stop_recording and retry_job):

```python
# Both process_async calls: change kwarg name and env var
ollama_model=os.getenv("OLLAMA_MODEL", "llama3")
# becomes:
summary_model=os.getenv("GROQ_SUMMARY_MODEL", "llama-3.3-70b-versatile")
```

**Step 3: Run all tests**

```bash
cd /home/pi/meeting-notes && python -m pytest -v
```

Expected: All tests PASS.

**Step 4: Commit**

```bash
cd /home/pi/meeting-notes && git add jobs.py app.py
git commit -m "feat: rename ollama_model param to summary_model, use GROQ_SUMMARY_MODEL env var"
```

---

### Task 3: Remove ollama from requirements.txt

**Files:**
- Modify: `requirements.txt`

**Step 1: Remove the ollama line**

Delete the line `ollama` from `requirements.txt`. Result should be:

```
Flask==3.1.1
groq==1.0.0
python-dotenv==1.0.1
pytest
pytest-flask
```

**Step 2: Run all tests one final time**

```bash
cd /home/pi/meeting-notes && python -m pytest -v
```

Expected: All tests PASS.

**Step 3: Commit**

```bash
cd /home/pi/meeting-notes && git add requirements.txt
git commit -m "chore: remove ollama dependency"
```
