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
