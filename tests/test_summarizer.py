from unittest.mock import patch, MagicMock
from summarizer import summarize

def test_summarize_returns_string():
    mock_response = MagicMock()
    mock_response.message.content = "- Discussed roadmap\nAction: Alice to follow up"
    with patch("summarizer.ollama.chat", return_value=mock_response):
        result = summarize("This is a transcript about the Q1 roadmap.", model="llama3")
    assert isinstance(result, str)
    assert len(result) > 0

def test_summarize_sends_transcript_in_prompt():
    mock_response = MagicMock()
    mock_response.message.content = "Summary here"
    transcript = "We talked about deadlines and deliverables."
    with patch("summarizer.ollama.chat") as mock_chat:
        mock_chat.return_value = mock_response
        summarize(transcript, model="llama3")
    call_args = mock_chat.call_args
    messages = call_args[1]["messages"]
    assert any(transcript in m["content"] for m in messages)

def test_summarize_uses_diarized_prompt_when_diarized():
    mock_response = MagicMock()
    mock_response.message.content = "Summary"
    with patch("summarizer.ollama.chat") as mock_chat:
        mock_chat.return_value = mock_response
        summarize("Speaker_00: Hello", model="llama3", diarized=True)
    messages = mock_chat.call_args[1]["messages"]
    prompt = " ".join(m["content"] for m in messages)
    assert "SPEAKERS" in prompt
    assert "KEY HIGHLIGHTS" in prompt
    assert "OPEN QUESTIONS" in prompt

def test_summarize_uses_simple_prompt_when_not_diarized():
    mock_response = MagicMock()
    mock_response.message.content = "Summary"
    with patch("summarizer.ollama.chat") as mock_chat:
        mock_chat.return_value = mock_response
        summarize("Hello world", model="llama3", diarized=False)
    messages = mock_chat.call_args[1]["messages"]
    prompt = " ".join(m["content"] for m in messages)
    assert "SPEAKERS" not in prompt

def test_summarize_defaults_to_simple_prompt():
    mock_response = MagicMock()
    mock_response.message.content = "Summary"
    with patch("summarizer.ollama.chat") as mock_chat:
        mock_chat.return_value = mock_response
        summarize("Hello world", model="llama3")
    messages = mock_chat.call_args[1]["messages"]
    prompt = " ".join(m["content"] for m in messages)
    assert "SPEAKERS" not in prompt
