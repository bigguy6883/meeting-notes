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
