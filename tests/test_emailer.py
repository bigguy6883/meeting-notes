import os
import pytest
from unittest.mock import patch, MagicMock
from emailer import send_notes

def test_send_notes_connects_to_gmail(tmp_path):
    transcript_file = tmp_path / "transcript.txt"
    transcript_file.write_text("Full meeting transcript here.")
    with patch("emailer.smtplib.SMTP") as mock_smtp_class:
        mock_smtp = MagicMock()
        mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)
        send_notes(
            gmail_user="test@gmail.com",
            gmail_password="app-password",
            to_address="test@gmail.com",
            meeting_label="2026-02-18 10:30",
            summary="- Discussed roadmap",
            transcript_path=str(transcript_file)
        )
    mock_smtp_class.assert_called_with("smtp.gmail.com", 587)

def test_send_notes_attaches_transcript(tmp_path):
    transcript_file = tmp_path / "transcript.txt"
    transcript_file.write_text("Full transcript content.")
    sent_messages = []
    with patch("emailer.smtplib.SMTP") as mock_smtp_class:
        mock_smtp = MagicMock()
        mock_smtp.send_message.side_effect = lambda msg: sent_messages.append(msg)
        mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)
        send_notes(
            gmail_user="test@gmail.com",
            gmail_password="app-password",
            to_address="test@gmail.com",
            meeting_label="2026-02-18 10:30",
            summary="Summary here",
            transcript_path=str(transcript_file)
        )
    assert len(sent_messages) == 1
    msg = sent_messages[0]
    assert "Meeting Notes" in msg["Subject"]
    payloads = msg.get_payload()
    assert len(payloads) == 2  # body + attachment
