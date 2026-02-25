import os
import pytest

# Provide required env vars so app.py's startup validation passes when imported by tests
os.environ.setdefault("GMAIL_USER", "test@example.com")
os.environ.setdefault("GMAIL_APP_PASSWORD", "test-password")
os.environ.setdefault("GMAIL_TO", "test@example.com")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
