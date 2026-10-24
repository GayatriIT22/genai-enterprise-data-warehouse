"""Step 1 for the GenAI owner: check that the API key works.   python -m genai.test_api"""
import os

from dotenv import load_dotenv

from config import ROOT

load_dotenv(ROOT / ".env")
import anthropic  # noqa: E402

client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY from .env
reply = client.messages.create(
    model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"),
    max_tokens=100,
    messages=[{"role": "user", "content": "Say hello in one line."}],
)
print(reply.content[0].text)
