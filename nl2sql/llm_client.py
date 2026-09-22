"""
Thin LLM client wrapping Google Gemini via the `google-genai` SDK (the
current SDK; the older `google-generativeai` package is deprecated).

Reads config from environment variables:

  GEMINI_API_KEY   (required)
    NL2SQL_MODEL     (optional, defaults to "gemini-3.7-flash")
"""
import os


DEFAULT_MODEL = "gemini-3.7-flash"


class LLMClient:
    def __init__(self, model: str | None = None, **_ignored):
        # **_ignored keeps backward compatibility with any old code that
        # still passes provider=... ; it's simply not used anymore.
        self.model = model or os.getenv("NL2SQL_MODEL", DEFAULT_MODEL)

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        """Returns the raw text completion from Gemini."""
        from google import genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Export it before running, "
                "or enable Demo mode in the UI."
            )

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
            ),
        )
        return resp.text
