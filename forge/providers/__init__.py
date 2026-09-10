"""
forge.providers — concrete LLM provider adapters.

Each module here wraps exactly one provider SDK and implements the
``forge.llm.LLMProvider`` interface.  Nothing outside this package imports a
provider SDK; the rest of FORGE depends only on ``forge.llm`` abstractions.

Step 2 ships:
- ``gemini`` — Google Gemini via the ``google-genai`` SDK.
"""
