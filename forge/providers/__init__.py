"""
forge.providers — concrete LLM provider adapters.

Each module here wraps exactly one provider API and implements the
``forge.llm.LLMProvider`` interface.  Nothing outside this package imports a
provider SDK; the rest of FORGE depends only on ``forge.llm`` abstractions.

Adapters:
- ``openai_compat`` — any OpenAI-compatible chat-completions endpoint
  (standard library only). Presets: ``opencode_go`` (default, OpenCode Go)
  and ``openai_compatible`` (any base URL, e.g. a local Ollama server).
- ``fallback`` — provider-agnostic ordered fallback over other adapters.
"""
