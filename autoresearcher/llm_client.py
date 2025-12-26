# llm_client.py
"""
LLM client utilities.

- Gemini (cloud) is used for multi-agent decision making.
- Ollama (local LLaMA) is used for PDF RAG.
- call_llm() is the backward-compatible wrapper used by rag_engine.py.
"""

import os
from typing import Optional
import requests
from dotenv import load_dotenv
import google.generativeai as genai

# ---------------------------------------------------------
# Load environment variables (.env + system env)
# ---------------------------------------------------------
load_dotenv()
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found. Put it in .env or system env.")

genai.configure(api_key=GEMINI_API_KEY)

# Recommended models
GEMINI_MODEL_NAME = "gemini-2.5-flash"   # safer, fewer quota issues
OLLAMA_MODEL_NAME = "llama3.1:latest"
OLLAMA_URL = "http://localhost:11434/api/generate"


# ---------------------------------------------------------
# GEMINI (Accepts 1 or 2 arguments)
# ---------------------------------------------------------
def call_gemini(*args) -> str:
    """
    Smart wrapper:
    - call_gemini(prompt)
    - call_gemini(system_prompt, user_prompt)

    Both work safely.
    """
    try:
        if len(args) == 1:
            # Single prompt mode
            full_prompt = str(args[0])

        elif len(args) == 2:
            # System + user
            system_prompt = str(args[0]).strip()
            user_prompt = str(args[1]).strip()
            full_prompt = f"{system_prompt}\n\nUSER:\n{user_prompt}"

        else:
            raise ValueError("call_gemini expects 1 or 2 arguments")

        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        resp = model.generate_content(full_prompt)

        # Newer SDK style
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()

        # Older SDK fallback
        if getattr(resp, "candidates", None):
            parts = resp.candidates[0].content.parts
            return "".join(p.text for p in parts if hasattr(p, "text")).strip()

        return str(resp)

    except Exception as e:
        return f"[GEMINI ERROR] {e}"


# ---------------------------------------------------------
# OLLAMA LOCAL MODEL
# ---------------------------------------------------------
def call_ollama(prompt: str, model: str = OLLAMA_MODEL_NAME) -> str:
    """Call local LLaMA via Ollama HTTP API."""
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=600,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()

    except Exception as e:
        return f"[OLLAMA ERROR] {e}"


# ---------------------------------------------------------
# BACKWARDS COMPATIBILITY FOR RAG ENGINE
# ---------------------------------------------------------
def call_llm(*args, **kwargs) -> str:
    """
    - rag_engine always uses call_llm()
    - This ALWAYS routes to Ollama, never Gemini
    """
    model = kwargs.pop("model", OLLAMA_MODEL_NAME)

    if len(args) == 1:
        full_prompt = str(args[0])

    elif len(args) >= 2:
        full_prompt = f"{args[0]}\n\n{args[1]}"

    else:
        raise ValueError("call_llm expects 1 or 2 positional arguments.")

    return call_ollama(full_prompt, model=model)
