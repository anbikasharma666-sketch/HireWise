"""
src/llm.py
Ollama LLM setup and pre-flight connectivity check.
"""

import requests
from typing import Tuple, Dict, Any, Optional
from langchain_ollama import ChatOllama
from src.config import OLLAMA_BASE_URL, DEFAULT_LLM_MODEL

def check_ollama_status(base_url: str = OLLAMA_BASE_URL) -> Tuple[bool, str, list]:
    """
    Checks if the local Ollama daemon is reachable and lists available models.
    """
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("name") for m in data.get("models", [])]
            return True, "Ollama service is reachable", models
        return False, f"Ollama returned status code {resp.status_code}", []
    except Exception as e:
        return False, f"Could not connect to Ollama at {base_url}. Ensure 'ollama serve' is running.", []

def get_chat_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    base_url: str = OLLAMA_BASE_URL
) -> ChatOllama:
    """
    Returns an initialized ChatOllama instance.
    """
    target_model = model_name or DEFAULT_LLM_MODEL
    return ChatOllama(
        model=target_model,
        base_url=base_url,
        temperature=temperature
    )

def test_llm_invocation(model_name: Optional[str] = None) -> Tuple[bool, str]:
    """
    Performs a quick sanity test with ChatOllama.
    """
    try:
        llm = get_chat_llm(model_name=model_name)
        response = llm.invoke("Hi! Please confirm you are working with a brief response.")
        return True, str(response.content).strip()
    except Exception as e:
        return False, f"LLM test failed: {str(e)}"
