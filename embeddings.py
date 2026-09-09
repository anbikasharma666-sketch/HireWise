"""
src/embeddings.py
OllamaEmbeddings setup with local model verification and fallback support.
"""

from typing import Optional
from langchain_ollama import OllamaEmbeddings
from src.config import DEFAULT_EMBEDDING_MODEL, FALLBACK_EMBEDDING_MODEL, OLLAMA_BASE_URL

def get_embeddings(
    model_name: Optional[str] = None,
    base_url: str = OLLAMA_BASE_URL
) -> OllamaEmbeddings:
    """
    Returns an initialized OllamaEmbeddings instance.
    Uses qwen3-embedding:0.6b by default, or the specified model.
    """
    target_model = model_name or DEFAULT_EMBEDDING_MODEL
    return OllamaEmbeddings(
        model=target_model,
        base_url=base_url
    )

def test_embeddings_connectivity(model_name: Optional[str] = None) -> bool:
    """
    Tests whether the embedding model produces vectors correctly.
    """
    try:
        embeddings = get_embeddings(model_name=model_name)
        vec = embeddings.embed_query("HireWise embeddings pre-flight test")
        return len(vec) > 0
    except Exception as e:
        print(f"[!] Embedding test failed: {e}")
        return False
