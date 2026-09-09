"""
src/vectorstore.py
Chroma vector database management with 3 strictly separated collections:
1. hirewise_policy
2. hirewise_resume
3. hirewise_job_description
"""

import os
from typing import Dict, Any, Optional
from langchain_chroma import Chroma

from src.config import (
    CHROMA_DIR,
    COLLECTION_POLICY,
    COLLECTION_RESUME,
    COLLECTION_JD
)
from src.embeddings import get_embeddings
from src.loaders import (
    load_policy_documents,
    load_job_description_documents,
    load_resume_documents
)
from src.text_splitters import split_documents

def get_chroma_collection(collection_name: str, embedding_function=None) -> Chroma:
    """
    Returns an instance of Chroma for the specified collection name with local persistence.
    """
    embed_fn = embedding_function or get_embeddings()
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embed_fn,
        persist_directory=str(CHROMA_DIR)
    )

def is_vector_store_initialized() -> bool:
    """
    Checks whether all three collections exist and contain documents.
    """
    try:
        embed_fn = get_embeddings()
        for col_name in [COLLECTION_POLICY, COLLECTION_RESUME, COLLECTION_JD]:
            store = Chroma(
                collection_name=col_name,
                embedding_function=embed_fn,
                persist_directory=str(CHROMA_DIR)
            )
            # Count elements in collection
            count = len(store.get(limit=1).get("ids", []))
            if count == 0:
                return False
        return True
    except Exception:
        return False

def initialize_vector_store(force_reload: bool = False) -> Dict[str, int]:
    """
    Loads, splits, and embeds documents into 3 isolated Chroma collections.
    Skips if already persisted and force_reload is False.
    """
    if not force_reload and is_vector_store_initialized():
        counts = {}
        embed_fn = get_embeddings()
        for col_name in [COLLECTION_POLICY, COLLECTION_RESUME, COLLECTION_JD]:
            store = Chroma(
                collection_name=col_name,
                embedding_function=embed_fn,
                persist_directory=str(CHROMA_DIR)
            )
            data = store.get()
            counts[col_name] = len(data.get("ids", []))
        return counts

    embed_fn = get_embeddings()
    os.makedirs(CHROMA_DIR, exist_ok=True)
    results = {}

    # 1. Policy Collection
    policy_docs = load_policy_documents()
    policy_chunks = split_documents(policy_docs, chunk_size=500, chunk_overlap=80)
    policy_store = Chroma(
        collection_name=COLLECTION_POLICY,
        embedding_function=embed_fn,
        persist_directory=str(CHROMA_DIR)
    )
    # Clear existing documents if reloading
    if force_reload:
        try:
            policy_store.delete_collection()
            policy_store = Chroma(
                collection_name=COLLECTION_POLICY,
                embedding_function=embed_fn,
                persist_directory=str(CHROMA_DIR)
            )
        except Exception:
            pass
    policy_store.add_documents(policy_chunks)
    results[COLLECTION_POLICY] = len(policy_chunks)

    # 2. Resume Collection
    resume_docs = load_resume_documents()
    resume_chunks = split_documents(resume_docs, chunk_size=500, chunk_overlap=80)
    resume_store = Chroma(
        collection_name=COLLECTION_RESUME,
        embedding_function=embed_fn,
        persist_directory=str(CHROMA_DIR)
    )
    if force_reload:
        try:
            resume_store.delete_collection()
            resume_store = Chroma(
                collection_name=COLLECTION_RESUME,
                embedding_function=embed_fn,
                persist_directory=str(CHROMA_DIR)
            )
        except Exception:
            pass
    resume_store.add_documents(resume_chunks)
    results[COLLECTION_RESUME] = len(resume_chunks)

    # 3. Job Description Collection
    jd_docs = load_job_description_documents()
    jd_chunks = split_documents(jd_docs, chunk_size=500, chunk_overlap=80)
    jd_store = Chroma(
        collection_name=COLLECTION_JD,
        embedding_function=embed_fn,
        persist_directory=str(CHROMA_DIR)
    )
    if force_reload:
        try:
            jd_store.delete_collection()
            jd_store = Chroma(
                collection_name=COLLECTION_JD,
                embedding_function=embed_fn,
                persist_directory=str(CHROMA_DIR)
            )
        except Exception:
            pass
    jd_store.add_documents(jd_chunks)
    results[COLLECTION_JD] = len(jd_chunks)

    return results
