"""
src/retrievers.py
Dedicated retrievers for Policy, Resume, and Job Description with zero cross-contamination.
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import COLLECTION_POLICY, COLLECTION_RESUME, COLLECTION_JD
from src.vectorstore import get_chroma_collection

def get_policy_retriever(k: int = 3) -> VectorStoreRetriever:
    """
    Returns a retriever strictly constrained to the Company Handbook / Policy collection.
    Used ONLY for Candidate FAQ inquiries.
    """
    store = get_chroma_collection(COLLECTION_POLICY)
    return store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

def get_resume_retriever(candidate_id: Optional[str] = None, k: int = 3) -> VectorStoreRetriever:
    """
    Returns a retriever strictly constrained to Candidate Resumes.
    Used ONLY for Resume Screening.
    """
    store = get_chroma_collection(COLLECTION_RESUME)
    search_kwargs: Dict[str, Any] = {"k": k}
    if candidate_id:
        search_kwargs["filter"] = {"candidate": candidate_id}
        
    return store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )

def get_jd_retriever(k: int = 3) -> VectorStoreRetriever:
    """
    Returns a retriever strictly constrained to the Job Description collection.
    Used ONLY for Resume Screening comparisons.
    """
    store = get_chroma_collection(COLLECTION_JD)
    return store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
