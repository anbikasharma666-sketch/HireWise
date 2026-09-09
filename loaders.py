"""
src/loaders.py
Document loading using PyPDFLoader with strict metadata tagging to prevent data bleed.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from src.config import HANDBOOK_PDF, JD_PDF, RESUME_PDF

def load_policy_documents(pdf_path: Path = HANDBOOK_PDF) -> List[Document]:
    """
    Loads company handbook PDF and attaches strict policy metadata.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Handbook PDF not found at {pdf_path}. Run generate_data.py first.")
    
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    for doc in docs:
        doc.metadata["document_type"] = "policy"
        doc.metadata["source"] = "handbook.pdf"
        doc.metadata["company"] = "HireWise Technologies"
    return docs

def load_job_description_documents(pdf_path: Path = JD_PDF) -> List[Document]:
    """
    Loads Job Description PDF and attaches job_description metadata.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Job Description PDF not found at {pdf_path}. Run generate_data.py first.")
    
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    for doc in docs:
        doc.metadata["document_type"] = "job_description"
        doc.metadata["source"] = "job_description.pdf"
        doc.metadata["role"] = "Software Engineer"
    return docs

def load_resume_documents(pdf_path: Path = RESUME_PDF, candidate_id: str = "candidate_1") -> List[Document]:
    """
    Loads Candidate Resume PDF and attaches resume metadata with candidate ID.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Resume PDF not found at {pdf_path}. Run generate_data.py first.")
    
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    for doc in docs:
        doc.metadata["document_type"] = "resume"
        doc.metadata["candidate"] = candidate_id
        doc.metadata["source"] = os.path.basename(str(pdf_path))
    return docs
