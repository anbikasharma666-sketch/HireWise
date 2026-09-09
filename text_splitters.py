"""
src/text_splitters.py
Text splitting using RecursiveCharacterTextSplitter while maintaining metadata boundaries.
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_documents(
    documents: List[Document],
    chunk_size: int = 600,
    chunk_overlap: int = 80
) -> List[Document]:
    """
    Splits documents into clean chunks while preserving document metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "•", ".", " ", ""],
        strip_whitespace=True
    )
    return text_splitter.split_documents(documents)
