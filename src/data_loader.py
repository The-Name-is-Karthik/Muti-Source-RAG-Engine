"""
A universal data loader for the RAG bot.

This module provides functions to load and extract text from various sources,
including web pages, PDFs, and DOCX files, using LangChain's Document Loaders.
Each function returns a list of Document objects, ready for indexing.
"""

import logging
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, Docx2txtLoader
from langchain.schema.document import Document
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_from_webpage(url: str) -> List[Document]:
    """Loads text from a web page."""
    logging.info(f"Loading content from URL: {url}")
    loader = WebBaseLoader(url)
    return loader.load()

def load_from_pdf(file_path: str) -> List[Document]:
    """Loads text from a PDF file."""
    logging.info(f"Loading content from PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

def load_from_docx(file_path: str) -> List[Document]:
    """Loads text from a DOCX file."""
    logging.info(f"Loading content from DOCX: {file_path}")
    loader = Docx2txtLoader(file_path)
    return loader.load()