"""Text splitter utilities for document chunking and processing.

This module provides a collection of text splitters for various chunking strategies, enabling efficient processing of documents for downstream tasks like embedding and retrieval.

Exported classes:
- CharacterTextSplitter: A text splitter that divides text into chunks based on character count.
- RecursiveCharacterTextSplitter: A text splitter that recursively divides text by character count while respecting logical boundaries.
- SemanticChunker: A semantic-based splitter leveraging embeddings for meaningful chunking.
- TextSplitter: A base class for implementing custom text splitting strategies.
- UnstructuredChunker: A splitter utilizing Unstructured.io for document segmentation.
"""

from dataloaders.text_splitters.character_text_splitter import CharacterTextSplitter
from dataloaders.text_splitters.recursive_character_splitter import RecursiveCharacterTextSplitter
from dataloaders.text_splitters.semantic_chunking import SemanticChunker
from dataloaders.text_splitters.splitter import TextSplitter
from dataloaders.text_splitters.unstructured_chunking import UnstructuredChunker

__all__ = [
    "CharacterTextSplitter",
    "RecursiveCharacterTextSplitter",
    "SemanticChunker",
    "TextSplitter",
    "UnstructuredChunker",
]
