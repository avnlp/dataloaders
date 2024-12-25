"""Utility class for transforming documents between LangChain and Haystack formats.

This module provides the `DocumentTransformer` class with the following methods:
- dict_to_documents: Converts a list of dictionaries into LangChain or Haystack Document objects.
- langchain_docs_to_dict: Converts a list of LangChain Document objects into dictionaries.
- haystack_docs_to_dict: Converts a list of Haystack Document objects into dictionaries.

Each method ensures proper validation of document structure before transformation.
"""

import logging
from typing import Literal

from haystack import Document as HaystackDocument
from langchain_core.documents import Document as LangchainDocument

from dataloaders.utils.logging import LoggerFactory

logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()


class DocumentTransformer:
    """A utility class to handle transformations between different document formats.

    Provides methods to convert between:
    - LangChain Document objects
    - Haystack Document objects
    - Standard dictionary format with 'text' and 'metadata'.
    """

    @staticmethod
    def dict_to_documents(data: list[dict], format_type: Literal["langchain", "haystack"]) -> list:
        """Convert a list of dictionaries into either LangChain or Haystack Document objects.

        Args:
            data (list[dict]): A list of dictionaries, each containing 'text' and 'metadata'.
            format_type (Literal['langchain', 'haystack']): The format type to convert to ('langchain' or 'haystack').

        Returns:
            list: A list of transformed document objects (either LangChain or Haystack).

        Raises:
            ValueError: If an unsupported format type is provided.
            KeyError: If any required keys ('text', 'metadata') are missing in the document.
            TypeError: If 'metadata' is not a dictionary.
        """
        format_types = {
            "langchain": LangchainDocument,
            "haystack": HaystackDocument,
        }

        if format_type not in format_types:
            msg = "Unsupported format type. Choose 'langchain' or 'haystack'."
            raise ValueError(msg)

        docs = []
        document_class = format_types[format_type]

        for document in data:
            try:
                # Validate that the document contains 'text' and 'metadata'
                missing_keys = [key for key in ["text", "metadata"] if key not in document]
                if missing_keys:
                    msg = f"Missing keys in document: {missing_keys}"
                    raise KeyError(msg)

                # Validate that 'metadata' is a dictionary
                if not isinstance(document["metadata"], dict):
                    msg = "Metadata must be a dictionary."
                    raise TypeError(msg)

                # Convert to the desired document format
                if format_type == "langchain":
                    doc = document_class(page_content=document["text"], metadata=document["metadata"])
                else:
                    doc = document_class(content=document["text"], meta=document["metadata"])

                docs.append(doc)
                logger.info(f"Successfully converted document to {format_type} format.")

            except (KeyError, TypeError) as e:
                logger.error(f"Failed to convert document: {e}")
                raise e

        return docs

    @staticmethod
    def langchain_docs_to_dict(langchain_documents: list[LangchainDocument]) -> list[dict]:
        """Convert a list of LangChain Document objects into a list of dictionaries.

        Args:
            langchain_documents (list[LangchainDocument]): A list of LangChain Document objects.

        Returns:
            list[dict]: A list of dictionaries where each dictionary contains:
                - 'text': The content of the document.
                - 'metadata': Metadata associated with the document.
        """
        data = [{"text": document.page_content, "metadata": document.metadata} for document in langchain_documents]
        logger.info(f"Converted {len(langchain_documents)} LangChain documents to dictionaries.")

        return data

    @staticmethod
    def haystack_docs_to_dict(haystack_documents: list[HaystackDocument]) -> list[dict]:
        """Convert a list of Haystack Document objects into a list of dictionaries.

        Args:
            haystack_documents (list[HaystackDocument]): A list of Haystack Document objects.

        Returns:
            list[dict]: A list of dictionaries where each dictionary contains:
                - 'text': The content of the document.
                - 'metadata': Metadata associated with the document.
        """
        data = [{"text": document.content, "metadata": document.meta} for document in haystack_documents]
        logger.info(f"Converted {len(haystack_documents)} Haystack documents to dictionaries.")

        return data
