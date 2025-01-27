"""Utility class for transforming dictionaries to Haystack Document objects and vice versa."""

import logging

from haystack import Document

from dataloaders.haystack.utils.logging import LoggerFactory

logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()


class DocumentTransformer:
    """A utility class to handle transformations between Haystack Document objects and dictionaries."""

    @staticmethod
    def dict_to_documents(data: list[dict]) -> list[Document]:
        """Convert a list of dictionaries into Haystack Document objects.

        Args:
            data (list[dict]): A list of dictionaries, each containing 'text' and 'metadata'.

        Returns:
            list[HaystackDocument]: A list of transformed Haystack Document objects.

        Raises:
            KeyError: If any required keys ('text', 'metadata') are missing in the document.
            TypeError: If 'metadata' is not a dictionary.
        """
        docs = []
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

                # Convert to Haystack Document
                doc = Document(content=document["text"], meta=document["metadata"])
                docs.append(doc)
                logger.info("Successfully converted document to Haystack format.")

            except (KeyError, TypeError) as e:
                logger.error(f"Failed to convert document: {e}")
                raise e

        return docs

    @staticmethod
    def haystack_docs_to_dict(haystack_documents: list[Document]) -> list[dict]:
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
