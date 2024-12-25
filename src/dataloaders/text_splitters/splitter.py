import logging
from typing import Optional, Union

from langchain_core.documents import Document

from dataloaders.text_splitters.character_text_splitter import CharacterTextSplitter
from dataloaders.text_splitters.recursive_character_splitter import RecursiveCharacterTextSplitter
from dataloaders.text_splitters.semantic_chunking import SemanticChunker
from dataloaders.text_splitters.unstructured_chunking import UnstructuredChunker
from dataloaders.utils import LoggerFactory

# Setup logging
logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()


class TextSplitter:
    """A class for splitting and chunking text into smaller segments for further processing.

    Attributes:
        splitter_name (str): The type of splitter to use. Defaults to "RecursiveCharacterTextSplitter".
        splitter_params (Optional[Dict[str, Union[str, int]]]): Parameters for the splitter (e.g., chunk size, overlap).
        splitter (Optional): The initialized splitter instance based on the splitter_name.
    """

    def __init__(
        self,
        splitter_name: str = "RecursiveCharacterTextSplitter",
        splitter_params: Optional[dict[str, Union[str, int]]] = None,
    ):
        """Initialize the Splitter with the specified type and parameters.

        Args:
            splitter_name (str): The type of splitter to use. Defaults to "RecursiveCharacterTextSplitter".
            splitter_params (Optional[Dict[str, Union[str, int]]]): Parameters for splitting the document.
        """
        self.splitter_name = splitter_name
        self.splitter_params = splitter_params or {}
        self.splitter = self._initialize_splitter()

    def _initialize_splitter(self):
        """Initialize the text splitter based on the specified splitter name and parameters.

        Returns:
            An initialized text splitter object based on the splitter name.

        Raises:
            NotImplementedError: If the specified splitter type is not recognized.
        """
        # Splitter type mapping
        splitter_mapping = {
            "CharacterTextSplitter": CharacterTextSplitter,
            "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter,
            "SemanticChunking": SemanticChunker,
            "UnstructuredChunker": UnstructuredChunker,
        }

        logger.info(f"Initializing splitter of type: {self.splitter_name}")

        # Check if the splitter name exists in the mapping
        if self.splitter_name not in splitter_mapping:
            err_msg = f"Splitter type '{self.splitter_name}' is not implemented."
            logger.error(err_msg)
            raise NotImplementedError(err_msg)

        # Initialize the splitter using the corresponding class and parameters
        splitter_class = splitter_mapping[self.splitter_name]
        logger.info(f"Using splitter class: {splitter_class.__name__}")
        return splitter_class(**self.splitter_params)

    def transform_documents(self, docs: list[Document]) -> list[Document]:
        """Split a list of documents into smaller chunks.

        The metadata for each chunked document will contain the original document's metadata.

        Args:
            docs (List[Document]): The list of LangChain Document objects to be split.

        Returns:
            List[Document]: A list of chunked LangChain Document objects.
        """
        # Ensure that the splitter is correctly initialized before processing
        if not self.splitter:
            msg = "TextSplitter has not been initialized properly."
            logger.error(msg)
            raise RuntimeError(msg)

        # Log the number of documents to be processed
        logger.info(f"Splitting {len(docs)} documents into smaller chunks.")

        # Split the documents into smaller chunks using the initialized splitter
        chunked_docs = self.splitter.transform_documents(docs)

        # Log the number of chunked documents
        logger.info(f"Successfully split documents into {len(chunked_docs)} chunks.")

        return chunked_docs
