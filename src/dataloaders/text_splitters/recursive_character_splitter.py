import logging
from typing import Callable, Literal, Optional, Union

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter as LangchainRecursiveCharacterTextSplitter

from dataloaders.utils import LoggerFactory

logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()


class RecursiveCharacterTextSplitter:
    r"""A class for recursively splitting and chunking documents into smaller text segments.

    This class extends Langchain's `RecursiveCharacterTextSplitter` to provide more customization options
    for splitting text into chunks. It recursively splits text using different separators, with overlap between
    chunks and options to keep the separator.

    Attributes:
        chunk_size (int): Maximum size of each chunk in characters. Defaults to 4000.
        chunk_overlap (int): Number of overlapping characters between adjacent chunks. Defaults to 200.
        length_function (Callable[[str], int]): Function used to compute the length of text. Defaults to `len`.
        add_start_index (bool): Whether to add an index to each chunk. Defaults to False.
        strip_whitespace (bool): Whether to strip leading/trailing whitespace from chunks. Defaults to True.
        separators (Optional[list[str]]): List of separator strings to split the text. Defaults to
        `["\n\n", "\n", " ", ""]`.
        keep_separator (Union[bool, Literal['start', 'end']]): Whether to keep the separator in the chunked text.
            Defaults to True.
        is_separator_regex (bool): Whether the separator is a regular expression. Defaults to False.

    Methods:
        transform_documents(documents: List[Document]) -> List[Document]:
            Splits a list of documents into smaller chunks based on the initialized parameters.
    """

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
        separators: Optional[list[str]] = None,
        keep_separator: Union[bool, Literal["start", "end"]] = True,
        is_separator_regex: bool = False,
    ):
        r"""Initialize the RecursiveCharacterTextSplitter with the specified parameters.

        Args:
            chunk_size (int): The maximum size of each chunk in characters. Defaults to 4000.
            chunk_overlap (int): The number of overlapping characters between adjacent chunks. Defaults to 200.
            length_function (Callable[[str], int]): The function used to compute the length of text. Defaults to `len`.
            add_start_index (bool): Whether to add an index to each chunk. Defaults to False.
            strip_whitespace (bool): Whether to strip whitespace from the beginning and end of each chunk.
            Defaults to True.
            separators (Optional[list[str]]): List of separator strings to split the text. Defaults to
            `["\n\n", "\n", " ", ""]`.
            keep_separator (Union[bool, Literal["start", "end"]]): Whether to keep the separator in the chunks.
            Defaults to True.
            is_separator_regex (bool): Whether the separator is a regular expression. Defaults to False.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.add_start_index = add_start_index
        self.strip_whitespace = strip_whitespace
        self.separators = separators
        self.keep_separator = keep_separator
        self.is_separator_regex = is_separator_regex

        logger.info("Initializing RecursiveCharacterTextSplitter with provided parameters.")

        # Initialize the LangChain RecursiveCharacterTextSplitter with the provided parameters
        self.chunker = LangchainRecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            add_start_index=self.add_start_index,
            strip_whitespace=self.strip_whitespace,
            separators=self.separators or ["\n\n", "\n", " ", ""],
            keep_separator=self.keep_separator,
            is_separator_regex=self.is_separator_regex,
        )
        logger.info("RecursiveCharacterTextSplitter initialized successfully.")

    def transform_documents(self, documents: list[Document]) -> list[Document]:
        """Split a list of LangChain Document objects into smaller chunks using the specified parameters.

        Args:
            documents (List[Document]): List of LangChain Document objects to be split into smaller chunks.

        Returns:
            List[Document]: A list of chunked LangChain Document objects, each representing a smaller portion
            of the original document.

        Raises:
            ValueError: If the chunker is not properly initialized.
            Exception: If there is an error during the document transformation process.
        """
        # Check if the chunker is initialized before proceeding
        if not self.chunker:
            msg = "Chunker is not initialized."
            logger.error(msg)
            raise ValueError(msg)

        # Log the number of documents being processed
        logger.info(f"Splitting {len(documents)} documents into smaller chunks.")

        try:
            # Perform the document chunking
            transformed_documents = self.chunker.transform_documents(documents)
            logger.info(f"Successfully split documents into {len(transformed_documents)} chunks.")
        except Exception as e:
            # Log any exceptions during document chunking
            logger.error(f"Error during document chunking: {e}")
            raise

        return transformed_documents
