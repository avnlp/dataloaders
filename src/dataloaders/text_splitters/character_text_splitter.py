import logging
from typing import Callable, Literal, Union

from langchain.text_splitter import CharacterTextSplitter as LangchainCharacterTextSplitter
from langchain_core.documents import Document

from dataloaders.utils import LoggerFactory

logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()


class CharacterTextSplitter:
    r"""A class for splitting and chunking documents into smaller text segments.

    This class extends the functionality of the Langchain `CharacterTextSplitter` by providing additional
    customization options and features for splitting text into manageable chunks. It supports parameters
    such as chunk size, overlap, separators, and more.

    Attributes:
        chunk_size (int): Maximum size of each chunk in characters. Defaults to 4000.
        chunk_overlap (int): Number of overlapping characters between adjacent chunks. Defaults to 200.
        length_function (Callable[[str], int]): Function used to compute the length of text. Defaults to `len`.
        add_start_index (bool): Whether to add a starting index to each chunk. Defaults to False.
        strip_whitespace (bool): Whether to strip leading/trailing whitespace from chunks. Defaults to True.
        separator (str): The separator to use for splitting the text. Defaults to double newline (`\n\n`).
        is_separator_regex (bool): Whether the separator is a regular expression. Defaults to False.
        keep_separator (Union[bool, Literal['start', 'end']]): Whether to keep the separator in the chunked text.
            Can be set to True to keep the separator in all chunks, or 'start'/'end' to only keep it at the
            start or end of chunks. Defaults to True.

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
        separator: str = "\n\n",
        is_separator_regex: bool = False,
        keep_separator: Union[bool, Literal["start", "end"]] = True,
    ):
        r"""Initialize the CharacterTextSplitter with the specified parameters.

        Args:
            chunk_size (int): The maximum size of each chunk in characters. Defaults to 4000.
            chunk_overlap (int): The number of overlapping characters between adjacent chunks. Defaults to 200.
            length_function (Callable[[str], int]): The function used to compute the length of text. Defaults to `len`.
            add_start_index (bool): Whether to add an index to each chunk. Defaults to False.
            strip_whitespace (bool): Whether to strip whitespace from the beginning and end of each chunk. Defaults to True.
            separator (str): The separator used to split the text. Defaults to `\n\n` (double newline).
            is_separator_regex (bool): Whether the separator is a regular expression. Defaults to False.
            keep_separator (Union[bool, Literal["start", "end"]]): Whether to keep the separator in the chunks.
            Defaults to True.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.add_start_index = add_start_index
        self.strip_whitespace = strip_whitespace
        self.separator = separator
        self.is_separator_regex = is_separator_regex
        self.keep_separator = keep_separator

        logger.info("Initializing CharacterTextSplitter with provided parameters.")

        # Initialize the LangChain CharacterTextSplitter with the parameters provided during initialization
        self.chunker = LangchainCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            add_start_index=self.add_start_index,
            strip_whitespace=self.strip_whitespace,
            separator=self.separator,
            is_separator_regex=self.is_separator_regex,
            keep_separator=self.keep_separator,
        )
        logger.info("CharacterTextSplitter initialized successfully.")

    def transform_documents(self, documents: list[Document]) -> list[Document]:
        """Split a list of LangChain Document objects into smaller chunks.

        This method uses the `chunker` (an instance of LangChain's `CharacterTextSplitter`) to break the
        documents into smaller chunks. Each chunk will adhere to the specified chunk size and overlap
        while preserving the content's integrity according to the separator and other options.

        Args:
            documents (List[Document]): List of LangChain Document objects to be split into smaller chunks.

        Returns:
            List[Document]: A list of chunked LangChain Document objects, each representing a smaller portion
                            of the original document.

        Raises:
            ValueError: If the chunker is not properly initialized.
            Exception: If there is an error during the document transformation process.
        """
        # Check if the chunker is initialized before processing
        if not self.chunker:
            msg = "Chunker is not initialized."
            logger.error(msg)
            raise ValueError(msg)

        # Log the number of documents being processed
        logger.info(f"Splitting {len(documents)} documents into smaller chunks.")

        try:
            # Chunk the documents using the CharacterTextSplitter
            transformed_documents = self.chunker.transform_documents(documents)
            logger.info(f"Successfully split documents into {len(transformed_documents)} chunks.")
        except Exception as e:
            # Log any exceptions during document chunking
            logger.error(f"Error during document chunking: {e}")
            raise

        return transformed_documents
