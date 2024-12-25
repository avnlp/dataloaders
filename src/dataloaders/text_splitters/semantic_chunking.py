import logging
from typing import Literal, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker as LangchainSemanticChunker

from dataloaders.utils import LoggerFactory

logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()

# Define acceptable threshold types for determining chunk breakpoints
BREAKPOINT_THRESHOLD_TYPE = Literal["percentile", "standard_deviation", "interquartile", "gradient"]


class SemanticChunker:
    r"""A class for performing semantic chunking on a list of documents using an embedding model.

    This class leverages the provided embedding model to calculate semantic similarity between text segments.
    The chunk boundaries are determined based on the semantic similarity between consecutive texts.
    There is also support for different chunking strategies.

    Attributes:
        embeddings (Embeddings): An embedding model used to calculate semantic similarity between text segments.
        buffer_size (int): Size of the buffer around each chunk to add context from neighboring chunks. Defaults to 1.
        add_start_index (bool): Whether to include the start index of each chunk in metadata. Defaults to False.
        breakpoint_threshold_type (BREAKPOINT_THRESHOLD_TYPE): Method to determine chunk breakpoints.
        Defaults to "percentile".
        breakpoint_threshold_amount (Optional[float]): Threshold value used to calculate breakpoints. Defaults to None.
        number_of_chunks (Optional[int]): Desired number of chunks to generate per document. Defaults to None.
        sentence_split_regex (str): Regular expression for sentence splitting within the text.
        Defaults to r"(?<=[.?!])\s+".
        min_chunk_size (Optional[int]): Minimum token count required for each chunk. Defaults to None.

    Methods:
        transform_documents(documents: List[Document]) -> List[Document]:
            Executes semantic chunking on the provided documents using the configured chunker.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        buffer_size: int = 1,
        add_start_index: bool = False,
        breakpoint_threshold_type: BREAKPOINT_THRESHOLD_TYPE = "percentile",
        breakpoint_threshold_amount: Optional[float] = None,
        number_of_chunks: Optional[int] = None,
        sentence_split_regex: str = r"(?<=[.?!])\s+",
        min_chunk_size: Optional[int] = None,
    ):
        r"""Initialize the SemanticChunker with specified parameters for chunking.

        Args:
            embeddings (Embeddings): The embedding model used to calculate semantic breakpoints.
            buffer_size (int, optional): The size of the buffer around each chunk to add context from neighboring
            chunks. Defaults to 1.
            add_start_index (bool, optional): Whether to include the start index in chunk metadata. Defaults to False.
            breakpoint_threshold_type (BREAKPOINT_THRESHOLD_TYPE, optional): Type of threshold for determining chunk
            breakpoints. Defaults to "percentile".
            breakpoint_threshold_amount (Optional[float], optional): The threshold amount for breaking based on the
            selected method. Defaults to None.
            number_of_chunks (Optional[int], optional): The desired number of chunks for each document.
            Defaults to None.
            sentence_split_regex (str, optional): The regex pattern for sentence splitting.
            Defaults to r"(?<=[.?!])\s+".
            min_chunk_size (Optional[int], optional): The minimum size for each chunk in terms of token count.
            Defaults to None.
        """
        self.embeddings = embeddings
        self.buffer_size = buffer_size
        self.add_start_index = add_start_index
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.number_of_chunks = number_of_chunks
        self.sentence_split_regex = sentence_split_regex
        self.min_chunk_size = min_chunk_size

        logger.info("Initializing SemanticChunker with provided parameters.")

        # Initialize the LangChain SemanticChunker with the given parameters
        self.chunker = LangchainSemanticChunker(
            embeddings=self.embeddings,
            buffer_size=self.buffer_size,
            add_start_index=self.add_start_index,
            breakpoint_threshold_type=self.breakpoint_threshold_type,
            breakpoint_threshold_amount=self.breakpoint_threshold_amount,
            number_of_chunks=self.number_of_chunks,
            sentence_split_regex=self.sentence_split_regex,
            min_chunk_size=self.min_chunk_size,
        )
        logger.info("SemanticChunker initialized successfully.")

    def transform_documents(self, documents: list[Document]) -> list[Document]:
        """Perform semantic chunking on the provided documents.

        This method uses the configured chunker to split documents into semantically meaningful chunks,
        determining the chunk boundaries based on semantic similarity, and possibly applying buffer sizes.

        Args:
            documents (List[Document]): A list of `Document` objects to be processed and split into chunks.

        Returns:
            List[Document]: A list of `Document` objects, each containing semantically chunked sections.

        Raises:
            ValueError: If no documents are provided for chunking.
            Exception: If there is an error during the chunking process.
        """
        # Ensure that documents are provided for transformation
        if not documents:
            msg = "No documents provided for transformation."
            logger.error(msg)
            raise ValueError(msg)

        logger.info(f"Starting semantic chunking for {len(documents)} documents.")

        try:
            # Perform semantic chunking using the chunker
            transformed_documents = self.chunker.transform_documents(documents)
            logger.info(f"Successfully chunked {len(transformed_documents)} documents.")
        except Exception as e:
            # Log any exceptions that occur during chunking
            logger.error(f"Error during document chunking: {e}")
            raise

        return transformed_documents
