import logging
from typing import Any, Optional, Union

import weave
from datasets import load_dataset
from haystack import Document
from haystack.components.preprocessors import RecursiveDocumentSplitter
from haystack.core.component import Component

from dataloaders.haystack.utils import DocumentTransformer, LoggerFactory

logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()


class EarningsCallDataloader:
    """A data loader class for processing and preparing financial document data from the Earnings Call dataset.

    This class handles loading, parsing, and structuring financial documents and their metadata from the Earnings Call
    dataset. It processes document names to extract key components (company, year, quarter, etc.), splits documents
    into manageable chunks, and prepares data for downstream tasks like retrieval-augmented QA systems.

    Attributes:
        dataset (Dataset): The loaded Earnings Call dataset split (e.g., "test").
        data (Optional[list[dict]]): Cached processed data containing questions, answers, and document metadata.
        corpus (list[dict]): Processed and chunked documents with metadata, ready for pipeline integration.
    """

    def __init__(
        self,
        dataset_name: str = "lamini/earnings-calls-qa",
        split: str = "train",
        text_splitter: Component = RecursiveDocumentSplitter(),
    ):
        """Initialize the EarningsCallDataloader with the specified dataset.

        Args:
            dataset_name (str): Name of the dataset to load. Defaults to "lamini/earnings-calls-qa".
            split (str): Split of the dataset to load (e.g., "test", "train"). Defaults to "train".
            text_splitter (Component): The text splitter to use for processing documents. Defaults to
                RecursiveDocumentSplitter.
        """
        self.dataset = load_dataset(dataset_name, split=split)
        self.corpus_dataset = load_dataset(dataset_name, "corpus", split=split)
        self.data: Optional[list[dict[str, Union[str, list[str], list[dict[str, str]]]]]] = None
        self.corpus: list[dict[str, Any]] = []
        self.text_splitter = text_splitter

    def load_data(self) -> list[dict[str, Union[str, list[str], list[dict[str, str]]]]]:
        """Load and transform the FinanceBench dataset into processed, structured format.

        Main processing method that:
        1. Extracts key financial QA pairs
        2. Processes supporting evidence documents
        3. Parses document metadata
        4. Applies text splitting/chunking
        5. Caches results for subsequent calls

        Returns:
            list[dict]: Processed dataset ready for use in retrieval pipelines, each entry containing:
                - text (str): Financial question
                - metadata (dict): Answer, evidence, document context, and parsed metadata
        """
        if self.data is None:
            logger.info("Loading and processing dataset.")
            self.data = []
            for row in self.dataset:
                question = row["question"]
                answer = row["answer"]
                date = row["date"]
                context = row["transcript"]
                year_quarter = row["q"]
                year, quarter = row["q"].split("-")
                ticker = row["ticker"]

                # Append processed data
                self.data.append(
                    {
                        "question": question,
                        "answer": answer,
                        "date": date,
                        "context": context,
                        "year": year,
                        "quarter": quarter,
                        "ticker": ticker,
                        "year-quarter": year_quarter,
                    }
                )
            logger.info(f"Processed {len(self.data)} rows from the dataset.")

        return self.data

    def _preprocess_docs(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Prepare documents for processing pipelines through chunking and normalization.

        Implements a complete preprocessing pipeline:
        1. Converts raw dictionaries to Haystack Document objects
        2. Splits documents using configured text splitter
        3. Maintains metadata across splits
        4. Converts back to dictionary format

        Args:
            data (list[dict]): Unprocessed documents with text and metadata

        Returns:
            list[dict]: Processed document chunks with preserved metadata

        Raises:
            ValueError: If documents lack required fields or have invalid structure
        """
        # Format documents for the splitter (convert dictionary to Haystack Documents)
        data = DocumentTransformer.dict_to_documents(data=data)

        # Apply the splitter to chunk the documents
        transformed_docs = self.text_splitter.run(data)["documents"]

        # Convert chunked documents back into dictionary format
        transformed_data = DocumentTransformer.haystack_docs_to_dict(transformed_docs)

        return transformed_data

    def get_corpus(self):
        """Load and transform the FinanceBench dataset into processed, structured format.

        Main processing method that:
        1. Extracts key financial QA pairs
        2. Processes supporting evidence documents
        3. Parses document metadata
        4. Applies text splitting/chunking
        5. Caches results for subsequent calls

        Returns:
            list[dict]: Processed dataset ready for use in retrieval pipelines, each entry containing:
                - text (str): Financial question
                - metadata (dict): Answer, evidence, document context, and parsed metadata
        """
        logger.info("Loading and processing dataset.")
        self.corpus = []
        for row in self.corpus_dataset:
            ticker = row["symbol"]
            year = row["year"]
            quarter = f"Q{row['quarter']}"
            date = row["date"]
            for doc in row["transcript"]:
                speaker = doc["speaker"]
                text = doc["text"]

                # Append processed data
                self.corpus.append(
                    {
                        "text": f"{speaker}: {text}",
                        "metadata": {
                            "ticker": ticker,
                            "year": year,
                            "quarter": quarter,
                            "date": date,
                            "speaker": speaker,
                        },
                    }
                )
        logger.info(f"Processed {len(self.corpus)} rows from the dataset.")

        return self.corpus

    def get_documents(self) -> list[Document]:
        """Convert the processed corpus to Haystack Document objects.

        This method converts the `self.corpus`, which contains the processed text and metadata, into Haystack Document objects
        for use in Haystack-based retrieval pipelines. It ensures that the data is loaded before processing.

        Returns:
            list[Document]: A list of Haystack Document objects, where each document represents a chunked text
            from the corpus with corresponding metadata.

        Raises:
            ValueError: If `self.corpus` is empty (i.e., the data has not been loaded).
        """
        if not self.corpus:
            self.get_corpus()

        # Convert the corpus to Haystack documents
        haystack_docs = DocumentTransformer.dict_to_documents(data=self.corpus)

        return haystack_docs

    def get_questions(self) -> list[str]:
        """Extract all the questions from the dataset.

        This method retrieves all questions from the loaded dataset. If the data has not been loaded,
        it will call the `load_data` method to load the necessary information before extracting the questions.

        Returns:
            list[str]:
                A list of strings where each string is a question extracted from the dataset.

        Raises:
            ValueError: If the dataset is empty or not properly loaded.
        """
        if self.data is None:
            self.load_data()

        if not self.data:
            msg = "No data available to extract questions. Ensure `load_data` has been called and contains valid data."
            logger.error(msg)
            raise ValueError(msg)

        questions = [row["question"] for row in self.data]
        return questions

    def get_evaluation_data(self) -> list[dict[str, Any]]:
        """Prepare and format data for evaluation.

        This method structures the data for evaluation purposes. Each evaluation instance consists of:
        - `question`: The question to be evaluated.
        - `answer`: The expected answer.
        - `context`: The supporting evidence context.
        - `year`: The fiscal year.
        - `quarter`: The fiscal quarter.
        - `ticker`: The company ticker symbol.
        - `year-quarter`: The fiscal year and quarter combined.

        If no data has been loaded, the method automatically calls `load_data` to load the necessary information.

        Returns:
            list[dict[str, Any]]:
                A list of dictionaries, each representing an evaluation instance with the following keys:
                - "question" (str): The question text.
                - "answer" (str): The expected answer text.
                - "context" (str): The supporting evidence context.
                - "year" (str): The fiscal year.
                - "quarter" (str): The fiscal quarter.
                - "ticker" (str): The company ticker symbol.
                - "year-quarter" (str): The fiscal year and quarter combined.
        """
        if self.data is None:
            self.load_data()

        if not self.data:
            msg = "No data available for evaluation. Ensure `load_data` has been called and contains valid data."
            logger.error(msg)
            raise ValueError(msg)

        # Prepare the evaluation data
        self.eval_data = [
            {
                "input": row["question"],
                "expected_output": row["answer"],
                "context": [row["context"]],
                "year": row["year"],
                "quarter": row["quarter"],
                "ticker": row["ticker"],
                "year-quarter": row["year-quarter"],
            }
            for row in self.data
        ]

        logger.info(f"Prepared {len(self.eval_data)} rows for evaluation.")
        return self.eval_data

    def publish_to_weave(
        self, weave_project_name: str, dataset_name: str, evaluation_dataset_name: Optional[str] = None
    ) -> None:
        """Publish processed and evaluation data to Weave.

        This method initializes a Weave project and publishes both the processed data and, if available,
        the evaluation data to Weave. If the dataset has not been loaded, it will call the `load_data` method
        before publishing the data.

        Args:
            weave_project_name (str): The name of the Weave project to which data will be published.
            dataset_name (str): The name of the dataset for the processed data.
            evaluation_dataset_name (Optional[str]): The name of the dataset for evaluation data. Defaults to None.

        Raises:
            ValueError: If the dataset is empty or improperly loaded when attempting to publish.
        """
        weave.init(project_name=weave_project_name)
        logger.info(f"Initializing Weave project: {weave_project_name}")

        # Load data if not already loaded
        if self.eval_data is None:
            self.get_evaluation_data()

        # Publish dataset
        logger.info(f"Publishing {len(self.eval_data)} rows to Weave dataset.")
        weave.publish(weave.Dataset(name=dataset_name, rows=self.eval_data))

        if evaluation_dataset_name:
            # Publish evaluation dataset
            evaluation_data = self.get_evaluation_data()
            logger.info(f"Publishing {len(evaluation_data)} rows to Weave evaluation dataset.")
            weave.publish(weave.Dataset(name=evaluation_dataset_name, rows=evaluation_data))
