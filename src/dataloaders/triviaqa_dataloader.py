import logging
from typing import Any, Optional, Union

import weave
from datasets import load_dataset
from haystack import Document as HaystackDocument
from langchain_core.documents import Document as LangchainDocument

from dataloaders.llms.groq import ChatGroqGenerator
from dataloaders.prompts.summarize_answers import SummarizeAnswersPrompt
from dataloaders.text_splitters import TextSplitter
from dataloaders.utils import DocumentTransformer, LoggerFactory

logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()


class TriviaQADataloader:
    """A data loader class for the TriviaQA dataset that processes questions, contexts, and answers.

    It integrates with a ChatGroqGenerator to summarize answers and prepares data for evaluation
    and use in downstream tasks.

    Attributes:
        dataset: The loaded TriviaQA dataset from the Hugging Face Hub.
        answer_summary_generator (ChatGroqGenerator): Generator to summarize answers using LLM.
        data (List[Dict[str, Union[str, List[str], List[Dict[str, str]]]]]): Cached processed data.
    """

    def __init__(
        self,
        answer_summary_generator: ChatGroqGenerator,
        dataset_name: str = "awinml/triviaqa",
        split: str = "test",
        text_splitter: str = "RecursiveCharacterTextSplitter",
        text_splitter_params: Optional[dict[str, Union[str, int]]] = None,
    ):
        """Initialize the TriviaQADataloader with the specified dataset and answer summarizer.

        Args:
            answer_summary_generator (ChatGroqGenerator): Instance of the ChatGroqGenerator for answer summarization.
            dataset_name (str): Name of the dataset to load. Defaults to "awinml/popqa_longtail".
            split (str): Split of the dataset to load (e.g., "test", "train"). Defaults to "test".
            text_splitter (str): The name of the text splitter to use for processing documents.
                Defaults to "RecursiveCharacterTextSplitter".
            text_splitter_params (Optional[dict[str, Union[str, int]]]): A dictionary of parameters to configure
                the text splitter. Defaults to None.

        Attributes:
            dataset (Any): The loaded dataset split as specified during initialization.
            data (Optional[list[dict[str, Union[str, list[str], list[dict[str, str]]]]]]): Cached processed data
                for the ARC dataset.
            text_splitter (str): The name of the text splitter used for splitting text.
            text_splitter_params (Optional[dict[str, Union[str, int]]]): Parameters for configuring the text splitter.
            corpus (list[dict[str, Any]]): A list of chunked documents with metadata processed from the dataset.
        """
        self.dataset = load_dataset(dataset_name, split=split)
        self.answer_summary_generator = answer_summary_generator
        self.data: Optional[list[dict[str, Union[str, list[str], list[dict[str, str]]]]]] = None
        self.text_splitter = text_splitter
        self.text_splitter_params = text_splitter_params
        self.corpus: list[dict[str, Any]] = []

    def _summarize_answers(self, question: str, answers: list[str]) -> str:
        """Summarizes the given answers to a question using the ChatGroqGenerator.

        Args:
            question (str): The question for which answers need to be summarized.
            answers (List[str]): List of possible answers.

        Returns:
            str: Summarized answer or an error message if the process fails.

        Example:
            >>> question = "What is the capital of France?"
            >>> answers = ["Paris", "PARIS", "Paris."]
            >>> loader._summarize_answers(question, answers)
            "Paris"
        """
        try:
            prompt = SummarizeAnswersPrompt.format(question=question, answers=answers)
            return self.answer_summary_generator.predict(user_prompts=[prompt])
        except Exception:
            return "Error generating answer!"

    def _extract_contexts(self, contexts: list[dict[str, str]]) -> tuple[list[str], list[dict[str, str]]]:
        """Extract and structure the context documents and their metadata.

        This method processes a list of contexts, extracting the text and relevant metadata
        (e.g., IDs and titles) into separate lists.

        Args:
            contexts (list[dict[str, str]]): A list of context dictionaries, where each dictionary
                contains the following keys:
                - "text" (str): The content of the context.
                - "id" (str): A unique identifier for the context.
                - "title" (str): The title of the context.

        Returns:
            tuple[list[str], list[dict[str, str]]]: A tuple containing:
                - docs (list[str]): A list of context text extracted from the dataset.
                - metadata (list[dict[str, str]]): A list of metadata dictionaries, where each dictionary
                contains the "id" and "title" of a context.
        """
        docs = [context["text"] for context in contexts]
        metadata = [{"id": context["id"], "title": context["title"]} for context in contexts]
        return docs, metadata

    def _format_data_into_text_metadata(
        self, data: list[dict[str, Union[str, list[str], list[dict[str, str]]]]]
    ) -> list[dict[str, dict[str, Any]]]:
        """Format the provided data into a structured list of dictionaries with 'text' and 'metadata' fields.

        This method processes the input data, which contains rows with fields like question, choices, answer,
        docs, and metadata. It transforms each row into a dictionary where:
        - 'text' contains the document string from the 'docs' field.
        - 'metadata' contains structured metadata, including the question, choices, answer, answerKey, and any
        additional metadata.

        Args:
            data (list[dict[str, Union[str, list[str], list[dict[str, str]]]]]):
                A list of dictionaries, where each dictionary represents a data entry containing:
                - 'question' (str): The question string.
                - 'choices' (list[str]): A list of choice strings.
                - 'answer' (str): The answer string.
                - 'answerKey' (str): The answer key.
                - 'docs' (list[str]): A list of document strings.
                - 'metadata' (list[dict[str, Any]]): A list of metadata dictionaries.

        Returns:
            list[dict[str, dict[str, Any]]]:
                A list of dictionaries, where each dictionary contains:
                - 'text' (str): A document string from the 'docs' field.
                - 'metadata' (dict[str, Any]): A dictionary containing metadata for the document, including
                the question, choices, answer, answerKey, and any additional metadata.

        Raises:
            ValueError: If the input data is not a list of dictionaries or does not contain the expected fields.
            ValueError: If the 'docs' field is not a list or contains non-string elements.
            ValueError: If a document in 'docs' is not a string.
        """
        # Validate that the input data is a list of dictionaries
        if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
            msg = "Input data must be a list of dictionaries."
            logger.error(msg)
            raise ValueError(msg)

        formatted_data = []

        for row in data:
            # Validate and extract required fields from the current row
            question: str = row.get("question", "")
            answers: list[str] = row.get("answers", [""])
            answer: str = row.get("answer", "")
            documents: list[str] = row.get("docs", [""])
            document_metadatas: list[dict[str, Any]] = row.get("metadata", [])

            # Validate that 'documents' is a list of texts
            if not isinstance(documents, list):
                msg = f"Expected 'docs' to be a list of texts, got {type(documents).__name__}"
                logger.error(msg)
                raise ValueError(msg)

            for document, metadata in zip(documents, document_metadatas):
                # Validate that each document is a string
                if not isinstance(document, str):
                    msg = f"Expected document in 'documents' to be a string, got {type(document).__name__}"
                    logger.error(msg)
                    raise ValueError(msg)

                formatted_data.append(
                    {
                        "text": document,
                        "metadata": {
                            "question": question,
                            "answers": answers,
                            "answer": answer,
                            **metadata,  # Include any additional metadata provided
                        },
                    }
                )

        logger.info(f"Formatted {len(formatted_data)} entries successfully.")

        return formatted_data

    def _preprocess_docs(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Preprocess documents by formatting, splitting, and chunking them into smaller units.

        This method applies a preprocessing pipeline to the input data:
        1. Initializes a text splitter using the specified splitter name and parameters.
        2. Converts the input data from dictionaries into LangChain Document objects.
        3. Applies the text splitter to chunk the documents into smaller pieces.
        4. Converts the chunked documents back into dictionary format for further processing.

        Args:
            data (list[dict[str, Any]]):
                A list of dictionaries, each representing a document to preprocess.
                Each dictionary must contain at least the `page_content` (str) and `metadata` (dict[str, Any]) fields.

        Returns:
            list[dict[str, Any]]:
                A list of dictionaries representing the processed documents, where the `page_content` is split into
                smaller chunks, as determined by the configuration of the text splitter. Each dictionary retains the original
                metadata along with the newly chunked `page_content`.

        Raises:
            ValueError: If the input data is improperly formatted or cannot be processed (e.g., missing required fields).
        """
        # Initialize text splitter
        self.splitter = TextSplitter(splitter_name=self.text_splitter, splitter_params=self.text_splitter_params)

        # Format documents for the splitter (convert dictionary to LangChain Documents)
        data = DocumentTransformer.dict_to_documents(data=data, format_type="langchain")

        # Apply the splitter to chunk the documents
        transformed_docs = self.splitter.transform_documents(data)

        # Convert chunked documents back into dictionary format
        transformed_data = DocumentTransformer.langchain_docs_to_dict(transformed_docs)

        return transformed_data

    def load_data(self) -> list[dict[str, Union[str, list[str], list[dict[str, str]]]]]:
        """Load and process the PopQA dataset,  format the questions and answers, and structure the context documents and metadata.

        This method processes the PopQA dataset by extracting the question, choices, answer, and context, formatting them
        into a structured form suitable for downstream tasks. The processed data is cached to avoid reprocessing,
        which helps in efficient reuse of the data.

        Returns:
            list[dict[str, Union[str, dict[str, Any]]]]:
            A list of dictionaries, where each dictionary contains:
            - 'text' (str): A document string from the 'docs' field.
            - 'metadata' (dict[str, Any]): A dictionary containing metadata for the document, including
            the question, choices, answer, answerKey, and any additional metadata.

        Raises:
            ValueError: If the dataset is empty or improperly formatted.
        """
        if self.data is None:
            logger.info("Loading and processing dataset.")
            self.data = []
            for row in self.dataset:
                # Extract and format the question, answer
                question = row["question"]
                answers = row["answers"]
                summarized_answer = self._summarize_answers(question, answers)

                # Extract contexts and metadata
                docs, metadata = self._extract_contexts(row["ctxs"])

                # Append processed data
                self.data.append(
                    {
                        "question": question,
                        "answers": answers,
                        "answer": summarized_answer,
                        "docs": docs,
                        "metadata": metadata,
                    }
                )
            logger.info(f"Processed {len(self.data)} rows from the dataset.")

        # Format the data into text and metadata
        self.corpus = self._format_data_into_text_metadata(self.data)

        # Preprocess the documents (chunking and formatting)
        self.corpus = self._preprocess_docs(self.corpus)

        return self.corpus

    def get_langchain_documents(self) -> list[LangchainDocument]:
        """Convert the processed corpus to LangChain Document objects.

        This method converts the `self.corpus`, which contains the processed text and metadata, into LangChain Document objects
        for use with LangChain-based pipelines. It ensures that the data is loaded before processing.

        Returns:
            list[LangchainDocument]: A list of LangChain Document objects, where each document represents a chunked
            text from the corpus with corresponding metadata.

        Raises:
            ValueError: If `self.corpus` is empty (i.e., the data has not been loaded).
        """
        if not self.corpus:
            err_msg = "Data must be loaded before creating documents. Please call load_data() first."
            logger.error(err_msg)
            raise ValueError(err_msg)

        # Convert the corpus to LangChain documents
        langchain_docs = DocumentTransformer.dict_to_documents(data=self.corpus, format_type="langchain")
        return langchain_docs

    def get_haystack_documents(self) -> list[HaystackDocument]:
        """Convert the processed corpus to Haystack Document objects.

        This method converts the `self.corpus`, which contains the processed text and metadata, into Haystack Document objects
        for use in Haystack-based retrieval pipelines. It ensures that the data is loaded before processing.

        Returns:
            list[HaystackDocument]: A list of Haystack Document objects, where each document represents a chunked text
            from the corpus with corresponding metadata.

        Raises:
            ValueError: If `self.corpus` is empty (i.e., the data has not been loaded).
        """
        if not self.corpus:
            err_msg = "Data must be loaded before creating documents. Please call load_data() first."
            logger.error(err_msg)
            raise ValueError(err_msg)

        # Convert the corpus to Haystack documents
        haystack_docs = DocumentTransformer.dict_to_documents(data=self.corpus, format_type="haystack")
        return haystack_docs

    def get_evaluation_data(self) -> list[dict[str, Any]]:
        """Prepare and format data for evaluation.

        This method structures the data for evaluation purposes. Each evaluation instance consists of:
        - `question`: The question to be evaluated.
        - `answer`: The expected answer.
        - `docs`: The associated documents that are relevant to the question.

        If no data has been loaded, the method automatically calls `load_data` to load the necessary information.

        Returns:
            list[dict[str, Any]]:
                A list of dictionaries, each representing an evaluation instance with the following keys:
                - "question" (str): The question text.
                - "answer" (str): The expected answer text.
                - "docs" (list[str]): A list of documents relevant to the question.

        Raises:
            ValueError: If no data is available after calling `load_data`. This occurs when `self.data` is empty
                        or not properly loaded.
        """
        if self.data is None:
            self.load_data()

        if not self.data:
            msg = "No data available for evaluation. Ensure `load_data` has been called and contains valid data."
            logger.error(msg)
            raise ValueError(msg)

        # Prepare the evaluation data
        eval_data = [
            {
                "question": row["question"],
                "answer": row["answer"],
                "docs": row["docs"],
            }
            for row in self.data
        ]

        logger.info(f"Prepared {len(eval_data)} rows for evaluation.")
        return eval_data

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
        if self.data is None:
            self.load_data()

        if not self.data:
            msg = "No data available to publish. Ensure `load_data` has been called and contains valid data."
            logger.error(msg)
            raise ValueError(msg)

        # Publish dataset
        logger.info(f"Publishing {len(self.data)} rows to Weave dataset.")
        weave.publish(weave.Dataset(name=dataset_name, rows=self.data))

        if evaluation_dataset_name:
            # Publish evaluation dataset
            evaluation_data = self.get_evaluation_data()
            logger.info(f"Publishing {len(evaluation_data)} rows to Weave evaluation dataset.")
            weave.publish(weave.Dataset(name=evaluation_dataset_name, rows=evaluation_data))
