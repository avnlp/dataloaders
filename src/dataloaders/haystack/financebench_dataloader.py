import logging
import re
from pathlib import Path
from typing import Any, Optional, Union

import fsspec
import weave
from datasets import load_dataset
from haystack import Document
from haystack.components.preprocessors import RecursiveDocumentSplitter
from haystack.core.component import Component

from dataloaders.haystack.finance_companies_map import COMPANY_MAP
from dataloaders.haystack.utils import DocumentTransformer, LoggerFactory

logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()


class FinanceBenchDataloader:
    """A data loader class for processing and preparing financial document data from the FinanceBench dataset.

    This class handles loading, parsing, and structuring financial documents and their metadata from the FinanceBench
    dataset. It processes document names to extract key components (company, year, quarter, etc.), splits documents
    into manageable chunks, and prepares data for downstream tasks like retrieval-augmented QA systems.

    Attributes:
        dataset (Dataset): The loaded FinanceBench dataset split (e.g., "test").
        data (Optional[list[dict]]): Cached processed data containing questions, answers, and document metadata.
        corpus (list[dict]): Processed and chunked documents with metadata, ready for pipeline integration.
    """

    def __init__(
        self,
        dataset_name: str = "PatronusAI/financebench",
        split: str = "test",
        text_splitter: Component = RecursiveDocumentSplitter(),
    ):
        """Initialize the FinanceBenchDataloader with the specified dataset.

        Args:
            dataset_name (str): Name of the dataset to load. Defaults to "PatronusAI/financebench".
            split (str): Split of the dataset to load (e.g., "test", "train"). Defaults to "test".
            text_splitter (Component): The text splitter to use for processing documents. Defaults to
                RecursiveDocumentSplitter.
        """
        self.dataset = load_dataset(dataset_name, split=split)
        self.data: Optional[list[dict[str, Union[str, list[str], list[dict[str, str]]]]]] = None
        self.text_splitter = text_splitter
        self.corpus: list[dict[str, Any]] = []

    def _extract_contexts(
        self,
        contexts: list[dict[str, Any]],
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """Extract and structure contextual evidence from financial documents.

        Processes a list of evidence contexts from financial filings, extracting:
        - Specific evidence text excerpts
        - Full document page texts
        - Document identifiers
        - Page number references

        Args:
            contexts (list[dict]): List of evidence context dictionaries containing:
                - "evidence_text" (str): Relevant text excerpt supporting a financial fact
                - "doc_name" (str): Source document identifier (e.g., "AMZN_2022_10K")
                - "evidence_page_num" (int): Page number where evidence was found
                - "evidence_text_full_page" (str): Full text content of the evidence page

        Returns:
            tuple: Four parallel lists containing:
                - docs (list[str]): Evidence text excerpts
                - doc_names (list[str]): Source document identifiers
                - evidence_page_nums (list[str]): Page numbers as strings
                - evidence_text_full_pages (list[str]): Full page texts

        Example:
            Given a context with doc_name "AMZN_2022_10K" and page_num 42, returns:
            (["Revenue increased..."], ["AMZN_2022_10K"], ["42"], ["Full page text..."])
        """
        docs = []
        doc_names = []
        evidence_page_nums = []
        evidence_text_full_pages = []

        for context in contexts:
            docs.append(context["evidence_text"])
            doc_names.append(context["doc_name"])
            evidence_page_nums.append(str(context["evidence_page_num"]))
            evidence_text_full_pages.append(context["evidence_text_full_page"])

        return docs, doc_names, evidence_page_nums, evidence_text_full_pages

    def _parse_document_name(self, doc_name: str) -> tuple[str, str, str, str, str, str]:
        """Deconstruct financial document names into structured components.

        Parses SEC-style document identifiers into their constituent parts using a defined pattern:
        [Company]_[Year][Quarter?]_[DocType]_[Date?]

        Args:
            doc_name (str): Document name following SEC filing conventions. Examples:
                - "AMZN_2022_10K" (Annual Report)
                - "MSFT_2023Q2_10Q" (Quarterly Report)
                - "JNJ_2023_8K_dated-2023-08-30" (Current Report with date)

        Returns:
            tuple: Structured components:
                - year (str): 4-digit fiscal year
                - quarter (str): Fiscal quarter (Q1-Q4) if present
                - date (str): Filing date in YYYY-MM-DD format if present
                - company_name (str): Normalized company name (lowercase with underscores)
                - company_ticker (str): Stock ticker symbol from COMPANY_MAP
                - document_type (str): SEC document type (10K, 10Q, 8K, EARNINGS)

        Example:
            Input: "AMZN_2022Q4_10Q_dated-2022-12-31"
            Output: ("2022", "Q4", "2022-12-31", "amzn", "amzn", "10Q")
        """
        parts = doc_name.split("_")
        year = ""
        quarter = ""
        date = ""
        company_name = ""
        document_type = ""

        # Find the index of the first part that is a 4-digit year
        year_part_index = None
        for i, part in enumerate(parts):
            if re.match(r"^\d{4}", part):
                year_part_index = i
                break

        if year_part_index is None:
            return ("", "", "", "", "", "")

        # Extract company name (parts before the year part, joined with underscores)
        company_name = "_".join(parts[:year_part_index]).lower()

        # Extract year and quarter from the year part
        year_part = parts[year_part_index]
        if "Q" in year_part:
            year_split = year_part.split("Q", 1)
            year = year_split[0]
            q_num = year_split[1]
            quarter = f"Q{q_num}" if q_num in {"1", "2", "3", "4"} else ""
        else:
            year = year_part
            quarter = ""

        # Extract document type (next part after year part if it's a valid type)
        allowed_doc_types = {"10K", "10Q", "8K", "EARNINGS"}
        doc_type_index = None
        for i in range(year_part_index + 1, len(parts)):
            if parts[i] in allowed_doc_types:
                document_type = parts[i]
                doc_type_index = i
                break
            elif parts[i].upper() == "EARNINGS":
                document_type = "EARNINGS"
                doc_type_index = i
                break

        if document_type == "":
            return ("", "", "", company_name, "", "")

        # Extract date from remaining parts (after document type or year)
        remaining_parts = parts[doc_type_index + 1 :] if doc_type_index is not None else parts[year_part_index + 1 :]
        date_str = "_".join(remaining_parts)
        date_match = re.search(r"dated[-_](19\d{2}|20\d{2})[-_](\d{2})[-_](\d{2})", date_str)
        if not date_match:
            date_match = re.search(r"(19\d{2}|20\d{2})[-_](\d{2})[-_](\d{2})", date_str)
        if date_match:
            date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"

        # Get company ticker from COMPANY_MAP (case-insensitive)
        company_ticker = COMPANY_MAP.get(company_name, "").lower()

        return (year, quarter, date, company_name, company_ticker, document_type)

    def _format_data_into_text_metadata(
        self, data: list[dict[str, Union[str, list[str], list[dict[str, str]]]]]
    ) -> list[dict[str, dict[str, Any]]]:
        """Structure financial QA data into standardized format for processing pipelines.

        Transforms raw dataset entries into a unified format containing:
        - Question text as main content
        - Comprehensive metadata including answers, evidence, and document context

        Args:
            data (list[dict]): Raw dataset entries containing:
                - question (str): Financial query text
                - answer (str): Ground truth answer
                - justification (str): Reasoning for correct answer
                - docs (list[str]): Relevant evidence passages
                - Various metadata fields (company, document type, fiscal year, etc.)

        Returns:
            list[dict]: Standardized entries with:
                - text (str): Question text
                - metadata (dict): All associated contextual information

        Raises:
            ValueError: If input data structure is invalid or missing required fields
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
            answer: str = row.get("answer", "")
            justification: str = row.get("justification", "")
            company: str = row.get("company", "")
            doc_name: str = row.get("doc_name", "")
            question_type: str = row.get("question_type", "")
            doc_type: str = row.get("doc_type", "")
            year: str = row.get("year", "")
            doc_link: str = row.get("doc_link", "")
            question_reasoning: str = row.get("question_reasoning", "")
            domain_question: str = row.get("domain_question", "")
            docs: list[str] = row.get("docs", [])
            doc_names: list[str] = row.get("doc_names", [])
            evidence_page_nums: list[str] = row.get("evidence_page_nums", [])
            evidence_text_full_pages: list[str] = row.get("evidence_text_full_pages", [])
            parsed_year: str = row.get("parsed_year", "")
            quarter: str = row.get("quarter", "")
            date: str = row.get("date", "")
            company_name: str = row.get("company_name", "")
            ticker: str = row.get("ticker", "")
            document_type: str = row.get("document_type", "")

            formatted_data.append(
                {
                    "text": question,
                    "metadata": {
                        "question": question,
                        "answer": answer,
                        "justification": justification,
                        "company": company,
                        "doc_name": doc_name,
                        "question_type": question_type,
                        "doc_type": doc_type,
                        "year": year,
                        "doc_link": doc_link,
                        "question_reasoning": question_reasoning,
                        "domain_question": domain_question,
                        "docs": docs,
                        "doc_names": doc_names,
                        "evidence_page_nums": evidence_page_nums,
                        "evidence_text_full_pages": evidence_text_full_pages,
                        "parsed_year": parsed_year,
                        "quarter": quarter,
                        "date": date,
                        "company_name": company_name,
                        "ticker": ticker,
                        "document_type": document_type,
                    },
                }
            )

        logger.info(f"Formatted {len(formatted_data)} entries successfully.")

        return formatted_data

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
                # Extract and format the question, answer
                question = row["question"]
                answer = row["answer"]
                justification = row["justification"]
                company = row["company"]
                doc_name = row["doc_name"]
                question_type = row["question_type"]
                doc_type = row["doc_type"]
                year = str(row["doc_period"])
                doc_link = row["doc_link"]

                question_reasoning = row["question_reasoning"]
                domain_question = row["domain_question_num"]

                # Extract contexts and metadata
                docs, doc_names, evidence_page_nums, evidence_text_full_pages = self._extract_contexts(row["evidence"])

                # Extract metadata from doc_name
                parsed_year, quarter, date, company_name, company_ticker, document_type = self._parse_document_name(
                    doc_name
                )

                # Append processed data
                self.data.append(
                    {
                        "question": question,
                        "answer": answer,
                        "justification": justification,
                        "company": company,
                        "doc_name": doc_name,
                        "question_type": question_type,
                        "doc_type": doc_type,
                        "year": year,
                        "doc_link": doc_link,
                        "question_reasoning": question_reasoning,
                        "domain_question": domain_question,
                        "docs": docs,
                        "doc_names": doc_names,
                        "evidence_page_nums": evidence_page_nums,
                        "evidence_text_full_pages": evidence_text_full_pages,
                        "parsed_year": parsed_year,
                        "quarter": quarter,
                        "date": date,
                        "company_name": company_name,
                        "company_ticker": company_ticker,
                        "document_type": document_type,
                    }
                )
            logger.info(f"Processed {len(self.data)} rows from the dataset.")

        # Format the data into text and metadata
        self.corpus = self._format_data_into_text_metadata(self.data)

        # Preprocess the documents (chunking and formatting)
        self.corpus = self._preprocess_docs(self.corpus)

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
            err_msg = "Data must be loaded before creating documents. Please call load_data() first."
            logger.error(err_msg)
            raise ValueError(err_msg)

        # Convert the corpus to Haystack documents
        haystack_docs = DocumentTransformer.dict_to_documents(data=self.corpus)

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
                - "input" (str): The question.
                - "expected_output" (str): The expected answer text.
                - "context" (list[str]): A list of documents relevant to the question.

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
                "input": row["question"],
                "expected_output": row["answer"],
                "context": row["docs"],
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

    def get_corpus_pdfs(self) -> None:
        """Download all the PDFs from the GitHub repository.

        This method downloads all the PDF files from the "pdfs/" directory in the GitHub repository.

        Raises:
            ValueError: If the dataset is empty or not properly loaded.
        """
        doc_names = list(set(self.dataset["doc_name"]))
        filenames = [f"{doc_name}.pdf" for doc_name in doc_names]

        # Directory to store the PDFs
        destination = Path("pdfs")
        destination.mkdir(exist_ok=True, parents=True)

        # Initialize the GitHub filesystem
        fs = fsspec.filesystem("github", org="patronus-ai", repo="financebench")

        # List files in the "pdfs/" directory in the repository
        repo_pdf_path = "pdfs/"
        pdf_files = fs.ls(repo_pdf_path)

        # Iterate over required files and download them
        for filename in filenames:
            # Full path in the repository
            file_path_in_repo = f"{repo_pdf_path}{filename}"
            # Full local path
            local_file_path = destination / filename

            # Check if the file exists in the repository
            if file_path_in_repo in pdf_files:
                try:
                    # Download the file to the local destination
                    fs.get(file_path_in_repo, local_file_path.as_posix())
                except Exception as e:
                    print(f"Failed to download {filename}: {e}")
            else:
                print(f"File not found in repository: {filename}")

        print("PDF download complete.")

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
