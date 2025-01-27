import base64
import io
import logging
import os
import re
from typing import Any, Optional, Union

import filetype
import pandas as pd
import weave
from datasets import load_dataset
from edgar import Company, set_identity
from haystack import Document as HaystackDocument
from langchain_core.documents import Document as LangchainDocument
from PIL import Image
from tqdm.auto import tqdm

from dataloaders.haystack.finance_companies_map import COMPANY_MAP, FilingItems10k, FilingItems10q
from dataloaders.haystack.llms.groq import ChatGroqGenerator
from dataloaders.haystack.prompts.financial_summary import FinancialSummaryKeywordsPrompt, FinancialSummaryPrompt
from dataloaders.haystack.utils import DocumentTransformer, LoggerFactory

# Set the identity for accessing EDGAR API
DEFAULT_SEC_IDENTITY = "john@finrag.org"
set_identity(DEFAULT_SEC_IDENTITY)

logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()


class EdgarDataloader:
    """A class for loading, processing, and managing SEC filings from the EDGAR database.

    This class interacts with the HuggingFace `datasets` library to load SEC filings. It includes a
    text splitter for breaking down filings and an image description generator based on LLMs.

    Attributes:
        dataset (pd.DataFrame): The dataset loaded from HuggingFace's datasets library.
        image_description_generator (ChatGroqGenerator): LLM-based generator for creating image descriptions.
        data (Optional[List[Dict]]): Cached processed data.
        valid_item_names_10k (set): A set of valid 10-K filing item names.
        valid_item_names_10q (set): A set of valid 10-Q filing item names.
        text_splitter (str): The name of the text splitter to use.
        text_splitter_params (Optional[dict]): Parameters for the text splitter.
        corpus (list): Processed document corpus.
    """

    def __init__(
        self,
        image_description_generator: ChatGroqGenerator,
        dataset_name: str = "PatronusAI/financebench",
        split: str = "train",
        text_splitter: str = "RecursiveCharacterTextSplitter",
        text_splitter_params: Optional[dict[str, Union[str, int]]] = None,
    ):
        """Initialize the EdgarDataLoader with the provided parameters.

        Args:
            image_description_generator (ChatGroqGenerator): An instance of a generator for image descriptions.
            dataset_name (str): The name of the dataset from HuggingFace's datasets library (default: 'PatronusAI/financebench').
            split (str): The split of the dataset to load (default: 'train').
            text_splitter (str): The text splitter to use for splitting the documents (default: 'RecursiveCharacterTextSplitter').
            text_splitter_params (Optional[dict[str, Union[str, int]]]): Optional parameters for the text splitter.

        Raises:
            ValueError: If the dataset cannot be loaded.
        """
        self.dataset = load_dataset(dataset_name, split)
        self.valid_item_names_10k = {item.value for item in FilingItems10k}
        self.valid_item_names_10q = {item.value for item in FilingItems10q}
        self.image_description_generator = image_description_generator
        self.data: Optional[list[dict[str, Union[str, list[str], list[dict[str, str]]]]]] = None
        self.text_splitter = text_splitter
        self.text_splitter_params = text_splitter_params
        self.corpus: list[dict[str, Any]] = []

    def _map_report_type(self, report_type: str) -> Optional[str]:
        """Map raw report types from 'document_name' to standardized types.

        Args:
            report_type (str): The raw report type (e.g., '10k', 'annualreport').

        Returns:
            Optional[str]: The standardized report type or None if the report type is not recognized.
        """
        report_type_mapping = {
            "10k": "10k",
            "annualreport": "10k",
            "quarterlyreport": "10q",
            "10q": "10q",
            "8k": "8k",
            "earnings": "earnings",
        }
        mapped_type = report_type_mapping.get(report_type.lower(), None)
        if mapped_type is None:
            logging.warning(f"Unrecognized report type: {report_type}")
        return mapped_type

    def _extract_document_info(self, doc_name: str) -> pd.Series:
        """Extract company name, year, quarter, date, and report type from a document name.

        This method processes the document name to extract structured information, such as:
        - Company name
        - Year
        - Quarter (if applicable)
        - Document date
        - Report type (e.g., 10-K, 10-Q)

        Args:
            doc_name (str): The document name, which typically includes the company name,
                            year, quarter, date, and report type (e.g., 'apple_2022Q1_10k_dated_2022-02-25').

        Returns:
            pd.Series: A Pandas Series containing the extracted information:
                - Company name (str or None)
                - Year (str or None)
                - Quarter (str or None)
                - Date (str or None)
                - Report type (str or None)

        Raises:
            ValueError: If the document name format is unrecognized or invalid.
        """
        try:
            parts = doc_name.split("_")
            company_parts = []

            # Extract company name
            for part in parts:
                if re.match(r"\d{4}(Q\d)?", part):
                    break
                company_parts.append(part)

            company = "_".join(company_parts).lower() if company_parts else None

            # Extract year and quarter
            year_quarter_match = re.match(r"(\d{4})(Q\d)?", parts[len(company_parts)])
            if year_quarter_match:
                year = year_quarter_match.group(1)
                quarter = year_quarter_match.group(2) if year_quarter_match.group(2) else None
            else:
                year = quarter = None

            # Extract report type
            report_type_raw = parts[len(company_parts) + 1].lower() if len(parts) > len(company_parts) + 1 else None
            report_type = self._map_report_type(report_type_raw)

            # Extract date
            date_match = (
                re.search(r"dated[-_]?(\d{4}-\d{2}-\d{2})", "_".join(parts[len(company_parts) + 2 :]))
                if len(parts) > len(company_parts) + 2
                else None
            )
            date = date_match.group(1) if date_match else None

            logging.info(f"Extracted information from document name: {doc_name}")
            return pd.Series([company, year, quarter, date, report_type])

        except Exception as e:
            logging.error(f"Error extracting document information from '{doc_name}': {e}")
            msg = f"Failed to extract information from document name: {doc_name}."
            raise ValueError(msg) from e

    def _extract_and_validate_company_information(self) -> None:
        """Extract and validate company information in the dataset.

        This method applies the `_extract_document_info` function to the 'doc_name' column of the dataset to
        extract company-related information, including the company name, year, quarter, date, and report type.
        It then validates the consistency of the report type by comparing the extracted report type with the
        provided `doc_type`. Additionally, it maps company names to ticker symbols using a predefined mapping.

        The extracted and validated information is added to the dataset as new columns:
        - 'Company': Extracted company name.
        - 'Year': Extracted year from the document name.
        - 'Quarter': Extracted quarter (if available).
        - 'Date': Extracted date from the document name.
        - 'Extracted_Report_Type': Extracted report type.
        - 'Ticker': Company ticker symbol mapped from company name.
        - 'is_consistent': Boolean flag indicating whether the extracted report type matches the provided `doc_type`.

        Raises:
            KeyError: If the 'doc_name' or 'doc_type' columns are missing in the dataset.
        """
        try:
            if "doc_name" not in self.dataset or "doc_type" not in self.dataset:
                msg = "The dataset must contain 'doc_name' and 'doc_type' columns."
                raise KeyError(msg)

            self.dataset[["Company", "Year", "Quarter", "Date", "Extracted_Report_Type"]] = self.dataset[
                "doc_name"
            ].apply(self._extract_document_info)
            self.dataset["Ticker"] = self.dataset["Company"].map(COMPANY_MAP)
            self.dataset["is_consistent"] = (
                self.dataset["Extracted_Report_Type"].str.lower() == self.dataset["doc_type"].str.lower()
            )
            logging.info("Company information extracted and validation completed successfully.")

        except KeyError as e:
            logging.error(f"Missing required column in the dataset: {e}")
            raise
        except Exception as e:
            logging.error(f"Error processing company information: {e}")
            raise

    def _encode_image(self, image: Image.Image) -> str:
        """Encode an image in Base64 format.

        This method converts an image (in PIL format) to a Base64 encoded string, which is useful for embedding
        images in HTML or other text-based formats.

        Args:
            image (Image.Image): The image to be encoded.

        Returns:
            str: The Base64 encoded image string, prefixed with the 'data:image/png;base64,' format.

        Raises:
            ValueError: If the image format is unsupported or if there is an error during encoding.
        """
        try:
            byte_arr = io.BytesIO()
            image.save(byte_arr, format="PNG")
            encoded_image = base64.b64encode(byte_arr.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{encoded_image}"

        except Exception as e:
            logging.error(f"Error encoding image: {e}")
            msg = "Failed to encode image in Base64."
            raise ValueError(msg) from e

    def _generate_image_description(self, images: list[Image.Image], filing_summary: str) -> list[str]:
        """Generate descriptions for images using the LLM-based description generator.

        This method processes a list of images, encoding each one into a Base64 string, and uses the provided
        LLM-based generator to create descriptions for each image. The descriptions are based on the filing summary
        and the image itself, which are passed as prompts to the generator.

        Args:
            images (list[Image.Image]): A list of images to be described.
            filing_summary (str): A summary of the filing to be included in the description prompt.

        Returns:
            list[str]: A list of image descriptions, where each description corresponds to an image in the input list.

        Logs:
            Errors encountered during the image description generation process.

        """
        image_descriptions = []
        for image in tqdm(images, desc="Generating image descriptions"):
            try:
                # Generate description for each image using the LLM-based generator
                description = self.image_description_generator.predict(
                    user_prompts=[
                        FinancialSummaryPrompt.format(filing_summary=filing_summary),
                        self._encode_image(image),
                    ]
                )
                image_descriptions.append(description)
            except Exception as e:
                logger.error(f"Error generating image description: {e}")
                image_descriptions.append("Error generating image description")
        return image_descriptions

    def _summarize_filing(self, filing_data: str) -> list[str]:
        """Summarizes filing data using the LLM-based generator.

        This method generates a summary of the filing data by passing it as a prompt to the LLM-based generator.
        The generator creates a textual summary of the provided filing data.

        Args:
            filing_data (str): The text data of the filing to be summarized.

        Returns:
            list[str]: A list containing the generated summary.

        Logs:
            Errors encountered during the filing summary generation process.

        """
        try:
            # Generate the filing summary using the LLM-based generator
            return self.image_description_generator.predict(
                user_prompts=[FinancialSummaryKeywordsPrompt.format(filing_data=filing_data)]
            )
        except Exception as e:
            logger.error(f"Error generating filing summary: {e}")
            return ["Error generating summary"]

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

    def _process_filing(
        self,
        company_name: str,
        form_type: str,
        upload_images: bool = True,
        upload_image_descriptions: bool = True,
    ) -> list[dict[str, Union[str, list[str]]]]:
        """Process filings for a given company and form type, and optionally upload images and descriptions.

        This method retrieves filings for a given company and form type, processes the filing data,
        generates summaries and image descriptions, and stores the results in a structured format. It supports
        uploading images and generating image descriptions for the filings.

        Args:
            company_name (str): The name of the company whose filings are to be retrieved.
            form_type (str): The form type (e.g., '10-K', '10-Q') for which filings are to be fetched.
            upload_images (bool, optional): If True, attempts to download and process images associated with the filings. Defaults to True.
            upload_image_descriptions (bool, optional): If True, generates descriptions for the images. Defaults to True.

        Returns:
            List[Dict[str, Union[str, List[str]]]]: A list of dictionaries, each containing metadata and processed data
            for a filing, including form type, filing date, accession number, summary, and image data.

        Logs:
            Errors encountered during the image loading and description generation process.

        """
        filings = Company(company_name).get_filings(form=[form_type])
        filings_data = []

        # Iterate through each filing and process the relevant information
        for filing in tqdm(filings, desc=f"Fetching {form_type} filings for {company_name}"):
            filing_markdown = filing.markdown()
            filing_summary = self._summarize_filing(filing_markdown)

            filing_date = re.search(r"\((.*?)\)", str(filing.filing_date or ""))
            filing_date = filing_date.group(1).replace("/", "-") if filing_date else "unknown-date"

            current_filing_data = {
                "form_type": filing.primary_doc_description,
                "filing_date": filing_date,
                "accession_no": filing.accession_no,
                "cik": filing.cik,
                "content": filing_markdown,
                "summary": filing_summary,
                "images": [],
            }

            # Upload images if the flag is set to True
            if upload_images:
                os.makedirs(os.path.join("./attachments", company_name), exist_ok=True)
                for idx, attachment in enumerate(filing.attachments):
                    attachment_path = os.path.join("./attachments", company_name, f"{idx}{attachment.extension}")
                    attachment.download(path=attachment_path)
                    if filetype.is_image(attachment_path):
                        try:
                            image = Image.open(attachment_path)
                            image.load()
                            current_filing_data["images"].append(image)
                        except Exception as e:
                            logger.error(f"Error loading image: {e}")

            # Generate and upload image descriptions if the flag is set to True
            if upload_image_descriptions:
                current_filing_data["image_descriptions"] = self._generate_image_description(
                    images=current_filing_data["images"], filing_summary=filing_summary
                )

            filings_data.append(current_filing_data)

        return filings_data

    def load_data(
        self,
    ) -> list[dict[str, Union[str, list[str]]]]:
        """Load and process SEC filings for all companies and form types in the dataset.

        This method loads the dataset, processes SEC filings for different companies and form types
        (such as '10-K' and '10-Q'), extracts relevant information, and preprocesses the documents
        for further use. It caches the processed data for efficiency.

        The processed data includes the following:
            - 'text': The content of the filing.
            - 'metadata': Contains filing-specific metadata like form type, filing date, accession number,
            CIK, summary, images, and image descriptions.

        If the data is already loaded, it returns the cached data.

        Returns:
            List[Dict[str, Union[str, List[str]]]]: A list of processed documents, each containing 'text'
            (filing content) and 'metadata' (filing-specific information).

        Logs:
            - Information on the number of rows processed and when the dataset is being loaded.
        """
        if self.data is None:
            logger.info("Loading and processing dataset.")
            self.data = []

            filings_data = []
            # Process filings for each unique company and form type
            for company in self.dataset["Company"].unique():
                for form in ["10-K", "10-Q"]:
                    filings_data += self._process_filing(company_name=company, form_type=form)

            # Process each filing and append the structured data
            for filing in filings_data:
                text = (filing["content"],)
                metadata = (
                    {
                        "form_type": filing["form_type"],
                        "filing_date": filing["filing_date"],
                        "accession_no": filing["accession_no"],
                        "cik": filing["cik"],
                        "summary": filing["summary"],
                        "images": filing["images"],
                        "image_descriptions": filing.get("image_descriptions", []),
                    },
                )

                self.data.append(
                    {
                        "text": text,
                        "metadata": metadata,
                    }
                )
            logger.info(f"Processed {len(self.data)} rows from the dataset.")

        # Preprocess the documents (chunking and formatting)
        self.corpus = self._preprocess_docs(self.data)

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
