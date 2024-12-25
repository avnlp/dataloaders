# Dataloaders

Dataloaders is a versatile library designed for processing and formatting datasets to support various Retrieval-Augmented Generation (RAG) pipelines, facilitating efficient evaluation and analysis.

## Features

The library provides a unified interface for working with datasets, offering methods to load, preprocess, and evaluate data tailored to RAG pipelines. Key features include:

1. **Data Loading**: Extracts and structures raw data into text and metadata fields.
2. **Question Retrieval**: Retrieves questions for evaluation in RAG pipelines.
3. **Document Conversion**: Prepares data for integration with LangChain and Haystack pipelines.
4. **Text Splitting**: Supports multiple chunking strategies to optimize document segmentation.
5. **Evaluation Publishing**: Publishes processed and evaluation data to Weave.

### Core Methods

#### `load_data()`

Processes the dataset into a structured format suitable for downstream tasks. Returns a list of dictionaries containing:

- **`text`** (str): Document content.
- **`metadata`** (dict): Associated metadata, such as questions, choices, answers, and additional fields.

#### `get_questions()`

Retrieves all questions from the dataset as a list of strings.

#### `get_eval_data()`

Structures data for evaluation, returning instances in the following format:

- **`question`**: The query to be evaluated.
- **`answer`**: The expected answer.
- **`docs`**: Relevant documents supporting the question.

#### `get_haystack_documents()`

Converts the processed data into Haystack `Document` objects, ready for use in Haystack pipelines.

#### `get_langchain_documents()`

Converts the processed data into LangChain `Document` objects, compatible with LangChain pipelines.

#### `publish_to_weave()`

Publishes the processed dataset and evaluation data to a Weave project.

---

## Text Chunking Strategies

Efficient text chunking ensures optimal performance in RAG pipelines. Dataloaders supports the following strategies:

1. **CharacterTextSplitter**: Divides text based on a specific character delimiter.
2. **RecursiveCharacterTextSplitter**: Recursively splits text using a hierarchy of delimiters (e.g., `\n\n`, `\n`, spaces).
3. **SemanticChunker**: Uses embedding models to create semantically coherent chunks.
4. **UnstructuredChunking**: Leverages the `unstructured` library for adaptive document chunking.

---

## Installation

To install the library, clone the repository and install the dependencies:

```bash
git clone https://github.com/avnlp/dataloaders
cd dataloaders
pip install -e .
```

---

## Usage Example

### Loading the ARC Dataset

```python
from dataloaders.arc_dataloader import ARCDataloader

dataloader = ARCDataloader(
    dataset_name="awinml/arc_challenge_processed",
    split="train",
    splitter="UnstructuredChunker",
    splitter_args={"chunking_strategy": "basic"},
)

data = dataloader.load_data()

# Sample output:
# [
#     {
#         "text": "An astronomer observes that a planet rotates faster after a meteorite impact. Which is the most likely effect of this increase in rotation?",
#         "metadata": {
#             "choices": [
#                 "Planetary density will decrease.",
#                 "Planetary years will become longer.",
#                 "Planetary days will become shorter.",
#                 "Planetary gravity will become stronger."
#             ],
#             "answer": "Planetary days will become shorter.",
#             "question": "An astronomer observes that a planet rotates faster..."
#         }
#     }
# ]

evaluation_data = dataloader.get_evaluation_data()

questions = dataloader.get_questions()

langchain_documents = dataloader.get_langchain_documents()

haystack_documents = dataloader.get_haystack_documents()

dataloader.publish_to_weave(
    weave_project_name="arc",
    dataset_name="arc_dataset",
    evaluation_dataset_name="arc_evaluation_dataset",
)
```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
