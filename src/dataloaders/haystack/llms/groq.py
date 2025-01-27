import os
from typing import Any, Optional

import weave
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret


class ChatGroqGenerator(weave.Model):
    """A generator class that interacts with the Groq API to generate responses based on user input.

    This class facilitates seamless interaction with the Groq API, including:
    - Formatting user and system prompts.
    - Handling LLM invocation with ChatGroq.
    - Formatting graded documents for input into the model.
    - Generating responses that integrate user prompts, system prompts, and document context.

    Attributes:
        model (str): The name of the LLM to use.
        api_key (str): API key for authentication with the Groq API.
        llm (ChatGroq): An instance of the ChatGroq LLM initialized with the model and API key.
        llm_params (Optional[dict]): Additional parameters for configuring the LLM.

    Methods:
        format_user_prompts(prompts): Formats a list of user prompts for compatibility with ChatGroq.
        format_system_prompt(prompt): Formats a system prompt for compatibility with ChatGroq.
        predict(user_prompts, system_prompt): Generates a response based on user and optional system prompts.
    """

    model: str
    api_key: str
    llm_params: dict[str, Any]
    llm: Optional[OpenAIChatGenerator] = None

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        llm_params: Optional[dict[str, Any]] = None,
    ):
        """Initialize the ChatGroqGenerator instance with the specified model and API key.

        Args:
            model (str): The name of the ChatGroq model to use.
            api_key (Optional[str]): API key for the ChatGroq service. Defaults to the environment variable "GROQ_API_KEY".
            llm_params (Dict): Additional parameters for configuring the ChatGroq model.

        Raises:
            ValueError: If the API key is not provided and the "GROQ_API_KEY" environment variable is not set.
        """
        # Retrieve API key from argument or environment variable
        if llm_params is None:
            llm_params = {}
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if api_key is None:
            msg = "GROQ_API_KEY is not set. Please provide an API key as an environment variable."
            raise ValueError(msg)

        super().__init__(model=model, api_key=api_key, llm_params=llm_params)
        self.model = model
        self.api_key = api_key
        self.llm_params = llm_params

        # Initialize the Groq Generator instance with the specified model, API key, and parameters
        self.llm = OpenAIChatGenerator(
            model=self.model,
            api_key=Secret.from_token(self.api_key),
            api_base_url="https://api.groq.com/openai/v1",
            **(llm_params),
        )

    @weave.op()
    def predict(self, user_prompts: list[str], system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate a response from the ChatGroq model based on user and optional system prompts.

        Args:
            user_prompts (List[str]): A list of user prompts to be processed.
            system_prompt (Optional[str]): An optional system-level instruction for the model.
            **kwargs: Additional keyword arguments for the ChatGroq invoke method.

        Returns:
            str: The generated response text from the ChatGroq model.
        """
        messages: list[ChatMessage] = []

        # Add system-level prompt if provided
        if system_prompt:
            messages.append(ChatMessage.from_system(system_prompt))

        # Format and append user prompts
        user_messages = [ChatMessage.from_user(user_prompt) for user_prompt in user_prompts]
        messages.extend(user_messages)

        # Invoke the Groq model with the formatted messages
        response = self.llm.run(messages, **kwargs)
        # Extract the generated response content
        generated_text = response["replies"][0].text

        return generated_text
