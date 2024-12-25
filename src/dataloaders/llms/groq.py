import os
from typing import Optional, Union

import weave
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq


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
    llm_params: dict[str, Union[str, int, float, bool, str]]
    llm: Optional[ChatGroq] = None

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        llm_params: Optional[dict[str, Union[str, int, float, bool, str]]] = None,
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

        # Initialize the ChatGroq instance with the specified model, API key, and parameters
        self.llm = ChatGroq(model=self.model, api_key=self.api_key, **(llm_params))

    @weave.op()
    def format_user_prompts(self, prompts: list[str]) -> list[dict[str, str]]:
        """Format a list of user prompts for compatibility with the ChatGroq API.

        This method distinguishes between text-based and image-based prompts,
        formatting each appropriately for the ChatGroq input requirements.

        Args:
            prompts (List[str]): A list of user prompts, where each prompt is a string.

        Returns:
            List[Dict[str, str]]: A list of formatted prompts, where each prompt is a dictionary
            in a compatible with the ChatGroq input format.
        """
        content = []
        for prompt in prompts:
            # Handle image prompts (base64-encoded)
            if prompt.startswith("data:image/png;base64,") or prompt.startswith("data:image/jpeg;base64,"):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": prompt, "detail": "high"},
                    }
                )
            # Handle text prompts
            else:
                content.append({"type": "text", "text": prompt})
        return content

    @weave.op()
    def format_system_prompt(self, prompt: str) -> list[dict[str, str]]:
        """Format a system-level prompt for compatibility with the ChatGroq API.

        Args:
            prompt (str): A string containing the system prompt.

        Returns:
            List[Dict[str, str]]: A formatted system prompt as a dictionary.
        """
        return [{"type": "text", "text": prompt}]

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
        messages: list[Union[SystemMessage, HumanMessage]] = []

        # Add system-level prompt if provided
        if system_prompt:
            system_prompt_contents = self.format_system_prompt(system_prompt)
            messages.append(SystemMessage(content=system_prompt_contents))

        # Format and append user prompts
        user_prompt_contents = self.format_user_prompts(user_prompts)
        messages.append(HumanMessage(content=user_prompt_contents))

        # Invoke the ChatGroq model with the formatted messages
        response = self.llm.invoke(messages, **kwargs)
        # Extract the generated response content
        generated_text = response.content

        return generated_text
