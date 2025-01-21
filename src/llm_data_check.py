import asyncio
import json
import math
import os

from datetime import datetime
from datetime import timezone
from io import StringIO
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import pandas as pd
import yaml

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langfuse.callback import CallbackHandler
from pydantic import BaseModel
from pydantic import Field
from pydantic import create_model
from src import logger


class LLMProcessor:
    """Process text data using a language model and save the results to a JSONL file."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        output_path: str = "output.jsonl",
        system_message: Optional[str] = None,
        output_fields: Optional[List[Dict[str, str]]] = None,
        session_name: Optional[str] = None,
    ) -> None:
        self.llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"), model_name=model_name, temperature=temperature
        )
        if session_name is None:
            session_name = ""
        else:
            session_name = f"{session_name}_"

        self.langfuse_handler = CallbackHandler(
            user_id=os.environ.get("USER_EMAIL"),
            session_id=f"{session_name}{datetime.today().isoformat()}",
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
            host=os.environ.get("LANGFUSE_HOST"),
        )
        self.output_path = Path(output_path)
        self.model_name = model_name
        self.temperature = temperature

        self.output_fields = output_fields or [
            {
                "name": "is_relevant",
                "type": "str",
                "description": "A one-word answer: 'yes' if the document is relevant, 'no' otherwise.",
            }
        ]
        self.system_message = self._generate_system_message(system_message)
        self.schema = self._generate_schema()

    def _generate_system_message(self, base_message: Optional[str]) -> str:
        """Generate the system message by including output field descriptions."""
        fields_description = "\n".join([f"{field['name']}: {field['description']}" for field in self.output_fields])
        return (
            (base_message or "Determine if the document is relevant to the scope of the project.")
            + "\n"
            + fields_description
        )

    def _generate_schema(self) -> Type[BaseModel]:
        """Generate a Pydantic model dynamically based on output fields."""
        fields = {
            field["name"]: (eval(field["type"]), Field(description=field["description"]))  # nosec
            for field in self.output_fields
        }
        return create_model("DynamicOutput", **fields)

    def _load_processed_ids(self) -> set:
        """Load already processed IDs from the output file."""
        try:
            with open(self.output_path, "r") as f:
                processed_df = pd.read_json(StringIO(f.read()), lines=True)
            return set(processed_df["id"].tolist())
        except (ValueError, FileNotFoundError):
            return set()

    def _get_prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([("system", self.system_message), ("user", "{input}")])

    async def _invoke_llm(self, input_text: str, _id: str) -> Dict:
        start_time = datetime.now(tz=timezone.utc).isoformat()
        structured_llm = self.llm.with_structured_output(self.schema)
        response = await structured_llm.ainvoke(input_text, config={"callbacks": [self.langfuse_handler]})
        response = response.dict()
        response["id"] = _id
        response["timestamp"] = start_time
        response["model"] = self.model_name
        response["temperature"] = self.temperature

        return response

    async def _process_batch(self, batch: List[str], batch_ids: List[str], prompt_template: str) -> List[Dict]:
        tasks = []
        for idx, text_data in enumerate(batch):
            formatted_prompt = prompt_template.format(input=text_data)
            tasks.append(self._invoke_llm(formatted_prompt, batch_ids[idx]))

        return await asyncio.gather(*tasks)

    async def process_text_data(
        self, text_data: Dict[str, str], batch_size: int = 10, sleep_time: float = 0.5
    ) -> None:
        """
        Process the text data using the language model.

        Args:
            text_data (Dict[str, str]): A dictionary of text data with IDs as keys and text as values.
            batch_size (int): The batch size for processing the text data.
            sleep_time (float): The time to sleep between each batch.
        """
        prompt_template = self._get_prompt_template()
        processed_ids = self._load_processed_ids()

        # Filter out already processed text data
        text_data = {k: v for k, v in text_data.items() if k not in processed_ids}

        # Message if all data has been processed
        if not text_data:
            logger.info("All data has already been processed.")
            return

        _text_data = list(text_data.values())
        _ids = list(text_data.keys())

        num_batches = math.ceil(len(_text_data) / batch_size)
        for i in range(num_batches):
            logger.info(f"Processing batch {i + 1}/{num_batches}")
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch = _text_data[start_idx:end_idx]
            batch_ids = _ids[start_idx:end_idx]

            responses = await self._process_batch(batch, batch_ids, prompt_template)

            with open(self.output_path, "a") as f:
                for response in responses:
                    f.write(json.dumps(response) + "\n")

            if i < num_batches - 1:
                await asyncio.sleep(sleep_time)

    def run(self, text_data: Dict[str, str], batch_size: int = 10, sleep_time: float = 0.5) -> None:
        """Run the processing of the text data."""
        if asyncio.get_event_loop().is_running():
            # If already in an event loop (e.g., Jupyter Notebook), use `create_task`
            return asyncio.create_task(self.process_text_data(text_data, batch_size, sleep_time))
        else:
            # For non-notebook environments
            asyncio.run(self.process_text_data(text_data, batch_size, sleep_time))


def generate_system_message(config: Union[str, dict]) -> str:
    """
    Generate a system message using the provided configuration.

    Args:
        config (dict or str): Configuration as a dictionary or file path to a YAML config.

    Returns:
        str: The generated system message.
    """
    if isinstance(config, str):
        with open(config, "r") as file:
            config = yaml.safe_load(file)

    # Extracting scope statements and keywords
    scope_statements = config.get("search_recipe", {}).get("scope_statements", [])
    keyword_sets = config.get("search_recipe", {}).get("keyword_sets", [])

    # Constructing scope section
    scope_section = "The scope of this task includes the following:\n" + "\n".join(
        [f"- {statement}" for statement in scope_statements]
    )

    # Constructing keyword section
    keywords_section = (
        "The relevant texts will usually combine these or highly similar keywords from each of the following sets:\n"
    )
    for keyword_set in keyword_sets:
        set_name = keyword_set.get("set_name", "Unnamed Set")
        keywords = ", ".join(keyword_set.get("keywords", []))
        keywords_section += f"- {set_name}: {keywords}\n"

    # Non-exhaustiveness disclaimer
    disclaimer = (
        "Use the sentences and keywords given above, to determine "
        "if the text provided by the user is relevant. Note that the lists of keywords"
        " and sentences are not exhaustive, but provide the main criteria for relevance."
        " Determine also other information from the text as defined below."
    )

    # Combining all parts
    system_message = (
        "Determine whether the text provided by the user is relevant based on the defined scope.\n\n"
        + scope_section
        + "\n\n"
        + keywords_section
        + "\n\n"
        + disclaimer
    )

    return system_message
