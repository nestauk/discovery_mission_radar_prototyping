"""Utils for synthesising debates"""

import os

from datetime import datetime
from typing import Dict
from typing import List
from typing import Union

import tiktoken
import yaml

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langfuse.callback import CallbackHandler
from pydantic import BaseModel
from pydantic import Field
from src import PROJECT_DIR
from src import logger


SYNTHESIS_CONFIG = PROJECT_DIR / "src/synthesis_config.yaml"
GPT_MODEL = "gpt-4o-mini"
TEMPERATURE = 0
MAX_TOKENS = 120000

langfuse_handler = CallbackHandler(
    user_id=os.environ.get("USER_EMAIL"),
    session_id=f"{datetime.today().isoformat()}",
    secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
    public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
    host=os.environ.get("LANGFUSE_HOST"),
)


def safe_yaml_load(yaml_str: str) -> Dict:
    """Safely load a YAML string"""
    try:
        return yaml.safe_load(yaml_str)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML: {e}")


llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model_name=GPT_MODEL, temperature=TEMPERATURE)
config = safe_yaml_load(open(SYNTHESIS_CONFIG).read())


class Debate(BaseModel):
    heading: str
    content: str


def tokenize_text(text: str, model_name: str = GPT_MODEL) -> int:
    """Tokenize text and return the number of tokens."""
    # Get tokenizer for the model
    tokenizer = tiktoken.encoding_for_model(model_name)
    tokens = tokenizer.encode(text)
    # Return both token count and tokens
    return len(tokens), tokens


def decode_tokens(tokens: list, model_name: str = GPT_MODEL) -> str:
    """Decode tokens back into text using the specified model's tokenizer."""
    tokenizer = tiktoken.encoding_for_model(model_name)
    text = tokenizer.decode(tokens)
    return text


def check_token_length(input: str, model_name: str = GPT_MODEL) -> bool:
    """Check token length"""
    n_tokens, tokens = tokenize_text(input, model_name)
    if n_tokens > MAX_TOKENS:
        logger.warning(f"Input text is too long ({n_tokens} tokens). Truncating to {MAX_TOKENS} tokens.")
        input = decode_tokens(tokens[:MAX_TOKENS])
    return input


def generate_output(
    input: Union[str, Debate],
    output_class: BaseModel,
    messages_config: dict,
) -> BaseModel:
    """General function for generating structured output from input text."""
    # Initialise LLM
    structured_llm = llm.with_structured_output(output_class)
    # Prepare prompt
    structured_summary_prompt = ChatPromptTemplate.from_messages(
        [("system", messages_config["system_message"]), ("user", messages_config["user_message"])]
    )
    # Check token length
    input = check_token_length(input)
    structured_summary_prompt = structured_summary_prompt.format(input=input)
    # Get response from LLM
    return structured_llm.invoke(structured_summary_prompt, config={"callbacks": [langfuse_handler]})


class StructuredSummaryOutput(BaseModel):
    purpose: str = Field(description="The main theme or purpose of the debate.")
    positives: List[str] = Field(
        description="A list of key positive aspects or arguments raised. (indicate who proposed and their party)"
    )
    negatives: List[str] = Field(
        description="A list of key criticisms or issues discussed. (indicate who proposed and their party)"
    )
    next_steps: List[str] = Field(
        description="A list of proposed follow-ups or action points (indicate who proposed and their party)."
    )


class QuoteSummaryOutput(BaseModel):
    summary: str = Field(description="The summary.")


class RelevanceClassifier(BaseModel):
    relevant: bool = Field(description="Whether the text is highly relevant to the defined topic.")


def summarise_debate_with_structure(debate: Debate) -> StructuredSummaryOutput:
    """"""
    return generate_output(
        input=debate.content,
        output_class=StructuredSummaryOutput,
        messages_config=config["debate_summary"],
    )


def summarise_quote(text: str) -> QuoteSummaryOutput:
    """"""
    return generate_output(
        input=text,
        output_class=QuoteSummaryOutput,
        messages_config=config["quote_summary"],
    )


def classify_relevance(input: str, mission: str) -> RelevanceClassifier:
    """"""
    # Initialise LLM
    structured_llm = llm.with_structured_output(RelevanceClassifier)
    # Prepare prompt
    messages_config = config["relevance_classifier"]
    structured_summary_prompt = ChatPromptTemplate.from_messages(
        [("system", messages_config["system_message"]), ("user", messages_config["user_message"])]
    )
    input = check_token_length(input)
    mission_info = config["mission_info"][mission]
    structured_summary_prompt = structured_summary_prompt.format(input=input, policy_area=mission_info)
    # Get response from LLM
    return structured_llm.invoke(structured_summary_prompt, config={"callbacks": [langfuse_handler]})
