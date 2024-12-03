"""Utils for synthesising debates"""

import os

from datetime import datetime
from typing import Dict
from typing import List

import yaml

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langfuse.callback import CallbackHandler
from pydantic import BaseModel
from pydantic import Field
from src import PROJECT_DIR


SYNTHESIS_CONFIG = PROJECT_DIR / "src/synthesis_config.yaml"

langfuse_handler = CallbackHandler(
    user_id=os.environ.get("USER_EMAIL"),
    session_id=f"{datetime.today().isoformat()}",
    secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
    public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
    host=os.environ.get("LANGFUSE_HOST"),
)

GPT_MODEL = "gpt-4o-mini"
TEMPERATURE = 0

llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model_name=GPT_MODEL, temperature=TEMPERATURE)


class Debate(BaseModel):
    heading: str
    content: str


def safe_yaml_load(yaml_str: str) -> Dict:
    """Safely load a YAML string"""
    try:
        return yaml.safe_load(yaml_str)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML: {e}")


config = safe_yaml_load(open(SYNTHESIS_CONFIG).read())


def prompt_template(system_message):
    return ChatPromptTemplate.from_messages([("system", system_message), ("user", "{input}")])


# Define the Output model to enforce True/False constraint
class RelevanceOutput(BaseModel):
    is_relevant: bool = Field(description="Indicates if the debate is relevant to the given domain")


def is_debate_relevant(debate: Debate, domain: str) -> bool:
    """Use LLM to determine if a debate is strictly relevant to a given domain"""
    structured_llm = llm.with_structured_output(RelevanceOutput)
    prompt = prompt_template(f"Is this debate relevant to {domain}?")
    prompt = prompt.format(input=debate.heading)
    response = structured_llm.invoke(prompt, config={"callbacks": [langfuse_handler]})
    response = response.dict()
    response["heading"] = debate.heading
    return response


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


def summarise_debate_with_structure(debate: Debate, domain: str) -> StructuredSummaryOutput:
    """
    Use an LLM to generate a structured summary of the debate with sections for
    the essence of the debate, positives, negatives, and next steps.

    Args:
        debate (Debate): A Debate instance containing heading and content.

    Returns:
        StructuredSummaryOutput: Structured output with the summary broken down
        into essence, positives, negatives, and next steps.
    """
    # Initialize the LLM with structured output
    structured_llm = llm.with_structured_output(StructuredSummaryOutput)

    system_message = """You are an expert researcher.

    Provide a well rounded overview and summary of the positive and negatives, considering the various inputs from the speakers.

    Use focused summarisation approach by highlighting essential information, reasoning and causal links.

    Rely strictly on the provided text.

    Use British English.

    Write using short sentences.

    Do not make up any information! Only use the information provided in the debate.

    """

    structured_summary_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            (
                "user",
                """
                Provide a structured summary of the debate content:
                    1. Purpose: Summarise the main theme or purpose of the debate - what is being proposed.
                    2. Positives: List key positive aspects or arguments raised.
                    3. Negatives: List key criticisms or issues discussed.
                    4. Next Steps: List proposed follow-ups or action points.

                When writing the summary, follow these guidelines:
                Incorporate main ideas and essential information, eliminating extraneous language and focusing on critical aspects.
                Use British English.
                Don't use markdown formatting

                Whenever possible, incorporate specific evidence, facts, hard numbers, policies, programmes, or places mentioned in the debate to make the summary highly informative.
                Be very specific - if possible, retain names of specific places, numbers and organisations.

                Write short sentences (5-10 words only + add hard numbers and stats from the text of there are such).
                Limit the number of points per each section to 3-4.

                Where possible, note which speaker or speakers proposed each point (using only last names) and their party (using abbreviations: Labour: Lab, Conservatives: Con, Liberal Democrats: Lib Dem, Reform: Ref, Greens: Greens, Scottish National Party: SNP, Independent: Ind).
                Keep the speaker names at the end of the sentences in brackets.

                \n##Debates:\n{input}
             """,
            ),
        ]
    )

    # # Format the prompt with the debate content and domain
    # formatted_prompt = _prompt_template.format(
    #     input=debate.content,
    # )

    # # Prompt for structured summary
    # structured_summary_prompt = prompt_template(
    #     """Provide a structured summary of the debate content.
    #     Incorporate main ideas and essential information, eliminating extraneous language and focusing on critical aspects.
    #     Extract mentioned specific evidence, facts, hard numbers, policies, programmes or places to make the summary highly informative.
    #     Rely strictly on the provided text.
    #     Note which speaker proposed each of the point highlighted points and their party (using last names and abbreviations)
    #     Use British English.
    #     1. Purpose: Summarise the main theme or purpose of the debate - what is being proposed.
    #     2. Positives: List key positive aspects or arguments raised.
    #     3. Negatives: List key criticisms or issues discussed.
    #     4. Next Steps: List proposed follow-ups or action points.

    #     Debate Content: {input}
    #     """
    # )
    structured_summary_prompt = structured_summary_prompt.format(input=debate.content, domain=domain)

    # Get response from LLM
    structured_summary_response = structured_llm.invoke(
        structured_summary_prompt, config={"callbacks": [langfuse_handler]}
    )

    # Return the structured response as a Pydantic model
    return structured_summary_response


class CrispSummaryOutput(BaseModel):
    summary: str = Field(description="A concise summary of the debate.")


class NestaSummaryOutput(BaseModel):
    summary: str = Field(description="Concise points for Nesta to consider.")


def generate_crisp_summary(detailed_summary: StructuredSummaryOutput) -> CrispSummaryOutput:
    """
    Use an LLM to transform a detailed structured summary into a concise four-sentence summary.


    Args:
        detailed_summary (StructuredSummaryOutput): Detailed structured summary with essence, positives, negatives, and next steps.

    Returns:
        CrispSummaryOutput: A concise summary in four sentences.
    """
    # Initialize the LLM with structured output
    structured_llm = llm.with_structured_output(CrispSummaryOutput)

    # Prompt to collapse the structured summary into a concise four-sentence summary
    system_message = """Given the detailed structured summary below, write a very crisp summary in four bullet points focussing on:
        the essence of the debate, the positives, the negatives, and the next steps.
        Incorporate main ideas and essential information, eliminating extraneous language and focusing on critical aspects.
        Extract mentioned specific evidence, facts, hard numbers, policies, programmes or places to make the summary highly informative.
        Rely strictly on the provided text.
        Note which speaker proposed each of the point (using only last names) highlighted points and their party (using abbreviations)
        Detailed Structured Summary:
        {essence}
        {positives}
        {negatives}
        {next_steps}
        """
    _prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_message), ("user", "{essence}\n{positives}\n{negatives}\n{next_steps}")]
    )

    # Format the prompt with structured summary content
    formatted_prompt = _prompt_template.format(
        essence=f"Essence: {detailed_summary.essence}",
        positives=f"Positives: {'; '.join(detailed_summary.positives)}",
        negatives=f"Negatives: {'; '.join(detailed_summary.negatives)}",
        next_steps=f"Next Steps: {'; '.join(detailed_summary.next_steps)}",
    )

    # Get the response from the LLM
    crisp_summary_response = structured_llm.invoke(formatted_prompt, config={"callbacks": [langfuse_handler]})

    # Return the response as a Pydantic model
    return crisp_summary_response


def generate_direct_summary(debate: Debate, domain: str) -> CrispSummaryOutput:
    """
    Use an LLM to generate a concise four-bullet-point summary directly from the debate content.

    Args:
        debate (Debate): A Debate instance containing heading and content.
        domain (str): The domain to contextualize mission points for relevance.

    Returns:
        CrispSummaryOutput: A concise summary in four bullet points.
    """
    # Initialize the LLM with structured output
    structured_llm = llm.with_structured_output(CrispSummaryOutput)

    # Prompt to generate a direct summary
    system_message = """You are an expert researcher.

        Provide a well rounded overview and summary of the positive and negatives, considering the various inputs from the speakers.

        Use focused summarisation approach by highlighting essential information, reasoning and causal links.

        Rely strictly on the provided text.

        Use British English.

        Write using short sentences.

        Do not make up any information! Only use the information provided in the debate.

        """

    _prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            (
                "user",
                """
             Write a brief, clear summary of this debate in exactly four bullet points, using the following headings:
        - Main purpose: overview of the debate
        - Positives: key positive aspects or arguments raised
        - Negatives: key criticisms or issues discussed
        - Next steps: follow-ups or action points emerging from the debate
        (Don't use markdown formatting, just separate each section with an empty line.
        \n\n
       Also note points made in the debate that are relevant to Nesta's missions (mission descriptions provided below).
        Only highlight very specific points (eg, specific policies) that are directly relevant to one of Nesta's missions.
        Do not highlight overly generic points (eg, "this debate is about climate change and Nesta is interested in climate change").
        Note this in the summary wherever relevant (eg, in the overview, positives, negatives or next steps bullet points),
        saying "(relevant to XXX)" where XXX is the relevant Nesta mission abbreviation.

        Nesta's missions
        - A Fairer Start (AFS): Specific policies of interest: Two Child Limit, Scottish Child Payment, Minimum income guaranteed, Sure Start 2.0
        - A Healthy Life (AHL): Specific policies of interest: Policies to reduce obesity, High fat salt and sugar (HFSS) food advertising ban;
        - A Sustainable Future (ASF): Specific policies of interest: Boiler Upgrade Scheme, Low Carbon Heat Scheme, Warm Homes Plan
            GB Energy, Heat networks regulation, and anything to do with heat pumps or heat networks.
            \n\n

        Whenever possible, incorporate specific evidence, facts, hard numbers, policies, programmes, or places mentioned in the debate to make the summary highly informative.
        Be very specific - if possible, retain names of specific places, numbers and organisations.

        Write very short sentences (3-5 words only + add hard numbers and stats from the text of there are such).

             Also note which speaker or speakers proposed each point (using only last names) and their party (using abbreviations: Labour: Lab, Conservatives: Con, Liberal Democrats: Lib Dem, Reform: Ref, Greens: Greens, Scottish National Party: SNP, Independent: Ind).

        ## Example
            Purpose: Renewable energy initiatives and climate commitments in the UK.

            Positives: Strong support for solar energy projects in schools (Opher, Lab). Carbon capture plans expected to create 50,000 jobs (Jones, Lab). Commitment to insulating homes to reduce fuel poverty (Fahnbulleh, Lab). Record 131 renewable projects secured (Shanks, Lab).

            Negatives: Concerns over airport expansions conflicting with net zero targets (Wilson, Lib Dem). Criticism of previous government's inaction on energy infrastructure (Shanks, Lab). Delays in grid connections hindering renewable projects (Chowns, Green).

            Next steps: Further announcements on carbon capture and energy infrastructure expected (Shanks, Lab). Meeting with stakeholders to discuss community energy projects (Shanks, Lab). Emergency home insulation plan needed before winter (Heylings, Lib Dem).

             \n##Debates:\n{input}
             """,
            ),
        ]
    )

    # Format the prompt with the debate content and domain
    formatted_prompt = _prompt_template.format(
        input=debate.content,
    )

    # Get the response from the LLM
    direct_summary_response = structured_llm.invoke(formatted_prompt, config={"callbacks": [langfuse_handler]})

    # Return the response as a Pydantic model
    return direct_summary_response


def generate_nesta_summary(debate: Debate) -> NestaSummaryOutput:
    """
    Use an LLM to generate a concise four-bullet-point summary directly from the debate content.

    Args:
        debate (Debate): A Debate instance containing heading and content.
        domain (str): The domain to contextualize mission points for relevance.

    Returns:
        CrispSummaryOutput: A concise summary in four bullet points.
    """
    # Initialize the LLM with structured output
    structured_llm = llm.with_structured_output(NestaSummaryOutput)

    # Prompt to generate a direct summary
    system_message = """Write 1-3 points from the debate that are particularly relevant to Nesta's mission.
        Particularly any mentions of heat pumps, Boiler Upgrade Scheme, Low Carbon Heat Scheme,
        Warm Homes Plan, GB Energy and Heat networks regulation.
        Be very specific - if possible, retain names of specific places, numbers and organisations.
        Where possible note which speaker or speakers proposed each point (using only last names) and their party (using abbreviations).
        Use British English.

        We don't care about general relevance (eg, "this debate is about climate change and Nesta is interested in climate change").
        If there is no real relevance then say so.

        For more context on Nesta's mission and areas of focus:
        Nesta is specifically focussed on decarbonisation of domestic households.
        Decarbonisation of Homes: Accelerating the transition to low-carbon heating solutions.
        Affordability: Increasing the financial accessibility of low-carbon heating systems for households.
        Desirability: Enhancing the appeal of low-carbon heating options to householders.
        Ease of Installation: Improving the installation process for low-carbon heating systems to make it more user-friendly.
        Skills and Capacity: Boosting the skills and capacity in the market to meet the demand for low-carbon heating solutions

        Debate Content: {input}
        """

    _prompt_template = ChatPromptTemplate.from_messages([("system", system_message), ("user", "{input}")])

    # Format the prompt with the debate content and domain
    formatted_prompt = _prompt_template.format(
        input=debate.content,
    )

    # Get the response from the LLM
    direct_summary_response = structured_llm.invoke(formatted_prompt, config={"callbacks": [langfuse_handler]})

    # Return the response as a Pydantic model
    return direct_summary_response


def generate_simple_summary(debate: Debate, domain: str) -> CrispSummaryOutput:
    """
    Use an LLM to generate a concise four-bullet-point summary directly from the debate content.

    Args:
        debate (Debate): A Debate instance containing heading and content.
        domain (str): The domain to contextualize mission points for relevance.

    Returns:
        CrispSummaryOutput: A concise summary in four bullet points.
    """
    # Initialize the LLM with structured output
    structured_llm = llm.with_structured_output(CrispSummaryOutput)

    # Prompt to generate a direct summary
    system_message = """Write a short summary of the debate in bullet points, highlighting main themes and speakers.

        Debate Content: {input}
        """

    _prompt_template = ChatPromptTemplate.from_messages([("system", system_message), ("user", "{input}")])

    # Format the prompt with the debate content and domain
    formatted_prompt = _prompt_template.format(
        input=debate.content,
    )

    # Get the response from the LLM
    direct_summary_response = structured_llm.invoke(formatted_prompt, config={"callbacks": [langfuse_handler]})

    # Return the response as a Pydantic model
    return direct_summary_response


#     prompt = PromptTemplate(
#         template="Is this debate relevant to {domain}?\n{content}",
#         input_variables=["domain", "content"]
#     )
#     chain = LLMChain(llm=llm, prompt=prompt)
#     response = chain.invoke(domain=domain, content=debate.content)
#     return response == "Yes"

# def summarize_debates(debates: List[Debate]) -> List[SummaryOutput]:
#     """
#     Summarize the content of each debate.
#     Input: List of Debate objects
#     Output: List of SummaryOutput objects
#     """
#     summaries = []
#     for debate in debates:
#         prompt = PromptTemplate(
#             template="Summarize this debate in 1-2 concise sentences: {content}",
#             input_variables=["content"]
#         )
#         chain = LLMChain(llm=llm, prompt=prompt)
#         summary = chain.run(debate.content)
#         summaries.append(SummaryOutput(title=debate.title, summary=summary))
#     return summaries
