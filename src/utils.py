import pandas as pd

from discovery_utils.utils.io import safe_yaml_load
from discovery_utils.utils.llm import batch_check
from src import PROJECT_DIR
from src import logging


PROJECT_NAME = "2025_02_MS_asf"
OUTPUT_DIR = PROJECT_DIR / f"data/{PROJECT_NAME}"

# CONFIG_NAMES = [
#     "bioenergy",
#     "biomass_heating",
#     "built_environment",
#     "ccus",
#     "district_heating",
#     "energy_efficiency",
#     "energy_grid",
#     "geothermal_energy",
#     "green_skills",
#     "heat_pumps",
#     "heat_storage",
#     "hydrogen_energy",
#     "hydrogen_heating",
#     "micro_chp",
#     "solar_thermal",
# ]

CONFIG_NAMES = [
    # "bioenergy",
    "biomass_heating",
    # "built_environment",
    "ccus",
    # "decarbonisation_general",
    "district_heating",
    # "energy_efficiency",
    # "energy_grid",
    # "energy_storage",
    "geothermal_energy",
    # "green_skills",
    "heat_pumps",
    "heat_storage",
    "hydrogen_energy",
    "hydrogen_heating",
    "micro_chp",
    # "renewables_general",
    "solar_thermal",
    # "solar",
    # "wind"
]


CB_CATEGORIES = {
    # "decarbonisation_general",
    "energy_storage": "Battery",
    "renewables_general": "Renewable Energy",
    "solar": "Solar",
    "wind": "Wind Energy",
}


def get_config_dict(config_name: str) -> dict:
    """Find companies in a specific category from a config file"""
    config_path = str(PROJECT_DIR / f"notebooks/{PROJECT_NAME}/config_{config_name}.yaml")
    return safe_yaml_load(open(config_path))


def get_companies_from_config(CB, config: dict) -> pd.DataFrame:
    """Get companies from a config file"""
    category_name = config["search_recipe"]["category_name"]
    return CB.get_companies_in_nesta_categories("topic_labels", [category_name])


async def check_relevance(CB, selected_df: pd.DataFrame, config_name: str, config: dict) -> None:
    """Check relevance of the selected companies"""
    selected_texts_df = CB.get_organisation_text(selected_df)
    check_data = dict(zip(selected_texts_df["id"], selected_texts_df["text"]))
    system_message = batch_check.generate_relevance_check_system_message(config)

    fields = [
        {"name": "is_relevant", "type": "str", "description": "A one-word answer: 'yes' or 'no'."},
    ]

    processor = batch_check.LLMProcessor(
        output_path=str(OUTPUT_DIR / f"llm_check_MS_{config_name}.jsonl"),
        system_message=system_message,
        session_name="mission_studio",
        output_fields=fields,
    )

    await processor.run(check_data, batch_size=10, sleep_time=0.5)


from typing import List
from typing import Literal


def get_projects_in_nesta_categories(
    GTR,
    enrichment_df: pd.DataFrame,
    category_type: Literal["mission_labels", "topic_labels"],
    categories: List[str],
) -> pd.DataFrame:
    """Get all companies belonging to the provided categories"""
    matching_ids = enrichment_df.explode(category_type).query(f"{category_type} in @categories").id.to_list()  # noqa
    return GTR.projects_enriched.query("id in @matching_ids").drop_duplicates(subset="id")


def get_projects_from_config(GTR, enrichment_df, config: dict) -> pd.DataFrame:
    """Get companies from a config file"""
    category_name = config["search_recipe"]["category_name"]
    return get_projects_in_nesta_categories(GTR, enrichment_df, "topic_labels", [category_name])
