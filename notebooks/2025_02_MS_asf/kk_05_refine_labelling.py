import pandas as pd

from discovery_utils.utils import google
from discovery_utils.utils.llm import batch_check
from src import PROJECT_DIR
from src import utils as src_utils


PROJECT_NAME = "2025_02_MS_asf"
OUTPUT_DIR = PROJECT_DIR / f"data/{PROJECT_NAME}/llm_labelling"

sheet_id = "1m9_tKyJDaSy2vDWxYVP_9HlfBbGysQUV-xrb1FW3vok"
tab_name = "ukri_check"

CONFIG_NAMES = [
    # "bioenergy",
    "biomass_heating",
    # "built_environment",
    # "ccus",
    "district_heating",
    # "energy_efficiency",
    # "energy_grid",
    # "geothermal_energy",
    # "green_skills",
    "heat_pumps",
    # "heat_storage",
    # "hydrogen_energy",
    # "hydrogen_heating",
    # "micro_chp",
    # "solar_thermal",
    # "energy_storage",
    # "solar",
    # "wind"
]

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    accuracy = report["accuracy"]
    precision = report["1"]["precision"]
    recall = report["1"]["recall"]
    f1_score = report["1"]["f1-score"]
    return accuracy, precision, recall, f1_score


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    plt.show()


def get_reports(compare_df):
    reports = dict()
    reports["total"] = classification_report(compare_df["reviewer"], compare_df["is_relevant"], output_dict=True)
    for dataset in compare_df["dataset"].unique():
        reports[dataset] = classification_report(
            compare_df.loc[compare_df["dataset"] == dataset, "reviewer"],
            compare_df.loc[compare_df["dataset"] == dataset, "is_relevant"],
            output_dict=True,
        )
    return reports
    # plot_confusion_matrix(compare_df["reviewer"], compare_df["is_relevant"], labels=["yes", "no"])


if __name__ == "__main__":
    # model_name = "gpt-4o-mini"
    # temperature=0.0
    model_name = "o3-mini"
    temperature = None

    checked_gtr_df = google.access_google_sheet(sheet_id, "ukri_check")
    checked_cb_df = google.access_google_sheet(sheet_id, "crunchbase_check")

    for category in CONFIG_NAMES:
        config_dict = src_utils.get_config_dict(category)
        theme = config_dict["search_recipe"]["category_name"]

        gtr_data_df = (
            checked_gtr_df
            # replace empty strings with NaN
            .replace("", pd.NA)
            .assign(reviewer=lambda df: df["reviewer (karlis)"].combine_first(df["reviewer (will)"]))
            .query("theme == @category")
            .assign(dataset="gtr")
        )

        cb_data_df = (
            checked_cb_df
            # .assign(reviewer = lambda df: df["reviewer (karlis)"].combine_first(df["reviewer (will)"]))
            .assign(reviewer=lambda df: df["reviewer (karlis)"])
            .query("theme == @category")
            .assign(dataset="crunchbase")
        )

        data_df = pd.concat([gtr_data_df, cb_data_df], ignore_index=True).dropna(subset=["reviewer"])
        human_data_df = data_df[["theme", "id", "text", "reviewer", "dataset"]]
        llm_data_df = data_df[["id", "is_relevant"]].copy()

        check_data = dict(zip(data_df["id"], data_df["text"]))

        system_message = batch_check.generate_relevance_check_system_message(config_dict)
        system_message += """
        Mark the text as 'yes' if (one or more of the following):
        - If the technology defined by the scope above, is the main focus
        - If the technology is one of the components or activities described by the text. For example the technology could be  mentioned
            as part of a larger project or business including other technologies, or be mentioned as one of the use cases or case studies.
        - If the text describes a company and the technology is the main focus, or a part of a broader range of the company's activities and offerings.
        - If the text is about a component or critical element of the technology defined above
        - If the text describes a technology or process and explicitly mentions that it can be applied on the technology defined above to improve it's performance or efficiency.

        If the text is about heating technology but application target is not mentioned then assume it could be relevant for households or buildings (as opposed to an industrial applications).

        However, mark it as 'no' if (one or more of the following):
        - The activities, or business described in the text does not have a discernable impact on or connection with the technology.
        - If the technology is mentioned only in passing or as a minor example in a broader discussion, for example,
            in only one sentence within a long text with many sentences, or at the very end of a long description.
        - The technology is mentioned only as a negative example (eg "unlike [technology]...")
        - The text mentions heat pumps for heating swimming pools
        - The text would be better captured by one of the other categories (comma separated) mentioned in this list:
        Bioenergy (biofuels), Biomass heating, Carbon capture and storage, District heating and heat networks, Energy grid, Geothermal energy,
        Heat pumps, Hydrogen energy, Hydrogen heating, Micro CHP, Solar thermal heating, Energy storage (batteries), Solar power, Wind power
        """

        fields = [
            {"name": "explanation", "type": "str", "description": "A short, 1-sentence explanation of the answer."},
            {"name": "is_relevant", "type": "str", "description": "A one-word answer: 'yes' or 'no'."},
        ]

        file_name = str(OUTPUT_DIR / f"llm_refinement_gtr_{category}.jsonl")
        # if file_name exists, delete it
        import os

        if os.path.exists(file_name):
            os.remove(file_name)

        processor = batch_check.LLMProcessor(
            # model_name="o3-mini",
            # temperature=None,
            model_name=model_name,
            temperature=temperature,
            output_path=file_name,
            system_message=system_message,
            session_name="mission_studio",
            output_fields=fields,
        )

        processor.run(check_data, batch_size=10, sleep_time=0.3)

        compare_df = human_data_df.merge(pd.read_json(file_name, lines=True), how="left", on="id").assign(
            agreement=lambda df: df["is_relevant"] == df["reviewer"]
        )

        results_df = (
            compare_df.groupby(["dataset", "reviewer"])
            .agg(total_acc=("agreement", "mean"), support=("id", "count"))
            .reset_index()
        )
        print(config_dict["search_recipe"]["category_name"])
        print(results_df)
        print(compare_df.agreement.mean())
        reports = get_reports(compare_df)
        print(f'F1: {reports["total"]["weighted avg"]["f1-score"]}')
        print(f'yes: {reports["total"]["accuracy"]}')
