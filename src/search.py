from typing import Dict
from typing import List

import pandas as pd
import yaml

from src import logger


def load_config(config_path: str) -> dict:
    """Load yaml config using safe loader"""
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            raise exc


class SearchDataset:
    """Search dataset using keyword and vector searches"""

    def __init__(
        self,
        Dataset,  # noqa
        data_df: pd.DataFrame,
        config_path: str,
    ) -> None:
        """Initialise search object"""
        self.Dataset = Dataset
        self.data_df = data_df
        self.config = load_config(config_path)
        self._n_keyword_results = 100000
        self._n_vector_results = 5000
        self.keyword_matches_df = None
        self.vector_matches_df = None
        self.final_matches_df = None

    @staticmethod
    def create_OR_query(keywords: List[str]) -> str:
        """Add single quotes and parentheses around each term and join with OR"""
        return " OR ".join(["('{}')".format(keyword) for keyword in keywords])

    def keyword_searches(self, keyword_sets: list) -> list:
        """Perform keyword searches"""
        search_results = []
        for keyword_set in keyword_sets:
            set_name = keyword_set["set_name"].replace(" ", "_").lower()
            query = self.create_OR_query(keyword_set["keywords"])
            search_results_df = self.Dataset.text_search(query, n_results=self._n_keyword_results).rename(
                columns={"_score": f"_score_{set_name}"}
            )
            search_results.append(
                {
                    "set_name": set_name,
                    "df": search_results_df,
                    "id": set(search_results_df["id"]),
                }
            )
        return search_results

    def keyword_search_results(self, search_results: List[Dict]) -> pd.DataFrame:
        """Combine search results into a single dataframe"""
        keyword_matches_ids = search_results[0]["id"]
        for result in search_results[1:]:
            keyword_matches_ids = keyword_matches_ids.intersection(result["id"])

        data_keywords_df = self.data_df.query("id in @keyword_matches_ids")
        for result in search_results:
            set_name = result["set_name"]
            data_keywords_df = data_keywords_df.merge(result["df"][["id", f"_score_{set_name}"]], on="id", how="left")
        keyword_set_names = [result["set_name"] for result in search_results]
        data_keywords_df = (
            data_keywords_df.fillna(
                {f"_score_{set_name}": 0 for set_name in self.config["search_recipe"]["keyword_sets"]}
            )
            .assign(_score=lambda df: df[[f"_score_{set_name}" for set_name in keyword_set_names]].sum(axis=1))
            # Normalise score
            .assign(_score=lambda df: (df._score - df._score.min()) / (df._score.max() - df._score.min()))
            .sort_values("_score", ascending=False)
            .reset_index()
        )

        return data_keywords_df

    def vector_searches(self, scope_statements: list) -> pd.DataFrame:
        """Perform vector searches"""
        search_results = []
        for statement in scope_statements:
            search_results.append(
                self.Dataset.vector_search(statement, n_results=self._n_vector_results).assign(statement=statement)
            )

        search_results = (
            pd.concat(search_results, ignore_index=True)
            .groupby("id")
            .agg(
                n_counts=("id", "count"),
                _distance=("_distance", "min"),
            )
            .assign(
                _score=lambda df: 1 - (df._distance - df._distance.min()) / (df._distance.max() - df._distance.min())
            )
            .drop(columns="_distance")
            .reset_index()
        )
        return search_results

    def vector_search_results(self, vector_searches_df: pd.DataFrame) -> pd.DataFrame:
        """Organise vector search results into a single dataframe"""
        return (
            self.data_df.query("id in @vector_searches_df.id")
            .merge(vector_searches_df, how="left", on="id")
            .sort_values("_score", ascending=False)
        )

    def final_results(self, data_keywords_df: pd.DataFrame, data_vectors_df: pd.DataFrame) -> pd.DataFrame:
        """Combine both searches"""
        final_ids = set(data_keywords_df.id).union(set(data_vectors_df.id))  # noqa
        final_matches_df = (
            self.data_df.query("id in @final_ids")
            .merge(data_keywords_df[["id", "_score"]], how="left", on="id")
            .rename(columns={"_score": "_score_keywords"})
            .merge(data_vectors_df[["id", "_score"]], how="left", on="id")
            .rename(columns={"_score": "_score_vectors"})
            .fillna({"_score_keywords": 0, "_score_vectors": 0})
            .assign(_score=lambda df: (df._score_keywords + df._score_vectors) / 2)
            .sort_values("_score", ascending=False)
            .reset_index()
        )

        return final_matches_df

    def do_search(self) -> pd.DataFrame:
        """Perform search"""
        keyword_searches = self.keyword_searches(self.config["search_recipe"]["keyword_sets"])
        data_keywords_df = self.keyword_search_results(keyword_searches)
        logger.info(f"Keyword search found {len(data_keywords_df)} matches")
        vector_searches_df = self.vector_searches(self.config["search_recipe"]["scope_statements"])
        data_vectors_df = self.vector_search_results(vector_searches_df)
        logger.info(f"Vector search found {len(data_vectors_df)} matches")
        final_matches_df = self.final_results(data_keywords_df, data_vectors_df)

        self.keyword_matches_df = data_keywords_df
        self.vector_matches_df = data_vectors_df
        self.final_matches_df = final_matches_df

        return final_matches_df
