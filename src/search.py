"""
Utility functions for searching datasets using keyword and vector searches.

Example usage:
```
from src import search

Search = search.SearchDataset(GTR, GTR.projects_enriched, "config.yaml")
search_df = Search.do_search()
```

The config file needs to contain the following structure:
```yaml
---
search_recipe:
    scope_statements:
        - "Scope statement 1"
        - "Scope statement 2"
    keyword_sets:
        - set_name: "Keyword Set 1"
          keywords:
            - "keyword1"
            - "keyword2"
        - set_name: "Keyword Set 2"
          keywords:
            - "keyword3"
            - "keyword4"
    ```
"""

from typing import Dict
from typing import List

import pandas as pd
import yaml

from src import logger


def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file using a safe loader.

    Args:
        config_path (str): The file path to the YAML configuration file.

    Returns:
        dict: The configuration data as a dictionary.

    Raises:
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            raise exc


class SearchDataset:
    """Search a dataset using keyword and vector searches

    This class integrates keyword and vector search techniques to identify
    relevant entries from a dataset based on provided configurations and
    search criteria.

    Attributes:
        Dataset: A dataset object providing search methods, has to be either GtrGetters or CrunchbaseGetters
        data_df (pd.DataFrame): The primary dataset table, should be either projects_enriched or
            organisations_enriched, for GtR or Crunchbase respectively
        config (dict): The loaded configuration for the search.
        _n_keyword_results (int): The maximum number of results for keyword searches.
        _n_vector_results (int): The maximum number of results for vector searches.
        keyword_matches_df (pd.DataFrame): The results of the keyword search.
        vector_matches_df (pd.DataFrame): The results of the vector search.
        final_matches_df (pd.DataFrame): The combined results of keyword and vector searches.
    """

    def __init__(
        self,
        Dataset,  # noqa
        data_df: pd.DataFrame,
        config_path: str,
    ) -> None:
        """Initialise a SearchDataset object"""
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
        """
        Construct an OR query string from a list of keywords.

        Args:
            keywords (List[str]): A list of keywords to include in the query.

        Returns:
            str: A string of keywords joined with OR, each wrapped in parentheses and single quotes.
        """
        return " OR ".join(["('{}')".format(keyword) for keyword in keywords])

    def keyword_searches(self, keyword_sets: List[Dict]) -> List[Dict]:
        """
        Perform keyword searches on the dataset.

        Args:
            keyword_sets (list): A list of dictionaries, each containing "set_name" and "keywords".

        Returns:
            list: A list of dictionaries containing set names, result DataFrames, and matched IDs.
        """
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
        """
        Combine keyword search results into a single DataFrame.

        Args:
            search_results (List[Dict]): A list of dictionaries containing search results.

        Returns:
            pd.DataFrame: A DataFrame containing combined and normalized keyword search results.
        """
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

    def vector_searches(self, scope_statements: List[str]) -> pd.DataFrame:
        """
        Perform vector searches on the dataset.

        Args:
            scope_statements (list): A list of scope statements for the vector search.

        Returns:
            pd.DataFrame: A DataFrame containing vector search results with scores.
        """
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
        """
        Combine and organize vector search results into a single DataFrame.

        Args:
            vector_searches_df (pd.DataFrame): The DataFrame containing raw vector search results.

        Returns:
            pd.DataFrame: A DataFrame containing organized vector search results.
        """
        return (
            self.data_df.query("id in @vector_searches_df.id")
            .merge(vector_searches_df, how="left", on="id")
            .sort_values("_score", ascending=False)
        )

    def final_results(self, data_keywords_df: pd.DataFrame, data_vectors_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine keyword and vector search results into a final result DataFrame.

        Args:
            data_keywords_df (pd.DataFrame): DataFrame of keyword search results.
            data_vectors_df (pd.DataFrame): DataFrame of vector search results.

        Returns:
            pd.DataFrame: A DataFrame containing combined search results with average scores.
        """
        final_ids = set(data_keywords_df.id).union(set(data_vectors_df.id))  # noqa
        final_matches_df = (
            self.data_df.query("id in @final_ids")
            .merge(data_keywords_df[["id", "_score"]], how="left", on="id")
            .rename(columns={"_score": "_score_keywords"})
            .merge(data_vectors_df[["id", "_score"]], how="left", on="id")
            .rename(columns={"_score": "_score_vectors"})
            .fillna({"_score_keywords": 0, "_score_vectors": 0})
            .assign(_score_avg=lambda df: (df._score_keywords + df._score_vectors) / 2)
            .sort_values("_score_avg", ascending=False)
            .reset_index()
        )

        return final_matches_df

    def do_search(self) -> pd.DataFrame:
        """
        Execute the entire search process and combine results from keyword and vector searches.

        This function orchestrates the search workflow by:
        1. Performing keyword searches using the sets defined in the configuration file.
        2. Combining and normalizing keyword search results into a unified DataFrame.
        3. Performing vector searches based on the scope statements defined in the configuration.
        4. Organizing vector search results into a unified DataFrame.
        5. Combining keyword and vector search results into a final DataFrame with aggregated scores.

        Scores in the final output:
            - `_score_keywords`: A normalized score calculated from the combined keyword search results.
            This is the sum of individual keyword set scores for each match, normalized to a 0–1 range.
            See `keyword_search_results` for details.
            - `_score_vectors`: A normalized score calculated from vector search results.
            This is derived from the proximity of vectors, where a smaller distance results in a higher score,
            normalized to a 0–1 range. See `vector_search_results` for details.
            - `_score_avg`: The average of `_score_keywords` and `_score_vectors`, providing a combined relevance score.

        Logs are generated to indicate the number of matches found for each search type.

        Returns:
            pd.DataFrame: A DataFrame containing the final combined search results, including the following columns:
                - `id`: Unique identifier for each entry in the dataset.
                - `_score_keywords`: Normalized relevance score from the keyword search.
                - `_score_vectors`: Normalized relevance score from the vector search.
                - `_score_avg`: Average of the keyword and vector search scores.
        """
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
