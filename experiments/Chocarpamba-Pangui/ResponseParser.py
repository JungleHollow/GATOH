from __future__ import annotations

import json
from typing import Any

import polars as pl


class ResponseParser:
    def __init__(self) -> None:
        self.agents: dict[str, dict[str, list[Any]]] = {}
        self.graphs: dict[str, dict[str, list[Any]]] = {}
        self.hierarchy_clusters: dict[str, dict[str, dict[str, list[str]]]] = {}

        for key in RESPONSE_CSV.keys():
            self.agents[key] = {
                "ids": [],  # String IDs
                "weightings": [],  # <hierarchy: float value> dictionary weightings
                "opinions": [],  # Floats representing the initial agent opinions (range [-1, 1])
                "benefits": [],  # Boolean indicating if supporting mining is personally beneficial
                "sociabilities": [],  # (str, float) of personality type and social susceptibility strength
            }
            self.graphs[key] = {
                "hierarchies": [],  # String names
                "adj_matrices": [],  # 2D matrices indicating presence and strength of graph relationships (ranges [-1, 1])
                "rw_params": [],  # (mean, variance) defining the dynamic relationship random walk distributions
                "agents": [],  # List of IDs of all agents contained within the hierarchy
            }
            self.hierarchy_clusters[
                key
            ] = {  # All values are dictionaries that will represent <cluster, list[agent ids]> within each hierarchy
                "Age": {"18-25": [], "26-35": [], "36-45": [], "46-60": [], "60+": []},
                "Gender": {"Male": [], "Female": []},
                "Religious": {"Yes": [], "No": []},
                "Cultural": {
                    "Involved": [],  # Participates in community activities
                    "Uninvolved": [],  # Does not participate
                    # Favourite sports
                    "Football": [],
                    "Basketball": [],
                    "Volleyball": [],
                    "Athletics": [],
                    # Favourite music genres
                    "TradMusic": [],
                    "PopMusic": [],
                    "RelMusic": [],
                    "IntMusic": [],
                },
                "General": {  # "Wildcard" groupings that will affect existing relationship strengths across all hierarchies
                    "Environmental": [],
                    "Economic": [],
                },
            }

        self.survey_values: dict[str, int | str]
        self.survey_question_types: dict[str, list[str]]

        with open(SURVEY_VALUES, "r") as survey_values:
            self.survey_values = json.load(survey_values)

        with open(SURVEY_TYPES, "r") as survey_types:
            self.survey_question_types = json.load(survey_types)

        self.surveys: dict[str, pl.DataFrame] = {}

        for key, value in RESPONSE_CSV.items():
            with open(value, "r") as file:
                self.surveys[key] = pl.read_csv(file)

    def calculate_weightings(self) -> None:
        """
        Calculates the weighting that each Agent gives to each social hierarchy.

        Also generates random weightings for those without preferential information.
        """
        for community, survey in self.surveys.items():
            for agent_row in survey.iter_rows(named=True):
                agent_values: dict[str, Any] = {}

                for column, value in agent_row.items():
                    if column == "AgentId":
                        agent_values["id"] = str(value)
                    elif column in self.survey_question_types["Dependant"]:
                        pass
                    elif column in self.survey_question_types["Age"]:
                        pass
                    elif column in self.survey_question_types["Gender"]:
                        pass
                    elif column in self.survey_question_types["Religious"]:
                        pass
                    elif column in self.survey_question_types["ReligiousStrength"]:
                        pass
                    elif column in self.survey_question_types["Cultural"]:
                        pass
                    elif column in self.survey_question_types["CulturalStrength"]:
                        pass
                    elif column in self.survey_question_types["Time"]:
                        pass
                    else:  # "General"
                        pass

    def create_base_hierarchies(self) -> None:
        pass

    def write_agents(self) -> None:
        pass

    def write_graphs(self) -> None:
        pass


if __name__ == "__main__":
    RESPONSE_CSV: dict[str, str] = {
        "CHPBA": "./data/Chocarpamba.csv",
        "PANGI": "./data/Pangui.csv",
    }

    AGENT_PATHS: dict[str, str] = {
        "CHPBA": "./experiments/Chocarpamba-Pangui/CHPBA_Agents.csv",
        "PANGI": "./experiments/Chocarpamba-Pangui/PANGI_Agents.csv",
    }

    GRAPH_PATHS: dict[str, str] = {
        "CHPBA": "./experiments/Chocarpamba-Pangui/CHPBA_Graphs.csv",
        "PANGI": "./experiments/Chocarpamba-Pangui/PANGI_Graphs.csv",
    }

    HIERARCHIES: list[str] = [
        "Age",
        "Gender",
        "Friends",
        "Family",
        "Religious",
        "Cultural",
    ]

    SURVEY_VALUES: str = "./data/SurveyValues.json"
    SURVEY_TYPES: str = "./data/SurveyQuestionTypes.json"
