from __future__ import annotations

import json
import random as rd
from typing import Any

import polars as pl

from src.GATOH.agents import PERSONALITIES


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
                "Gender": {"Male": [], "Female": [], "Other": []},
                "Religious": {"Yes": [], "No": []},
                "Cultural_1": {
                    "Yes": [],  # Participates in community dialogue
                    "No": [],  # Does not participate
                    "Maybe": [],  # May participate
                },
                "Cultural_2": {
                    # Favourite sports
                    "Football": [],
                    "Basketball": [],
                    "Volleyball": [],
                    "Athletics": [],
                    "Other": [],
                },
                "Cultural_3": {
                    # Favourite music genres
                    "Traditional": [],
                    "Popular": [],
                    "Relgiious": [],
                    "International": [],
                    "Varied": [],
                },
                "General": {  # "Wildcard" groupings that will affect existing relationship strengths across all hierarchies
                    "Environmental": [],
                    "Economic": [],
                },
            }

        self.survey_values: dict[str, Any]
        self.survey_question_types: dict[str, list[str]]

        with open(SURVEY_VALUES, "r") as survey_values:
            self.survey_values = json.load(survey_values)

        with open(SURVEY_TYPES, "r") as survey_types:
            self.survey_question_types = json.load(survey_types)

        self.surveys: dict[str, pl.DataFrame] = {}

        for key, value in RESPONSE_CSV.items():
            with open(value, "r") as file:
                self.surveys[key] = pl.read_csv(file)

    def generate_agents(self) -> None:
        """
        Calculates the weighting that each Agent gives to each social hierarchy. Also generates random weightings
        for those without preferential information.

        Additionally calculates or stochastically generates other important agent attributes from the given information.
        """
        dependant_total: float = float(self.survey_values["DependantTotal"])
        religious_total: float = float(self.survey_values["ReligiousTotal"])
        cultural_total: float = float(self.survey_values["CulturalTotal"])

        for community, survey in self.surveys.items():
            for agent_row in survey.iter_rows(named=True):
                agent_weightings: dict[str, float] = {}
                for hierarchy in HIERARCHIES:
                    agent_weightings[hierarchy] = 0.0

                agent_values: dict[str, Any] = {
                    "dependant_sum": 0.0,
                    "religious_sum": 0.0,
                    "cultural_sum": 0.0,
                }

                for column, value in agent_row.items():
                    if column == "AgentId":
                        agent_values["id"] = str(value)
                    elif column in self.survey_question_types["Dependant"]:
                        agent_values["dependant_sum"] += float(
                            self.survey_values[column][value]
                        )
                    elif column in self.survey_question_types["Age"]:
                        agent_values["age"] = str(value)
                    elif column in self.survey_question_types["Gender"]:
                        agent_values["gender"] = str(value)
                    elif column in self.survey_question_types["Religious"]:
                        agent_values["religious"] = str(value)
                    elif column in self.survey_question_types["ReligiousStrength"]:
                        agent_values["religious_sum"] += float(
                            self.survey_values[column][value]
                        )
                    elif column in self.survey_question_types["Cultural"]:
                        match column:
                            case "Q20":  # Willing to participate in dialogue
                                agent_values["dialogue"] = str(value)
                            case "Q22":  # Preferred sport
                                agent_values["sport"] = str(value)
                            case "Q29":  # Favourite music genre
                                agent_values["music"] = str(value)
                            case _:
                                pass
                    elif column in self.survey_question_types["CulturalStrength"]:
                        agent_values["cultural_sum"] += float(
                            self.survey_values[column][value]
                        )
                    elif column in self.survey_question_types["Time"]:
                        agent_values["time"] = str(value)
                    else:  # "General"
                        match column:
                            case "Q17":  # Specific influencing factor
                                agent_values["influencing_factor"] = str(value)
                            case "Q28":  # Preferred activity to participate in
                                agent_values["activity"] = str(value)
                            case _:
                                pass

                # Assign the Agent to all the relevant clusters
                self.hierarchy_clusters["Age"][agent_values["age"]].append(
                    agent_values["id"]
                )
                self.hierarchy_clusters["Gender"][agent_values["gender"]].append(
                    agent_values["id"]
                )
                self.hierarchy_clusters["Religious"][agent_values["religious"]].append(
                    agent_values["id"]
                )
                self.hierarchy_clusters["Cultural_1"][agent_values["dialogue"]].append(
                    agent_values["id"]
                )
                self.hierarchy_clusters["Cultural_2"][agent_values["sport"]].append(
                    agent_values["id"]
                )
                self.hierarchy_clusters["Cultural_3"][agent_values["music"]].append(
                    agent_values["id"]
                )

                # Calculate the relevant hierarchy weightings
                agent_weightings["Age"] = rd.uniform(-1.0, 1.0)
                agent_weightings["Gender"] = rd.uniform(-1.0, 1.0)
                agent_weightings["Family"] = rd.uniform(-1.0, 1.0)
                agent_weightings["Friends"] = rd.uniform(-1.0, 1.0)
                if agent_values["religious"] == "Yes":
                    # Rescale the values from [0, 1] to [-1, 1]
                    agent_weightings["Religious"] = 2.0 * (
                        agent_values["religious_sum"] / religious_total - 0.5
                    )
                else:
                    pass  # This agent does not "exist" in the religious hierarchy
                if agent_values["dialogue"] == "Yes":
                    # Rescale the values from [0, 1] to [-1, 1]
                    agent_weightings["Cultural"] = 2.0 * (
                        agent_values["cultural_sum"] / cultural_total - 0.5
                    )
                elif agent_values["dialogue"] == "Maybe":
                    # Weaker range of [-0.75, 0.75]
                    agent_weightings["Cultural"] = 1.5 * (
                        agent_values["cultural_sum"] / cultural_total - 0.5
                    )
                else:
                    # Weakest range of [-0.5, 0.5]
                    agent_weightings["Cultural"] = (
                        agent_values["cultural_sum"] / cultural_total - 0.5
                    )

                # Generate the opinion, benefit, and sociability attributes
                agent_values["opinion"] = 2.0 * (
                    agent_values["dependant_sum"] / dependant_total - 0.5
                )
                agent_values["benefit"] = rd.choice([True, False])
                agent_personality: str = rd.choice(PERSONALITIES)
                agent_susceptibility: float = rd.uniform(-1.0, 1.0)
                agent_values["sociability"] = (agent_personality, agent_susceptibility)

                # Append all the relevant attributes to self.agents
                self.agents["ids"].append(agent_values["id"])
                self.agents["weightings"].append(agent_weightings)
                self.agents["opinions"].append(agent_values["opinion"])
                self.agents["benefits"].append(agent_values["benefit"])
                self.agents["sociabilities"].append(agent_values["sociability"])

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
