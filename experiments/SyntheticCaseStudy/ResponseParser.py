from __future__ import annotations

import json
import gc
import os
import pickle
import random as rd
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
import polars as pl

import gatoh.agents as agt
import gatoh.graphs as gr
import gatoh.groups as grp


@dataclass
class AgentAttributes:
    """
    A dataclass used to collect all relevant agent attributes that will be saved for Agent creation,
    and used to determine relationships during hierarchy creation.
    """

    id: str
    age: str
    gender: str
    weightings: dict[str, float]
    opinion: float
    benefit: bool
    sociability: tuple[str, float]

    def __init__(
        self,
        id: str,
        age: str,
        gender: str,
        weightings: dict[str, float],
        opinion: float,
        benefit: bool,
        sociability: tuple[str, float],
    ) -> None:
        self.id = id
        self.age = age
        self.gender = gender
        self.weightings = weightings
        self.opinion = opinion
        self.benefit = benefit
        self.sociability = sociability


class GraphsDict(TypedDict):
    """
    A helper class for typechecking ResponseParser.graphs.
    """
    hierarchies: list[str]
    adj_matrices: list[np.ndarray]
    rw_params: list[tuple[float, float]]
    agents: list[list[str]]


class ResponseParser:
    """
    A class that is used to generate GATOH agents, graphs, and groups for the synthetic case study.

    The data used corresponds to the synthetically generated `survey responses' based on the distributions
    of the specific, anonymised surveys that were carried out by research collaborators.
    """
    def __init__(self) -> None:
        self.agents: dict[str, list[AgentAttributes]] = {}
        self.agent_objects: dict[str, dict[str, agt.Agent]] = {}
        self.graphs: dict[str, GraphsDict] = {}
        self.graph_objects: dict[str, list[gr.Graph]] = {}
        self.group_objects: dict[str, list[grp.Group]] = {}
        self.group_rels: dict[str, list[tuple[int, int]]] = {}
        self.hierarchy_clusters: dict[str, dict[str, dict[str, list[str]]]] = {}

        for key in RESPONSE_CSV:
            self.agents[key] = []
            # <ID: Agent> mapping to simplify lookups later
            self.agent_objects[key] = {}
            self.graphs[key] = {
                "hierarchies": [],  # String names
                "adj_matrices": [],  # 2D matrices indicating presence and strength of graph relationships (range [-1, 1])
                "rw_params": [],  # (mean, variance) defining the dynamic relationship random walk distributions
                "agents": [],  # List of IDs of all agents contained within the hierarchy
            }
            self.graph_objects[key] = []
            self.group_objects[key] = []
            self.group_rels[key] = []
            # All values are dictionaries that will represent <cluster : list[agent IDs]> within each hierarchy
            self.hierarchy_clusters[key] = {
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
                    "Religious": [],
                    "International": [],
                    "Varied": [],
                },
                "General": {
                    # "Wildcard" groupings that will affect existing relationships strengths across all hierarchies
                    "Environmental": [],
                    "Economic": [],
                },
            }

        self.survey_values: dict[str, Any]
        self.survey_question_types: dict[str, list[str]]
        self.adjacency_matrix_values: dict[str, Any]

        with open(SURVEY_VALUES, "r") as survey_values:
            self.survey_values = json.load(survey_values)

        with open(SURVEY_TYPES, "r") as survey_types:
            self.survey_question_types = json.load(survey_types)

        with open(HIERARCHY_MATRIX_VALUES, "r") as hierarchy_values:
            self.adjacency_matrix_values = json.load(hierarchy_values)

        self.surveys: dict[str, pl.DataFrame] = {}

        for key, value in RESPONSE_CSV.items():
            with open(value, "r") as file:
                self.surveys[key] = pl.read_csv(file)

    def generate_agents(self) -> None:
        """
        Calculates the weighting that each Agent gives to each social hierarchy.

        Also generates random weightings for those without preferential information.

        Additionally calculates or stochastically generates other important agent attributes from the given information.
        """
        dependant_total: float = float(self.survey_values["DependantTotal"])
        religious_total: float = float(self.survey_values["ReligiousTotal"])
        cultural_total: float = float(self.survey_values["CulturalTotal"])

        class AgentValues(TypedDict):
            """
            A helper class for typechecking agent_values.
            """
            dependant_sum: float
            religious_sum: float
            cultural_sum: float
            age: str
            id: str
            gender: str
            religious: str
            dialogue: str
            sport: str
            music: str
            time: str
            influencing_factor: str
            activity: str
            opinion: float
            benefit: bool
            sociability: tuple[str, float]

        for community, survey in self.surveys.items():
            for agent_row in survey.iter_rows(named=True):
                agent_weightings: dict[str, float] = {}
                for hierarchy in HIERARCHIES:
                    agent_weightings[hierarchy] = 0.0

                # Define all of the necessary Agent values and initialise them as empty
                agent_values: AgentValues = {
                    "dependant_sum": 0.0,
                    "religious_sum": 0.0,
                    "cultural_sum": 0.0,
                    "age": "",
                    "id": "",
                    "gender": "",
                    "religious": "",
                    "dialogue": "",
                    "sport": "",
                    "music": "",
                    "time": "",
                    "influencing_factor": "",
                    "activity": "",
                    "opinion": 0.0,
                    "benefit": False,
                    "sociability": ("", 0.0),
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
                        if str(value) == "No response":
                            agent_values["gender"] = "Other"
                        else:
                            agent_values["gender"] = str(value)
                    elif column in self.survey_question_types["Religious"]:
                        agent_values["religious"] = str(value)
                    elif column in self.survey_question_types["ReligiousStrength"]:
                        agent_values["religious_sum"] += float(
                            self.survey_values[column][value]
                        )
                    elif column in self.survey_question_types["Cultural"]:
                        match column:
                            case "Q20":
                                # Willing to participate in dialogue
                                agent_values["dialogue"] = str(value)
                            case "Q22":
                                # Preferred sport
                                agent_values["sport"] = str(value)
                            case "Q29":
                                # Favourite music genre
                                agent_values["music"] = self.survey_values["Q29"][value]
                            case _:
                                pass
                    elif column in self.survey_question_types["CulturalStrength"]:
                        agent_values["cultural_sum"] += float(
                            self.survey_values[column][value]
                        )
                    elif column in self.survey_question_types["Time"]:
                        agent_values["time"] = str(value)
                    else:
                        # "General"
                        match column:
                            case "Q17":
                                # Specific influencing factor
                                agent_values["influencing_factor"] = str(value)
                            case "Q28":
                                # Preferred type of activity to participate in
                                agent_values["activity"] = str(value)
                            case _:
                                pass

                # Assign the Agent to all the relevant clusters
                self.hierarchy_clusters[community]["Age"][agent_values["age"]].append(agent_values["id"])
                self.hierarchy_clusters[community]["Gender"][agent_values["gender"]].append(agent_values["id"])
                self.hierarchy_clusters[community]["Religious"][agent_values["religious"]].append(agent_values["id"])
                self.hierarchy_clusters[community]["Cultural_1"][agent_values["dialogue"]].append(agent_values["id"])
                self.hierarchy_clusters[community]["Cultural_2"][agent_values["sport"]].append(agent_values["id"])
                self.hierarchy_clusters[community]["Cultural_3"][agent_values["music"]].append(agent_values["id"])

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
                    # The agent does not "exist" in the religious hierarchy
                    pass
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
                    agent_weightings["Cultural"] = agent_values["cultural_sum"] / cultural_total - 0.5

                # Generate the opinion, benefit, and sociability attributes
                agent_values["opinion"] = 2.0 * (agent_values["dependant_sum"] / dependant_total - 0.5)
                agent_values["benefit"] = rd.choice([True, False])
                agent_personality: str = agt.draw_personality()
                agent_susceptibility: float = rd.uniform(-1.0, 1.0)
                agent_values["sociability"] = (agent_personality, agent_susceptibility)

                # Append all the relevant attributes to self.agents
                agent_attributes: AgentAttributes = AgentAttributes(
                    agent_values["id"],
                    agent_values["age"],
                    agent_values["gender"],
                    agent_weightings,
                    agent_values["opinion"],
                    agent_values["benefit"],
                    agent_values["sociability"],
                )

                self.agents[community].append(deepcopy(agent_attributes))

                # Manual garbage collection
                del agent_personality, agent_susceptibility, agent_attributes
                _ = gc.collect()

        return None
