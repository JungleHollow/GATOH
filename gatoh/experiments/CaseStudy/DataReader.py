from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import polars as pl

from gatoh.agents.agents import Agent
from gatoh.graphs.graphs import Graph, GraphEdge, GraphNode
from gatoh.model.model import ABModel


@dataclass
class ModelParameters:
    """
    Dataclass that is used to store the model initialisation parameters for each model that is being run in the
    experiment.
    """

    model_id: str
    max_iterations: int
    hierarchy_names: list[str]
    hierarchy_rw_distributions: list[tuple[float, float]]
    agent_opinion_rw: tuple[float, float]
    silencing_threshold: float
    negation_threshold: float
    radicalisation_threshold: float
    suppress_warnings: bool
    save_dir: str
    data_file: str

    def __init__(
        self,
        parameters_dict: dict[str, Any],
    ) -> None:
        self.model_id = parameters_dict["model_id"]
        self.hierarchy_names = deepcopy(parameters_dict["hierarchies"])
        self.hierarchy_rw_distributions = deepcopy(parameters_dict["hierarchy_rw"])
        self.max_iterations = parameters_dict["iterations"]

        if "agent_opinion_rw" in parameters_dict.keys():
            self.agent_opinion_rw = deepcopy(parameters_dict["agent_opinion_rw"])
        else:
            self.agent_opinion_rw = TEST_PARAMETERS["DEFAULT"]["agent_opinion_rw"]

        if "silencing_threshold" in parameters_dict.keys():
            self.silencing_threshold = parameters_dict["silencing_threshold"]
        else:
            self.silencing_threshold = TEST_PARAMETERS["DEFAULT"]["silencing_threshold"]

        if "negation_threshold" in parameters_dict.keys():
            self.negation_threshold = parameters_dict["negation_threshold"]
        else:
            self.negation_threshold = TEST_PARAMETERS["DEFAULT"]["negation_threshold"]

        if "radicalisation_threshold" in parameters_dict.keys():
            self.radicalisation_threshold = parameters_dict["radicalisation_threshold"]
        else:
            self.radicalisation_threshold = TEST_PARAMETERS["DEFAULT"][
                "radicalisation_threshold"
            ]

        if "save_dir" in parameters_dict.keys():
            self.save_dir = parameters_dict["save_dir"]
        else:
            self.save_dir = SAVEDIRS[self.model_id]

        if "data_file" in parameters_dict.keys():
            self.data_file = parameters_dict["data_file"]
        else:
            self.data_file = SAVEFILES[self.model_id]


class DataReader:
    """
    A class that will be used to create and handle an appropriate GATOH model from the given agent and
    social hierarchy graph subdirectories.

    The expected agent and graph inputs for this class are structured according to the outputs from the
    ResponseParser script.
    """

    def __init__(
        self,
        agent_paths: dict[str, str],
        graph_paths: dict[str, str],
        initial_hierarchies: list[str],
        opinion_paths: dict[str, str] | None = None,
        test_parameters: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """
        :param agent_paths: A <model name : path> mapping pointing to the subdirectories at which each model's Agent objects are saved.
        :param graph_paths: A <model name: path> mapping pointing to the subdirectories at which each model's Graph objects are saved.
        :param initial_hierarchies: A list of the social hierarchies that will be present in the initial data passed to the reader.
        :optional param opinion_paths: An optional <model name: path> mapping pointing to csv files containing dependant variable data (for model validation after running).
        :optional param test_parameters: An optional mapping of <model: parameters> specifying explicit initialisation parameters for each model.
        """
        self.agent_paths: dict[str, str] = agent_paths
        self.graph_paths: dict[str, str] = graph_paths

        self.initial_hierarchies: list[str] = initial_hierarchies
        self.hierarchy_influences: dict[str, dict[str, Any]] = {}

        self.opinion_paths: dict[str, str] | None = opinion_paths
        self.opinion_dfs: dict[str, pl.DataFrame] = {}

        self.agent_objects: list[Agent] = []
        self.graph_objects: list[Graph] = []

        if self.opinion_paths:
            for key, value in self.opinion_paths.items():
                with open(value, "r") as csv_file:
                    self.opinion_dfs[key] = pl.read_csv(csv_file)

        self.model_params: dict[str, ModelParameters] = {}
        if test_parameters:
            for key, value in test_parameters.items():
                if key == "DEFAULT":
                    continue
                else:
                    param_struct: ModelParameters = ModelParameters(value)
                    self.model_params[key] = deepcopy(param_struct)
        else:
            for key in agent_paths.keys():
                param_dict: dict[str, Any] = {
                    "model_id": key,
                    "hierarchies": deepcopy(TEST_PARAMETERS["DEFAULT"]["hierarchies"]),
                    "hierarchy_rw": deepcopy(
                        TEST_PARAMETERS["DEFAULT"]["hierarchy_rw"]
                    ),
                    "iterations": TEST_PARAMETERS["DEFAULT"]["iterations"],
                }
                param_struct = ModelParameters(param_dict)
                self.model_params[key] = deepcopy(param_struct)


if __name__ == "__main__":
    SAVEDIRS: dict[str, str] = {
        "NONMN": "./gatoh/experiments/CaseStudy/Results/NONMN",
        "MINNG": "./gatoh/experiments/CaseStudy/Results/MINNG",
    }

    SAVEFILES: dict[str, str] = {
        "NONMN": "./gatoh/experiments/CaseStudy/Results/NONMN_model_variables.csv",
        "MINNG": "./gatoh/experiments/CaseStudy/Results/MINNG_model_variables.csv",
    }

    AGENT_PATHS: dict[str, str] = {
        "NONMN": "./gatoh/experiments/CaseStudy/Agents/NONMN_Agents",
        "MINNG": "./gatoh/experiments/CaseStudy/Agents/MINNG_Agents",
    }

    GRAPH_PATHS: dict[str, str] = {
        "NONMN": "./gatoh/experiments/CaseStudy/Graphs/NONMN_Graphs",
        "MINNG": "./gatoh/experiments/CaseStudy/Graphs/MINNG_Graphs",
    }

    BASE_HIERARCHIES: list[str] = [
        "Age",
        "Gender",
        "Friends",
        "Family",
        "Cultural",
        "Religious",
        "Geographical",
        "Social",
    ]

    HIERARCHY_RW: dict[str, tuple[float, float]] = {
        "Age": (0.0, 0.04),
        "Gender": (0.0, 0.02),
        "Friends": (0.0, 0.05),
        "Family": (0.0, 0.01),
        "Religious": (0.0, 0.1),
        "Cultural": (0.0, 0.15),
        "Geographical": (0.0, 0.0),
        "Social": (0.0, 0.1),
    }

    # The relevant parameters that are defined for the model instances
    TEST_PARAMETERS: dict[str, dict[str, Any]] = {
        "NONMN": {
            "model_id": "NONMN",
            "iterations": 100,
            "hierarchies": deepcopy(BASE_HIERARCHIES),
            "hierarchy_rw": deepcopy(HIERARCHY_RW),
        },
        "MINNG": {
            "model_id": "MINNG",
            "iterations": 100,
            "hierarchies": deepcopy(BASE_HIERARCHIES),
            "hierarchy_rw": deepcopy(HIERARCHY_RW),
        },
        "DEFAULT": {
            "iterations": 100,
            "hierarchies": deepcopy(BASE_HIERARCHIES),
            "hierarchy_rw": deepcopy(HIERARCHY_RW),
            "agent_opinion_rw": (0.0, 0.05),
            "silencing_threshold": 0.95,
            "negation_threshold": 0.999,
            "radicalisation_threshold": 0.99,
            "suppress_warnings": True,
        },
    }

    # Specify the dependant variable CSV paths here:
    OPINION_PATHS: dict[str, str] = {}

    data_reader: DataReader = DataReader()
