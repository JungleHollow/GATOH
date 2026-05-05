from __future__ import annotations

import os
import random as rd
from copy import deepcopy
from typing import Any

import numpy as np

import src.GATOH.agents as agt
import src.GATOH.graphs as gr
import src.GATOH.model as md


class RandomWalkTester:
    """
    A test class that sets up models which use a variety of random walk distributions for their
    dynamic relationships and hierarchy weighting mechanisms.

    The main scenarios are:
        - A `base' model with identical random walk parameters for all dynamic relationships and hierarchy weightings
        - A model with identical parameters for dynamic relationships and unique parameters for hierarchy weightings
        - A model with identical parameters for hierarchy weightings and unique parameters for dynamic relationships
        - A model with unique parameters across both dynamic relationships and hierarchy weightings

    For each of these models, there will also be an exploration of the effects that different strengths of parameters
    may have on emergent behaviour; i.e. positive vs. negative distribution means, small vs. large variances,
    parameter strength tied to agent influentiality, etc.

    Each instance will use identical model parameters, graphs, and agent populations; with only the
    random walk parameters being different between them.
    """

    def __init__(self, existing: bool = False) -> None:
        """
        :param existing: A flag indicating if the tester is loading an existing experiment.
        """
        self.model_names: list[str] = TEST_PARAMETERS["model_names"]
        self.num_agents: int = AGENT_PARAMETERS["n_agents"]

        self.existing: bool = existing

        # Define the lists that will contain the shared populations of Agents and Graphs
        self.model_agents: list[agt.Agent] = []
        self.model_graphs: list[gr.Graph] = []

        # Dynamic model space
        self.models: dict[str, md.ABModel] = {}

        # Create the appropriate models
        # TODO: FINISH THE MODEL CREATION

    def create_agents(self, num_agents: int) -> list[agt.Agent]:
        """
        Generates and returns the population of Agents that will be used across all models.

        :param num_agents: The number of agents to create.
        :return: A list containing all the created Agent objects.
        """
        # TODO: IMPLEMENT THIS FUNCTION
        return []

    def create_graphs(self) -> list[gr.Graph]:
        """
        Creates the graphs that will be used across all models.

        :return: A list containing all the created Graph objects
        """
        # TODO: IMPLEMENT THIS FUNCTION
        return []

    def load_models(self, existing_saves: list[str] | None = None) -> None:
        """
        Loads the model objects that have been previously saved at their respective directories.

        :param existing_saves: An optional partial list of the model names representing the existing models that can be loaded.
        """
        if existing_saves:
            for existing_save in existing_saves:
                self.models[existing_save].load_model(SAVEDIRS[existing_save])
            return None

        for model_name in self.model_names:
            self.models[model_name].load_model(SAVEDIRS[model_name])
        return None

    def setup_models(self, missing_saves: list[str] | None = None) -> None:
        """
        Adds the appropriate Agent and Graph obkects to all models.

        :param missing_saves: An optional partial list of the model names representing models that should be setup.
        """
        # TODO: IMPLEMENT THIS FUNCTION
        pass

    def run_models(self, missing_saves: list[str] | None = None) -> None:
        """
        Runs each model in the tester class.

        :param missing_saves: An optional partial list of the model names representing models that should be run.
        """
        # TODO: IMPLEMENT THIS FUNCTION
        pass


if __name__ == "__main__":
    # The relevant parameters that are being applied in this experiment
    TEST_PARAMETERS: dict[str, Any] = {
        "iterations": 100,
        "model_names": ["BASE", "RELS", "HIER", "BOTH"],
        "shared_hierarchy_rw": {
            "A": (0.0, 1.0),
            "B": (0.0, 1.0),
            "C": (0.0, 1.0),
            "D": (0.0, 1.0),
            "E": (0.0, 1.0),
            "F": (0.0, 1.0),
        },
        "unique_hierarchy_rw": {
            "A": (0.0, 1.5),
            "B": (0.0, 0.3),
            "C": (0.0, 0.1),
            "D": (-0.1, 0.2),
            "E": (0.0, 0.01),
            "F": (0.1, 0.2),
        },
        "shared_relaitonship_rw": (0.0, 0.1),
        "hierarchy_names": ["A", "B", "C", "D", "E", "F"],
        "graph_generation_alg": "small-world",
    }

    # The parameters that will be used to create the shared population of Agents to be used across models
    AGENT_PARAMETERS: dict[str, Any] = {
        "n_agents": 100,
        "opinions": (-0.8, 0.8),
        "relationships": (-0.8, 0.8),
        "social_susceptibility": (0.2, 0.8),
        "personality": {
            "social": 0.4,
            "neutral": 0.4,
            "impulsive": 0.2,
        },
        "personal_benefit": {True: 0.25, False: 0.75},
        "id_base": "EXRW",
    }

    # The save directories for each model instance
    SAVEDIRS: dict[str, str] = {
        "BASE": "./experiments/Base/RandomWalk/RandomWalk_BASE",
        "RELS": "./experiments/Base/RandomWalk/RandomWalk_RELS",
        "HIER": "./experiments/Base/RandomWalk/RandomWalk_HIER",
        "BOTH": "./experiments/Base/RandomWalk/RandomWalk_BOTH",
    }

    # The save paths for each model's logger outputs (must point to .csv files)
    SAVEFILES: dict[str, str] = {
        "BASE": "./experiments/Base/RandomWalk/BASE_model_variables.csv",
        "RELS": "./experiments/Base/RandomWalk/RELS_model_variables.csv",
        "HIER": "./experiments/Base/RandomWalk/HIER_model_variables.csv",
        "BOTH": "./experiments/Base/RandomWalk/BOTH_model_variables.csv",
    }

    tester: RandomWalkTester

    # Check for existing saved models and store the relevant information
    save_dirs = list(os.walk("./experiments/Base/RandomWalk"))[0][1]

    directory_missing: bool = False
    existing_savedirs: list[str] = []
    missing_savedirs: list[str] = []

    for model, save_dir in SAVEDIRS.items():
        dir_name: str = deepcopy(save_dir).split("/")[-1]
        if dir_name in save_dirs:
            existing_savedirs.append(model)
        else:
            directory_missing = True
            missing_savedirs.append(model)

    if directory_missing:
        tester = RandomWalkTester()

        if len(existing_savedirs) > 0:  # At least one model exists
            tester.load_models(existing_saves=existing_savedirs)
            tester.setup_models(missing_saves=missing_savedirs)
            tester.run_models(missing_saves=missing_savedirs)
        else:
            tester.setup_models()
            tester.run_models()
    else:
        tester = RandomWalkTester(existing=True)
        tester.load_models()

    # TODO: Add the graph visualisation functions here once those have been implemented...
