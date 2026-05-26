from __future__ import annotations

import os
import random as rd
from copy import deepcopy
from typing import Any

import numpy as np

import gatoh.agents as agt
import gatoh.graphs as gr
import gatoh.model as md


class SocialSusceptibilityTester:
    """
    A test class that sets up models which are identical, and with each agent in the population having the same social susceptibility,
    but the shared susceptibility value differs across models.

    The models will contain shared social susceptibility values ranging from 0.0 to 1.0 at intervals of 0.2.

    Each instance will use identical random walk parameters, graph structures, agent hierarchy membership,
    and initial conditions, with only the shared social susceptibility value for all agents changing between models.
    """

    def __init__(self, existing: bool = False) -> None:
        """
        :param existing: A flag indicating if the experiment has already been run and saved models are present to inspect.
        """
        pass

    def create_agents(self) -> list[agt.Agent]:
        """
        Generates and returns the shared population of Agent objects that will be used across the instances.

        :return: A list containing the generated Agent objects.
        """
        pass

    def create_graphs(self, agents: list[agt.Agent]) -> list[gr.Graph]:
        """
        Generates and returns the shared collection of social hierarchy Graph objects that will be used across the instances.

        :param agents: The population of Agent objects that is used to create the graphs.
        :return: A list containing the generated Graph objects.
        """
        pass

    def load_models(self, existing_saves: list[str] | None = None) -> None:
        """
        Loads the model objects that have been previously saved in their respective directories.

        :param existing_saves: An optional partial list of the model names representing the models that can be loaded.
        """
        pass

    def setup_models(self, missing_saves: list[str] | None = None) -> None:
        """
        Adds the appropriate Agent and Graph objects to all the models.

        :missing_saves: An optional partial list of the model names representing models that should be setup.
        """
        pass

    def run_models(self, missing_saves: list[str] | None = None) -> None:
        """
        Runs each model instance in the tester class.

        :missing_saves: An optional partial list of the model names representing models that should be run.
        """
        pass


if __name__ == "__main__":
    # The relevant parameters that are being applied to this experiment
    TEST_PARAMETERS: dict[str, Any] = {
        "iterations": 100,
        "model_names": [
            "ZERO",
            "POINT-TWO",
            "POINT-FOUR",
            "POINT-SIX",
            "POINT-EIGHT",
            "ONE",
        ],
        "hierarchy_names": ["A", "B", "C", "D"],
        "hierarchy_rw": {
            "A": (0.0, 0.3),
            "B": (0.0, 0.1),
            "C": (0.0, 0.45),
            "D": (0.0, 0.15),
        },
        "relationship_rw": (0.0, 0.1),
        "graph_generation_alg": "small-world",
    }

    # The parameters that will be used to create the Agent population that is shared across models
    AGENT_PARAMETERS: dict[str, Any] = {
        "n_agents": 40,
        "opinions": (-0.8, 0.8),
        "relationships": (-0.8, 0.8),
        "hierarchy_weighting": (-0.6, 0.6),
        "personal_benefit": {True: 0.25, False: 0.75},
        "id_base": "EXSS",
    }

    # The save directories for each model instance
    SAVEDIRS: dict[str, str] = {
        "ZERO": "./experiments/Base/SocialSusceptibility/SocialSusceptibility_ZERO",
        "POINT-TWO": "./experiments/Base/SocialSusceptibility/SocialSusceptibility_POINT-TWO",
        "POINT-FOUR": "./experiments/Base/SocialSusceptibility/SocialSusceptibility_POINT-FOUR",
        "POINT-SIX": "./experiments/Base/SocialSusceptibility/SocialSusceptibility_POINT-SIX",
        "POINT-EIGHT": "./experiments/Base/SocialSusceptibility/SocialSusceptibility_POINT-EIGHT",
        "ONE": "./experiments/Base/SocialSusceptibility/SocialSusceptibility_ONE",
    }

    # The save paths for each model's logger output (must point to a .csv file)
    SAVEFILES: dict[str, str] = {
        "ZERO": "./experiments/Base/SocialSusceptibility/ZERO_model_variables.csv",
        "POINT-TWO": "./experiments/Base/SocialSusceptibility/POINT-TWO_model_variables.csv",
        "POINT-FOUR": "./experiments/Base/SocialSusceptibility/POINT-FOUR_model_variables.csv",
        "POINT-SIX": "./experiments/Base/SocialSusceptibility/POINT-SIX_model_variables.csv",
        "POINT-EIGHT": "./experiments/Base/SocialSusceptibility/POINT-EIGHT_model_variables.csv",
        "ONE": "./experiments/Base/SocialSusceptibility/ONE_model_variables.csv",
    }

    tester: SocialSusceptibilityTester

    # Check for existing saved models and store the relevant information
    save_dirs: list[str] = list(os.walk("./experiments/Base/SocialSusceptibility"))[0][
        1
    ]

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
        tester = SocialSusceptibilityTester()

        if len(existing_savedirs) > 0:  # At least one model exists
            tester.load_models(existing_saves=existing_savedirs)
            tester.setup_models(missing_saves=missing_savedirs)
            tester.run_models(missing_saves=missing_savedirs)
        else:
            tester.setup_models()
            tester.run_models()
    else:
        tester = SocialSusceptibilityTester(existing=True)
        tester.load_models()

    # TODO: Add the graph visualisation functions here when they have been implemented...
