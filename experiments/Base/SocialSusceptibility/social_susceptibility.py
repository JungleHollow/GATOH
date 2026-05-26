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
    pass
