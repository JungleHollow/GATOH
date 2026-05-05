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
        pass


if __name__ == "__main__":
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
    }

    # The social hierarchies that will exist within the models
    HIERARCHY_NAMES: list[str] = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
    ]
