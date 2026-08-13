from __future__ import annotations

import os
from multiprocessing.pool import Pool
import unittest as ut
from typing import Any, override

import gatoh.model as md
import gatoh.agents as agt
import gatoh.graphs as gr

MODEL_ID: str = "TEST_CALCULATIONS"
HIERARCHY_NAMES: list[str] = ["A", "B"]
HIERARCHY_RW_DISTRIB: dict[str, tuple[float, float]] = {
    "A": (0.0, 0.0),
    "B": (0.0, 0.1),
}

# Define 4 agents to be used in the experiment calculations
AGENTS: list[agt.Agent] = [
    agt.Agent(
        "CALC0001",
        {
            "A": 0.2,
            "B": 0.4,
        },
        0.1,
        False,
        ("social", 0.7),
    ),
    agt.Agent(
        "CALC0002",
        {
            "A": 0.8,
            "B": 0.15,
        },
        0.55,
        True,
        ("social", 0.8),
    ),
    agt.Agent(
        "CALC0003",
        {
            "A": 0.15,
            "B": 0.1,
        },
        0.05,
        False,
        ("neutral", 0.1),
    ),
    agt.Agent(
        "CALC0004",
        {
            "A": 0.2,
            "B": 0.2
        },
        -0.7,
        True,
        ("impulsive", 0.05),
    ),
]

# Define the graphs to be used for calculations
GRAPH_A: gr.Graph = gr.Graph(
    "A",
    (0.0, 0.0),
    suppress_warnings=True,
    dynamic_rels=False,
)
GRAPH_B: gr.Graph = gr.Graph(
    "B",
    (0.0, 0.1),
    suppress_warnings=True,
    dynamic_rels=False,
)

# Define the relationships for A
A_RELS: dict[str, list[Any]] = {
    "from_node": [0, 0, 0, 3, 1, 2, 1, 2, 3],
    "to_node": [1, 2, 3, 2, 3, 3, 0, 0, 0],
    "weighting": [0.4, 0.8, 0.2, 0.45, 0.75, -0.3, 0.35, 0.85, -0.1],
}
# Define the relationships for B
B_RELS: dict[str, list[Any]] = {
    "from_node": [0, 2, 1, 1],
    "to_node": [1, 1, 0, 2],
    "weighting": [0.4, 0.3, 0.45, -0.1],
}


class TestModelCalculations(ut.TestCase):
    @classmethod
    @override
    def setUpClass(cls) -> None:
        cls._model: md.ABModel = md.ABModel(
            HIERARCHY_NAMES,
            list(HIERARCHY_RW_DISTRIB.values()),
            suppress_warnings=True,
            iterations=10,
            model_id=MODEL_ID,
        )
        cls._agents: list[agt.Agent] = AGENTS
        # Make The last agent be radicalised
        cls._agents[-1].change_radicalisation(True)
        # Add all agents to A
        cls._model.add_agents_to_hierarchy(cls._agents, "A")
        # Add all agents minus the last to B
        cls._model.add_agents_to_hierarchy(cls._agents[:-1], "B")
        # Add the relationships to each hierarchy
        cls._model.add_relationships_to_hierarchy(A_RELS, "A")
        cls._model.add_relationships_to_hierarchy(B_RELS, "B")
        # Initialise the worker pool used for multiprocessed methods
        cls._pool: Pool = Pool()

    def test_calculate_aggregate_opinion(self) -> None:
        """
        Test that the model's calculate_aggregate_opinion method is working as intended.
        """
        return None

    def test_calculate_radicalisation_logodds(self) -> None:
        """
        Test that the model's calculate_radicalisation_logodds method is working as intended.
        """
        return None

    def test_calculate_layers_polarisation(self) -> None:
        """
        Test that the model's calculate_layers_polarisation method is working as intended.
        """
        return None

    def test_calculate_layers_polarisation_multi(self) -> None:
        """
        Test that the multiprocessed calculate_layers_polarisation is working as intended.
        """
        return None

    def test_calculate_density(self) -> None:
        """
        Test that the model's calculate_density method is working as intended.
        """
        return None

    def test_calculate_navigability(self) -> None:
        """
        Test that the model's calculate_navigability method is working as intended.
        """
        return None

    def test_calculate_interdependences(self) -> None:
        """
        Test that the model's calculate_interdependences method is working as intended.
        """
        return None

    def test_calculate_interdependences_multi(self) -> None:
        """
        Test that the multiprocessed calculate_interdependences is working as intended.
        """
        return None

    @classmethod
    @override
    def tearDownClass(cls) -> None:
        del cls._model, cls._agents
        cls._pool.close()
