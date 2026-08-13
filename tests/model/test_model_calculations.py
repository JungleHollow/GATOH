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
        aggregate_opinion: float = self._model.calculate_aggregate_opinion()
        # Worked example:
        #   aggregate_opinion = sum(agent opinions) / number of agents
        #                     = sum([0.1, 0.55, 0.05, -0.7]) / 4
        #                     = 0.0 / 4
        #                     = 0.0
        self.assertIsInstance(
            aggregate_opinion,
            float,
            "ABModel -- calculate_aggregate_opinion is not returning a float value",
        )
        self.assertAlmostEqual(
            aggregate_opinion,
            0.0,
            5,
            "ABModel -- calculate_aggregate_opinion is not calculating the correct value",
        )

    def test_calculate_aggregate_opinion_deviation(self) -> None:
        """
        Test that the model's calculate_aggregate_opinion_deviation method is working as intended.
        """
        aggregate_opinion_sdev: float = self._model.calculate_aggregate_opinion_deviation()
        # Worked example:
        #   SD(aggregate_opinion) = sqrt(avg((x - mean)^2))
        #   mean == aggregate_opinion = 0.0
        #   Therefore, sqrt(avg((x - 0.0)^2)) for all agent opinions x
        #   = sqrt(avg([0.1^2, 0.55^2, 0.05^2, -0.7^2]))
        #   = sqrt(avg([0.01, 0.3025, 0.0025, 0.49]))
        #   = sqrt(0.20125)
        #   = 0.448608961123
        self.assertIsInstance(
            aggregate_opinion_sdev,
            float,
            "ABModel -- calculate_aggregate_opinion_deviation is not returning a float value",
        )
        self.assertAlmostEqual(
            aggregate_opinion_sdev,
            0.448608961123,
            5,
            "ABModel -- calculate_aggregate_opinion_deviation is not calculating the correct value",
        )

    def test_calculate_radicalisation_logodds(self) -> None:
        """
        Test that the model's calculate_radicalisation_logodds method is working as intended.
        """
        radicalisation_logodds: float = self._model.calculate_radicalisation_logodds()
        # Worked example:
        #   radicalisation_logodds = log(radicalisation_p / (1.0 - radicalisation_p))
        #   radicalisation_p = count(radicalised) / len(agents) = 1 / 4 = 0.25
        #   Therefore, radicalisation_logodds = log(0.25 / (1.0 - 0.25))
        #   = log(0.25 / 0.75)
        #   = -0.47712125472
        self.assertIsInstance(
            radicalisation_logodds,
            float,
            "ABModel -- calculate_radicalisation_logodds is not returning a float value",
        )
        self.assertAlmostEqual(
            radicalisation_logodds,
            -0.47712125472,
            5,
            "ABModel -- calculate_radicalisation_logodds is not calculating the correct value",
        )

    def test_calculate_layers_polarisation(self) -> None:
        """
        Test that the model's calculate_layers_polarisation method is working as intended.
        """
        layers_polarisation: dict[str, float] = self._model.calculate_layers_polarisation()
        # Worked example:
        # --- Equation ---
        #   polarisation = 1 / K(K-1) * sum((d_{ij} - y)^2) for all distances i->j in the layer where i != j
        # --- Layer A ----
        #   polarisation = 1/4(3) * sum((d_{ij} - 0.633333333)^2)
        #                = 1/12 * sum(-0.18333^2, -0.58333^2, 0.16666^2, -0.18333^2, -0.13333^2,
        #                             0.61666^2, -0.58333^2, -0.18333^2, 0.11666^2, 0.16666^2,
        #                             0.61666^2, 0.11666^2)
        #                = 1/12 * sum(0.03361, 0.34027, 0.02777, 0.03361, 0.01777, 0.38027,
        #                             0.34027, 0.03361, 0.01361, 0.02777, 0.38027, 0.01361)
        #                = 1/12 * 1.6425
        #                = 0.136875
        # --- Layer B ---
        #   polarisation = 1/3(2) * sum((d_{ij} - 0.33333)^2)
        #                = 1/6 * sum(0.11666^2, -0.28333^2, 0.11666^2, 0.16666^2, -0.28333^2, 0.16666^2)
        #                = 1/6 * sum(0.01361, 0.08027, 0.01361, 0.02777, 0.08027, 0.02777)
        #                = 1/6 * 0.24333
        #                = 0.040555
        self.assertIsInstance(
            layers_polarisation,
            dict,
            "ABModel -- calculate_layers_polarisation is not returning a dictionary",
        )
        for hierarchy in HIERARCHY_NAMES:
            self.assertIn(
                hierarchy,
                layers_polarisation.keys(),
                "ABModel -- calculate_layers_polarisation is not calculating a value for every layer in the model",
            )
        self.assertAlmostEqual(
            layers_polarisation["A"],
            0.136875,
            5,
            "ABModel -- calculate_layers_polarisation is not calculating a value correctly for a layer",
        )
        self.assertAlmostEqual(
            layers_polarisation["B"],
            0.04055555,
            5,
            "ABModel -- calculate_layers_polarisation is not calculating a value correctly for a layer",
        )

    def test_calculate_layers_polarisation_multi(self) -> None:
        """
        Test that the multiprocessed calculate_layers_polarisation is working as intended.
        """
        layers_polarisation: dict[str, float] = self._model.calculate_layers_polarisation(worker_pool=self._pool)
        # Worked example:
        # --- Equation ---
        #   polarisation = 1 / K(K-1) * sum((d_{ij} - y)^2) for all distances i->j in the layer where i != j
        # --- Layer A ----
        #   polarisation = 1/4(3) * sum((d_{ij} - 0.633333333)^2)
        #                = 1/12 * sum(-0.18333^2, -0.58333^2, 0.16666^2, -0.18333^2, -0.13333^2,
        #                             0.61666^2, -0.58333^2, -0.18333^2, 0.11666^2, 0.16666^2,
        #                             0.61666^2, 0.11666^2)
        #                = 1/12 * sum(0.03361, 0.34027, 0.02777, 0.03361, 0.01777, 0.38027,
        #                             0.34027, 0.03361, 0.01361, 0.02777, 0.38027, 0.01361)
        #                = 1/12 * 1.6425
        #                = 0.136875
        # --- Layer B ---
        #   polarisation = 1/3(2) * sum((d_{ij} - 0.33333)^2)
        #                = 1/6 * sum(0.11666^2, -0.28333^2, 0.11666^2, 0.16666^2, -0.28333^2, 0.16666^2)
        #                = 1/6 * sum(0.01361, 0.08027, 0.01361, 0.02777, 0.08027, 0.02777)
        #                = 1/6 * 0.24333
        #                = 0.040555
        self.assertIsInstance(
            layers_polarisation,
            dict,
            "ABModel -- multiprocessed calculate_layers_polarisation is not returning a dictionary",
        )
        for hierarchy in HIERARCHY_NAMES:
            self.assertIn(
                hierarchy,
                layers_polarisation.keys(),
                "ABModel -- multiprocessed calculate_layers_polarisation is not calculating a value for every layer in the model",
            )
        self.assertAlmostEqual(
            layers_polarisation["A"],
            0.136875,
            5,
            "ABModel -- multiprocessed calculate_layers_polarisation is not calculating a value correctly for a layer",
        )
        self.assertAlmostEqual(
            layers_polarisation["B"],
            0.04055555,
            5,
            "ABModel -- multiprocessed calculate_layers_polarisation is not calculating a value correctly for a layer",
        )

    def test_calculate_density(self) -> None:
        """
        Test that the model's calculate_density method is working as intended.
        """
        density: float = self._model.calculate_density()
        # Worked example:
        #   D = l / sum((n(n-1))) for all layer graph node counts 'n'
        #   where l is the total number of existing relationships across all layers
        #   Therefore D = 13 / (4(3) + 3(2))
        #               = 13 / (12 + 6)
        #               = 13 / 18
        #               = 0.72222
        self.assertIsInstance(
            density,
            float,
            "ABModel -- calculate_density is not returning a float value",
        )
        self.assertAlmostEqual(
            density,
            0.722222222,
            5,
            "ABModel -- calculate_density is not calculating the correct value"
        )

    def test_calculate_navigability(self) -> None:
        """
        Test that the model's calculate_navigability method is working as intended.
        """
        # Calculating navigability from Agent 0 in Graph A to Agent 1 in Graph A:
        navigability: float = self._model.calculate_navigability((0, 0), (1, 0))
        self.assertIsInstance(
            navigability,
            float,
            "ABModel -- calculate_navigability is not returning a float value",
        )
        # Worked example:
        # ---
        # Navigability = -log_{2}(sum(P[p(s,t)])) for all shortest paths p(s,t) from s to t
        # P[p(s,t)] = w(s->first node)/(out degree of s) * product(w(j -> j + 1)/(out degree of j - 1)) for all nodes
        #   j that make up the shortest path
        # ---
        # A(0) -> A(1) exists as a direct path, but B(0) -> B(1) also exists, therefore:
        #     P[p(s,t)] = w(0 -> 1) / (out degree of 0)
        #     For A(0) -> A(1), this = 0.4 / 3
        #     For B(0) -> B(1), this = 0.4 / 1 = 0.4
        # ---
        # Following, navigability = -log_{2}(sum([0.4 / 3, 0.4]))
        #    = -log_{2}(1.6 / 3)
        #    = log_{2}(3 / 1.6)
        #    = log_{2}(1.875)
        #    = 0.90689
        self.assertAlmostEqual(
            navigability,
            0.90689,
            5,
            "ABModel -- calculate_navigabitliy is not calculating the correct value",
        )

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
