from __future__ import annotations

import unittest as ut
from typing import override

import gatoh.agents.agents as agt
import gatoh.graphs.graphs as gr
import gatoh.model.model as md

HIERARCHY_NAMES: list[str] = ["Test_1", "Test_2"]
HIERARCHY_RW_DISTRIB: list[tuple[float, float]] = [(0, 0.01), (0, 0.2)]


class TestModelCreation(ut.TestCase):
    @override
    def setUp(self) -> None:
        """
        Initialise a model object with specific parameters for testing purposes.
        """
        self.model: md.ABModel = md.ABModel(
            HIERARCHY_NAMES,
            HIERARCHY_RW_DISTRIB,
            iterations=99,
            negation_threshold=0.89,
            radicalisation_threshold=0.45,
            visualise=False,
            model_id="TEST_MODEL",
        )
        self.assertIsInstance(
            self.model, md.ABModel, "ABModel object is not being created appropriately"
        )

    def test_parameter_setting(self) -> None:
        """
        Test function which checks that all input ABmodel parameters are being assigned correctly.
        """
        self.assertEqual(
            self.model.current_iteration,
            0,
            "Current iteration not being initialised properly",
        )
        self.assertEqual(
            self.model.max_iterations, 99, "Max iterations not being stored correctly"
        )
        self.assertEqual(
            self.model.negation_threshold,
            0.89,
            "Negation threshold not being stored correctly",
        )
        self.assertEqual(
            self.model.radicalisation_threshold,
            0.45,
            "Radicalisation threshold not being stored correctly",
        )
        self.assertEqual(
            self.model.hierarchy_information,
            {
                HIERARCHY_NAMES[0]: HIERARCHY_RW_DISTRIB[0],
                HIERARCHY_NAMES[1]: HIERARCHY_RW_DISTRIB[1],
            },
            "Hierarchy information not being stored correctly as a dictionary",
        )
        self.assertFalse(
            self.model.visualise, "Visualisation flag not being stored correctly"
        )
        self.assertEqual(
            self.model.model_id, "TEST_MODEL", "Model ID not being stored correctly"
        )

    def test_add_agent(self) -> None:
        """
        Test function that checks if adding Agents to the ABModel is working correctly.
        """
        test_agent: agt.Agent = agt.Agent(
            "TEST_AGENT",
            {
                "Test_1": 0.2,
                "Test_2": 0.77,
            },
            -0.44,
            True,
            ("social", 0.05),
        )
        agent_index: int = self.model.add_agent(test_agent)
        # Index should equal 0 as no other agents have been previously added
        self.assertGreaterEqual(
            agent_index,
            0,
            "The add_agent function is reporting an incorrect index when adding the first agent",
        )
        self.assertIn(
            test_agent,
            self.model.agents,
            "The add_agent function did not correctly add the first agent to the AgentSet",
        )
        test_agent_2: agt.Agent = agt.Agent(
            "TEST_AGENT_2",
            {
                "Test_1": 0.1,
                "Test_2": 0.98,
            },
            0.3,
            False,
            ("impulsive", 0.99),
        )
        agent_2_index: int = self.model.add_agent(test_agent_2)
        # Now agent_2_index should equal 1 as an agent was previously added
        self.assertGreaterEqual(
            agent_2_index,
            1,
            "The add_agent function is reporting an incorrect index when adding subsequent agents",
        )
        self.assertIn(
            test_agent_2,
            self.model.agents,
            "The add_agent function did not correctly add subsequent agents to the AgentSet",
        )
        # Check that the original agent is still in the AgentSet
        self.assertIn(
            test_agent,
            self.model.agents,
            "The add_agent function caused an existing agent to disappear from the AgentSet",
        )

    def test_add_graph(self) -> None:
        """
        Test function that checks if adding Graphs to the ABModel is working correctly.
        """
        test_graph: gr.Graph = gr.Graph(HIERARCHY_NAMES[0], HIERARCHY_RW_DISTRIB[0])
        model_graphset: gr.GraphSet = self.model.add_graph(test_graph)
        self.assertIsInstance(
            model_graphset,
            gr.GraphSet,
            "The add_graph function's return type is not the expected one",
        )
        self.assertIn(
            test_graph,
            self.model.graphs,
            "The add_graph function is not adding the first Graph to the GraphSet",
        )
        test_graph_2: gr.Graph = gr.Graph(HIERARCHY_NAMES[1], HIERARCHY_RW_DISTRIB[1])
        model_graphset = self.model.add_graph(test_graph_2)
        self.assertIn(
            test_graph_2,
            self.model.graphs,
            "The add_graph function is not adding subsequent graphs to the GraphSet",
        )
        self.assertIn(
            test_graph,
            self.model.graphs,
            "The add_graph function is causing existing graphs to disappear from the GraphSet",
        )

    @override
    def tearDown(self) -> None:
        """
        Reset the model object to run subsequent tests.
        """
        del self.model
