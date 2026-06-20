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

    def test_model_creation(self) -> None:
        """
        Check that the created model object is of the correct ABModel type.
        """
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
        self.assertEqual(
            self.model.suppress_warnings,
            False,
            "Default value for suppress_warnings is not being applied correctly",
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

    def test_add_agents(self) -> None:
        """
        Test function that checks if addings multiple Agents at once to the ABModel is working correctly.
        """
        agents_to_add: list[agt.Agent] = [
            agt.Agent(
                "TEST_AGENT_1",
                0.1,
                True,
                ("social", 0.1),
            ),
            agt.Agent(
                "TEST_AGENT_2",
                0.2,
                False,
                ("social", 0.2),
            ),
            agt.Agent(
                "TEST_AGENT_3",
                0.3,
                True,
                ("impulsive", 0.3),
            ),
            agt.Agent(
                "TEST_AGENT_4",
                0.99,
                True,
                ("erratic", 0.01),
            ),
        ]
        agentset: agt.AgentSet = self.model.add_agents(agents_to_add)
        self.assertIsInstance(
            agentset,
            agt.AgentSet,
            "The add_agents function's return type is not the expected one",
        )
        for idx, agent in enumerate(agents_to_add):
            self.assertIn(
                agent,
                self.model.agents,
                f"The add_agents function did not add Agent {agent.id} correctly",
            )
            agent_in_agentset: agt.Agent | None = self.model.agents.get_agent_by_id(
                agent.id
            )
            if (
                agent_in_agentset is not None
            ):  # Simply added to satisfy typing requirements
                self.assertEqual(
                    idx,
                    agent_in_agentset.index,
                    "The add_agents function is adding agent objects out of order",
                )
                self.assertEqual(
                    agent.id,
                    agent_in_agentset.id,
                    "The add_agents function is dropping the id attribute during addition",
                )
                self.assertEqual(
                    agent.opinion,
                    agent_in_agentset.opinion,
                    "The add_agents function is dropping the opinion attribute during addition",
                )
                self.assertEqual(
                    agent.personal_benefit,
                    agent_in_agentset.personal_benefit,
                    "The add_agents function is dropping the personal_benefit attribute during addition",
                )
                self.assertEqual(
                    agent.personality,
                    agent_in_agentset.personality,
                    "The add_agents function is dropping the personality attribute during addition",
                )
                self.assertEqual(
                    agent.social_susceptibility,
                    agent_in_agentset.social_susceptibility,
                    "The add_agents function is dropping the social_susceptibility attribute during addition",
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

    def test_add_graphs(self) -> None:
        """
        Test function that checks if adding multiple Graphs at once to the ABModel is working correctly.
        """
        graphs_to_add: list[gr.Graph] = [
            gr.Graph(HIERARCHY_NAMES[0], HIERARCHY_RW_DISTRIB[0]),
            gr.Graph(HIERARCHY_NAMES[1], HIERARCHY_RW_DISTRIB[1]),
        ]
        graphset: gr.GraphSet = self.model.add_graphs(
            graphs_to_add, HIERARCHY_NAMES, HIERARCHY_RW_DISTRIB
        )
        self.assertIsInstance(
            graphset, gr.GraphSet, "The add_graphs return type is not the expected one"
        )
        for graph in graphs_to_add:
            self.assertIn(
                graph,
                self.model.graphs,
                f"The add_graphs function did not add Graph {graph.name} correctly",
            )
            graph_in_graphset: gr.Graph | None = self.model.graphs.get_hierarchy(
                graph.name
            )
            if graph_in_graphset is not None:
                self.assertEqual(
                    graph.rw_params,
                    graph_in_graphset.rw_params,
                    "The add_graphs function is dropping the rw_params attribute during addition",
                )
                # Checks on further parameters are done from the graph-centric tests scripts

    def test_generate_agents(self) -> None:
        """
        Test function that checks if random Agent generation is working as intended within an ABModel.
        """
        self.model.generate_agents(
            "TEST",  # id_base
            {  # personality_probs
                "social": 0.4,
                "rational": 0.4,
                "impulsuive": 0.2,
            },
            number=8,
        )
        self.assertEqual(
            len(self.model.agents),
            8,
            "The generate_agents function is creating an unexpected number of Agents",
        )
        for agent in self.model.agents:
            self.assertStartsWith(
                agent.id,
                "TEST",
                "The generate_agents function is not applying the correct ID base",
            )
            self.assertEndsWith(
                agent.id,
                f"{agent.index:04}",
                "The generate_agents function is not numbering the IDs appropriately",
            )
            self.assertIn(
                agent.personality,
                ["social", "rational", "impulsive"],
                "The generate_agents function is creating agents with non-specified personality types",
            )
            for hierarchy in self.model.hierarchy_information.keys():
                self.assertIn(
                    hierarchy,
                    list(agent.social_weightings.keys()),
                    "The generate_agents function is not generating hierarchy weightings for all model hierarchies",
                )

    @override
    def tearDown(self) -> None:
        """
        Reset the model object to run subsequent tests.
        """
        del self.model
