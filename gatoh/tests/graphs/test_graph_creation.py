from __future__ import annotations

import unittest as ut
from typing import override

from rustworkx import PyDiGraph

import gatoh.graphs.graphs as gr
from gatoh.agents.agents import Agent


class TestGraphCreation(ut.TestCase):
    @override
    def setUp(self) -> None:
        """
        Initialise a Graph object with basic initial parameters, and a population of Agents.
        """
        self.graph: gr.Graph = gr.Graph("TestGraph", (0.0, 0.1))
        self.agents: list[Agent] = [
            Agent(
                f"TEST{i + 1:04}",
                {"TestGraph": 0.5},
                0.0,
                True,
                ("neutral", 0.5),
            )
            for i in range(40)
        ]

    def test_add_nodes(self) -> None:
        """
        Test that Graph.add_nodes() works correctly.
        """
        self.graph.add_nodes(self.agents)
        self.assertEqual(
            self.graph.node_count,
            40,
            "Graph -- add_nodes() is not adding the correct number of GraphNodes from an Agent iterable",
        )
        # Check for three arbitrary Agents that were added to the Graph
        self.assertTrue(
            self.graph.agent_in_graph(self.agents[4]),
            "Graph -- add_nodes() is not adding the correct Agent objects as GraphNodes",
        )
        self.assertTrue(
            self.graph.agent_in_graph(self.agents[31]),
            "Graph -- add_nodes() is not adding the correct Agent objects as GraphNodes",
        )
        self.assertTrue(
            self.graph.agent_in_graph(self.agents[15]),
            "Graph -- add_nodes() is not adding the correct Agent objects as GraphNodes",
        )

    def test_add_edges_basic(self) -> None:
        """
        Test that Graph.add_edges() with no optional attributes works correctly.
        """
        self.graph.add_nodes(self.agents)
        edges_to_add = {
            "from_node": [13, 30, 4, 23],
            "to_node": [12, 1, 6, 19],
        }
        self.graph.add_edges(edges_to_add)
        self.assertEqual(
            self.graph.edge_count,
            4,
            "Graph -- basic add_edges() is not creating the appropriate number of GraphEdges",
        )
        self.assertTrue(
            self.graph.relationship_exists(13, 12),
            "Graph -- basic add_edges() is not creating the GraphEdges between the appropriate nodes",
        )
        edge_object = self.graph.graph.edges()[0]
        self.assertEqual(
            edge_object.weighting,
            0.0,
            "Graph -- basic add_edges() is creating GraphEdges with non-zero weightings",
        )
        self.assertIsNone(
            edge_object.rw_params,
            "Graph -- basic add_edges() is creating a GraphEdge with an rw_param attribute",
        )

    def test_add_edges(self) -> None:
        """
        Test that Graph.add_edges() with explicit attributes works correctly.
        """
        self.graph.add_nodes(self.agents)
        edges_to_add = {
            "from_node": [13, 30, 4, 23],
            "to_node": [12, 1, 6, 19],
            "weighting": [0.1, 0.2, 0.3, 0.4],
            "rw_param": [(0.0, 0.1), None, None, (0.0, 0.2)],
        }
        self.graph.add_edges(edges_to_add)
        self.assertEqual(
            self.graph.edge_count,
            4,
            "Graph -- keyworded add_edges() is not creating the appropriate number of GraphEdges",
        )
        self.assertTrue(
            self.graph.relationship_exists(13, 12),
            "Graph -- keyworded add_edges() is not creating the GraphEdges between the appropriate nodes",
        )
        edge_object = self.graph.graph.edges()[0]
        self.assertEqual(
            edge_object.weighting,
            0.1,
            "Graph -- keyworded add_edges() is not setting the correct weighting for the GraphEdges",
        )
        self.assertEqual(
            edge_object.rw_params,
            (0.0, 0.1),
            "Graph -- keyworded add_edges() is not setting the correct rw_params for the GraphEdges",
        )
        other_edge = self.graph.graph.edges()[2]
        self.assertEqual(
            other_edge.weighting,
            0.3,
            "Graph -- keyworded add_edges() is not setting weightings correctly for subsequent GraphEdges",
        )
        self.assertIsNone(
            other_edge.rw_params,
            "Graph -- keyworded add_edges() is setting rw_params in GraphEdges for which None was specified",
        )

    def test_default_generate(self) -> None:
        """
        Test that Graph.generate_graph() with no optional arguments supplied works correctly.
        """
        _ = self.graph.generate_graph(self.agents)
        self.assertEqual(
            self.graph.generation_method,
            "small-world",
            "Graph -- generate_graph() is not using the correct default generation method (small-world)",
        )
        self.assertIsInstance(
            self.graph.graph,
            PyDiGraph,
            "Graph -- generate_graph() is generating a non-PyDiGraph object (small-world)",
        )
        self.assertEqual(
            self.graph.node_count,
            40,
            "Graph -- generate_graph() is generating a graph with an unexpected number of nodes (small-world)",
        )

    def test_generate_scale_free(self) -> None:
        """
        Test that Graph.generate_graph() with method = "scale-free" works correctly.
        """
        _ = self.graph.generate_graph(self.agents, method="scale-free")
        self.assertEqual(
            self.graph.generation_method,
            "scale-free",
            "Graph -- generate_graph() is not using the supplied generation method (scale-free)",
        )
        self.assertIsInstance(
            self.graph.graph,
            PyDiGraph,
            "Graph -- generate_graph() is generating a non-PyDiGraph object (scale-free)",
        )
        self.assertEqual(
            self.graph.node_count,
            40,
            "Graph -- generate_graph() is generating a graph with an unexpected number of nodes (scale-free)",
        )

    def test_generate_random(self) -> None:
        """
        Test that Graph.generate_graph() with method = "random" works correctly.
        """
        _ = self.graph.generate_graph(self.agents, method="random")
        self.assertEqual(
            self.graph.generation_method,
            "random",
            "Graph -- generate_graph() is not using the supplied generation method (random)",
        )
        self.assertIsInstance(
            self.graph.graph,
            PyDiGraph,
            "Graph -- generate_graph() is generating a non-PyDiGraph object (random)",
        )
        self.assertEqual(
            self.graph.node_count,
            40,
            "Graph -- generate_graph() is generating a graph with an unexpected number of nodes (random)",
        )

    def test_generate_blockmodel(self) -> None:
        """
        Test that Graph.generate_graph() with method = "blockmodel" works correctly.
        """
        _ = self.graph.generate_graph(self.agents, method="blockmodel")
        self.assertEqual(
            self.graph.generation_method,
            "blockmodel",
            "Graph -- generate_graph() is not using the supplied generation method (blockmodel)",
        )
        self.assertIsInstance(
            self.graph.graph,
            PyDiGraph,
            "Graph -- generate_graph() is generating a non-PyDiGraph object (blockmodel)",
        )
        self.assertEqual(
            self.graph.node_count,
            40,
            "Graph -- generate_graph() is generating a graph with an unexpected number of nodes (blockmodel)",
        )

    @override
    def tearDown(self) -> None:
        """
        Reset the Graph object to run subsequent tests.
        """
        del self.graph, self.agents
