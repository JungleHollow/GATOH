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
