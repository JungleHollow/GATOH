from __future__ import annotations

import unittest as ut
from typing import override

import gatoh.graphs.graphs as gr
from gatoh.agents.agents import Agent


class TestGraphObjects(ut.TestCase):
    @override
    def setUp(self) -> None:
        """
        Initialise a Graph object with basic initial parameters, and a population of Agents.
        """
        self.graph: gr.Graph = gr.Graph("TestGraph", (0.0, 0.1))
        self.agents: list[Agent] = [
            Agent(
                f"TEST{i + 1:04}",
                {"TestGraph", 0.5},
                0.0,
                True,
                ("neutral", 0.5),
            )
            for i in range(22)
        ]
        self.graph.add_nodes(self.agents)

    def test_get_node(self) -> None:
        """
        Test that get_node() on a populated Graph works correctly.
        """
        node_object = self.graph.get_node(0)
        self.assertIsNotNone(
            node_object,
            "Graph -- get_node() on a valid index is not returning any object",
        )
        self.assertIsInstance(
            node_object,
            gr.GraphNode,
            "Graph -- get_node() on a valid index is not returning a GraphNode object",
        )
        self.assertEqual(
            node_object.index,
            0,
            "Graph -- get_node() is showing a mismatch between the input index and the returned GraphNode's index",
        )
