from __future__ import annotations

import unittest as ut

import gatoh.graphs as gr
from gatoh.agents import Agent


class TestGraphNodes(ut.TestCase):
    def test_node_creation(self) -> None:
        """
        Test that a GraphNode is initialised correctly
        """
        test_agent: Agent = Agent("TEST0001", {"TestGraph", 1.0})
        graph_node: gr.GraphNode = gr.GraphNode(test_agent)
        self.assertEqual(
            test_agent,
            graph_node.agent,
            "GraphNode -- agent not being initialised correctly",
        )
        self.assertIsNone(
            graph_node.index,
            "GraphNode -- Existing index attribute before any assignation",
        )
        graph_node.set_index(4)
        self.assertEqual(
            graph_node.index,
            4,
            "GraphNode -- set_index not updating the node index correctly",
        )
        str_repr: str = graph_node.__str__()
        self.assertEqual(
            str_repr,
            "Agent (TEST0001) at graph node (4)",
            "GraphNode -- __str__ is not producing the expected representation",
        )
