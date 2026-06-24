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
        Initialise a Graph object with basic initial parameters.
        """
        self.graph: gr.Graph = gr.Graph("Test Graph", (0.0, 0.1))

    def test_node_creation(self) -> None:
        """
        Create a GraphNode object with some arbitrary Agent and check that it has been
        initialised appropriately.
        """
        test_agent: Agent = Agent("TEST0001", {"Test Graph": 0.5})
        graph_node: gr.GraphNode = gr.GraphNode(test_agent)
        self.assertEqual(
            test_agent,
            graph_node.agent,
            "GraphNode object is not assigning its contained Agent correctly",
        )
        self.assertIsNone(
            graph_node.index,
            "GraphNode has an existing index attribute before any assignation",
        )
        graph_node.set_index(4)
        self.assertEqual(
            graph_node.index,
            4,
            "GraphNode set_index is not correctly updating the object's index",
        )
        node_str_repr: str = graph_node.__str__()
        self.assertEqual(
            node_str_repr,
            "Agent (TEST0001) at graph node (4)",
            "GraphNode __str__ representation is not producing the expected output",
        )

    def test_edge_creation(self) -> None:
        """
        Create a GraphEdge object with some arbitrary properties and check that it has been\
        initialised correctly.
        """
        graph_edge: gr.GraphEdge = gr.GraphEdge(
            "Test Graph",
            13,
            12,
            weighting=0.44,
            rw_params=(0.0, 0.1),
        )
        self.assertEqual(
            graph_edge.hierarchy,
            "Test Graph",
            "GraphEdge is not correctly storing its hierarchy attribute",
        )
        self.assertEqual(
            graph_edge.from_node,
            13,
            "GraphEdge is not correctly storing its from_node attribute",
        )
        self.assertEqual(
            graph_edge.to_node,
            12,
            "GraphEdge is not correctly storing its to_node attribute",
        )
        self.assertEqual(
            graph_edge.weighting,
            0.44,
            "GraphEdge is not correctly storing its weighting attribute",
        )
        self.assertEqual(
            graph_edge.rw_params,
            (0.0, 0.1),
            "GraphEdge is not correctly storing its rw_params attribute",
        )
        self.assertIsNone(
            graph_edge.index, "GraphEdge has an existing index before any assignation"
        )
        graph_edge.set_index(72)
        self.assertEqual(
            graph_edge.index,
            72,
            "GraphEdge set_index is not correctly updating the object's index",
        )
        graph_edge.set_weighting(0.82)
        self.assertEqual(
            graph_edge.weighting,
            0.82,
            "GraphEdge set_weighting is not correctly updating the object's weighting",
        )
        graph_edge.set_rw_params((0.1, 0.0))
        self.assertEqual(
            graph_edge.rw_params,
            (0.1, 0.0),
            "GraphEdge set_rw_params is not correctly updating the object's rw_params",
        )
        graph_edge.update_from_node(3)
        self.assertEqual(
            graph_edge.from_node,
            3,
            "GraphEdge update_from_node is not correctly updating the object's from_node",
        )
        graph_edge.update_to_node(9)
        self.assertEqual(
            graph_edge.to_node,
            9,
            "GraphEdge update_to_node is not correctly updating the object's to_node",
        )
        self.assertTrue(
            graph_edge.has_rw_params(),
            "GraphEdge has_rw_params is not detecting the presence of existing rw_params",
        )
        edge_str_repr: str = graph_edge.__str__()
        self.assertEqual(
            edge_str_repr,
            "GraphEdge of weight (0.82) from node (3) to node (9) in the Test Graph social layer",
            "GraphEdge __str__ representation is not producing the expected outcome",
        )

    @override
    def tearDown(self) -> None:
        """
        Reset the Graph object to run subsequent tests.
        """
        del self.graph
