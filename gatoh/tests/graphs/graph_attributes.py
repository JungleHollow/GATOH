from __future__ import annotations

import unittest as ut
from typing import override

from rustworkx import PyDiGraph

import gatoh.graphs.graphs as gr


class TestGraphAttributes(ut.TestCase):
    def test_no_attributes(self) -> None:
        """
        Test that unassigned variables are None at Graph init.
        """
        empty_graph: gr.Graph = gr.Graph("TestGraph", (0.0, 0.0))
        self.assertIsInstance(
            empty_graph.graph, PyDiGraph, "Graph -- graph not initialised correctly"
        )
        self.assertEqual(
            empty_graph.node_count, 0, "Graph -- node_count not initialised correctly"
        )
        self.assertEqual(
            empty_graph.edge_count, 0, "Graph -- edge_count not initialised correctly"
        )
        self.assertEqual(
            empty_graph.name, "TestGraph", "Graph -- name not initialised correctly"
        )
        self.assertTrue(
            empty_graph.dynamic_rels, "Graph -- dynamic_rels not initialised correctly"
        )
        self.assertFalse(
            empty_graph.suppress_warnings,
            "Graph -- suppress_warnings not initialised correctly",
        )
        self.assertEqual(
            empty_graph.rw_params,
            (0.0, 0.0),
            "Graph -- rw_params not initialised correctly",
        )
        self.assertEqual(
            empty_graph.generation_params["p"],
            0.25,
            "Graph -- generation_params (p) not initialised correctly",
        )
        self.assertEqual(
            empty_graph.generation_params["m"],
            3,
            "Graph -- generation_params (m) not initialised correctly",
        )
        self.assertEqual(
            empty_graph.generation_params["sbm_sizes"],
            10,
            "Graph -- generation_params (sbm_sizes) not initialised correctly",
        )

    def test_initial_str(self) -> None:
        """
        Test that the __str__ representation is returning the expected output initially.
        """
        empty_graph: gr.Graph = gr.Graph("TestGraph", (0.0, 0.0))
        str_repr: str = empty_graph.__str__()
        self.assertEqual(
            str_repr,
            "Graph representing the TestGraph social hierarchy with 0 nodes and 0 edges",
            "Graph -- __str__ is not returning the expected representation (from __init__)",
        )
