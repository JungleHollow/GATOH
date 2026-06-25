from __future__ import annotations

import unittest as ut

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
        self.assertEqual(
            empty_graph.generation_method,
            "",
            "Graph -- generation_method not initialised correctly",
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

    def test_graph_in(self) -> None:
        """
        Test that __in__() correctly sees a Graph in an iterable.
        """
        empty_graph: gr.Graph = gr.Graph("TestGraph", (0.0, 0.0))
        graph_iterable: list[gr.Graph] = []
        self.assertFalse(
            empty_graph in graph_iterable,
            "Graph -- __in__() is not returning False when the Graph is not in an iterable",
        )
        graph_iterable.append(empty_graph)
        self.assertTrue(
            empty_graph in graph_iterable,
            "Graph -- __in__() is not returning True when the Graph is in an iterable",
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

    def test_optional_attributes(self) -> None:
        """
        Test that the keyword values for Graph __init__ are working correctly.
        """
        empty_graph: gr.Graph = gr.Graph(
            "TestGraph",
            (0.0, 0.0),
            generation_method="random",
            suppress_warnings=True,
            dynamic_rels=False,
        )
        self.assertEqual(
            empty_graph.generation_method,
            "random",
            "Graph -- optional argument generation_method is not working correctly",
        )
        self.assertTrue(
            empty_graph.suppress_warnings,
            "Graph -- optional argument suppress_warnings is not working correctly",
        )
        self.assertFalse(
            empty_graph.dynamic_rels,
            "Graph -- optional argument dynamic_rels is not working correctly",
        )

    def test_change_generation_params(self) -> None:
        """
        Test that Graph.change_generation_params() is working correctly.
        """
        empty_graph: gr.Graph = gr.Graph("TestGraph", (0.0, 0.0))
        empty_graph.change_generation_params(p=0.4, m=5, sbm_sizes=6)
        self.assertEqual(
            empty_graph.generation_params["p"],
            0.4,
            "Graph -- change_generation_params() is not updating 'p' correctly",
        )
        self.assertEqual(
            empty_graph.generation_params["m"],
            5,
            "Graph -- change_generation_params() is not updating 'm' correctly",
        )
        self.assertEqual(
            empty_graph.generation_params["sbm_sizes"],
            6,
            "Graph -- change_generation_params() is not updating 'sbm_sizes' correctly",
        )
        empty_graph.change_generation_params(ensure_complete=False)
        self.assertIsNotNone(
            empty_graph.generation_params["ensure_complete"],
            "Graph -- change_generation_params() is not adding a new generation parameter correctly",
        )
        self.assertFalse(
            empty_graph.generation_params["ensure_complete"],
            "Graph -- change_generation_params() is not storing new generation parameter values correctly",
        )
