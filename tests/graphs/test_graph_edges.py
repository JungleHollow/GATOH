from __future__ import annotations

import unittest as ut
from typing import override

import gatoh.graphs as gr


class TestGraphEdges(ut.TestCase):
    @override
    def setUp(self) -> None:
        """
        The base GraphEdge object that is repeated across tests.
        """
        self.graph_edge: gr.GraphEdge = gr.GraphEdge("TestGraph", 13, 12)

    def test_edge_creation(self) -> None:
        """
        Test that GraphEdges are initialised correctly with no optional arguments passed.
        """
        self.assertNotHasAttr(
            self.graph_edge,
            "index",
            "GraphEdge -- existing edge index before any assignation",
        )
        self.assertEqual(
            self.graph_edge.weighting,
            0.0,
            "GraphEdge -- default value for optional weighting argument is not being used",
        )
        self.assertEqual(
            self.graph_edge.from_node,
            13,
            "GraphEdge -- from_node not being initialised correctly",
        )
        self.assertEqual(
            self.graph_edge.to_node,
            12,
            "GraphEdge -- to_node not being initialised correctly",
        )
        self.assertEqual(
            self.graph_edge.hierarchy,
            "TestGraph",
            "GraphEdge -- hierarchy not being initialised correctly",
        )
        self.assertIsNone(
            self.graph_edge.rw_params,
            "GraphEdge -- default value for optional rw_params argument is not being used",
        )
        str_repr: str = self.graph_edge.__str__()
        self.assertEqual(
            str_repr,
            "GraphEdge of weight (0.0) from node (13) to node (12) in the TestGraph social layer",
            "GraphEdge -- __str__ is not producing the expected representation (from __init__)",
        )

    def test_set_index(self) -> None:
        """
        Test that GraphEdge.set_index() is working correctly.
        """
        self.graph_edge.set_index(44)
        self.assertEqual(
            self.graph_edge.index,
            44,
            "GraphEdge -- set_index() not updating the index attribute correctly",
        )

    def test_set_weighting(self) -> None:
        """
        Test that GraphEdge.set_weighting() is working correctly.
        """
        self.graph_edge.set_weighting(0.44)
        self.assertEqual(
            self.graph_edge.weighting,
            0.44,
            "GraphEdge -- set_weighting() not updating the weighting attribute correctly",
        )

    def test_set_rw_params(self) -> None:
        """
        Test that GraphEdge.set_rw_params() is working correctly.
        """
        self.graph_edge.set_rw_params((0.0, 0.1))
        self.assertEqual(
            self.graph_edge.rw_params,
            (0.0, 0.1),
            "GraphEdge -- set_rw_params() not updating the rw_params attribute correctly",
        )

    def test_update_from_node(self) -> None:
        """
        Test that GraphEdge.update_from_node() is working correctly.
        """
        self.graph_edge.update_from_node(3)
        self.assertEqual(
            self.graph_edge.from_node,
            3,
            "GraphEdge -- update_from_node() not updating the from_node attribute correctly",
        )

    def test_update_to_node(self) -> None:
        """
        Test that GraphEdge.update_to_node() is working correctly.
        """
        self.graph_edge.update_to_node(2)
        self.assertEqual(
            self.graph_edge.to_node,
            2,
            "GraphEdge -- update_to_node() not updating the to_node attribute correctly",
        )

    def test_has_rw_params(self) -> None:
        """
        Test that GraphEdge.set_rw_params() is working correctly
        """
        self.assertFalse(
            self.graph_edge.has_rw_params(),
            "GraphEdge -- has_rw_params() not detecting the absence of rw_params",
        )
        self.graph_edge.set_rw_params((0.0, 0.0))
        self.assertTrue(
            self.graph_edge.has_rw_params(),
            "GraphEdge -- has_rw_params() not detecting the presence of rw_params",
        )

    def test_str_repr(self) -> None:
        """
        Test that GraphEdge.__str__() returns the correct representation after operations on the object.
        """
        self.graph_edge.set_weighting(0.44)
        self.graph_edge.update_from_node(3)
        self.graph_edge.update_to_node(2)
        str_repr: str = self.graph_edge.__str__()
        self.assertEqual(
            str_repr,
            "GraphEdge of weight (0.44) from node (3) to node (2) in the TestGraph social layer",
            "GraphEdge -- __str__ is not producing the expected representation (after operations)",
        )

    @override
    def tearDown(self) -> None:
        del self.graph_edge
