from __future__ import annotations

import unittest as ut
from typing import override

import gatoh.graphs as gr
from gatoh.agents import Agent


class TestGraphObjectsEmpty(ut.TestCase):
    @override
    def setUp(self) -> None:
        """
        Initialise a Graph object with basic initial parameters.
        """
        self.graph: gr.Graph = gr.Graph("TestGraph", (0.0, 0.0))

    def test_get_node_empty(self) -> None:
        """
        Test that get_node() on an empty Graph returns None.
        """
        with self.assertWarns(RuntimeWarning, msg="WARNING: Node with index 1 is out of bounds for graph TestGraph with 0 total nodes.") as cm:
            node_object = self.graph.get_node(1)
        self.assertIsNone(
            node_object,
            "Graph -- get_node() is returning a GraphNode from an empty Graph",
        )

    def test_get_edge_empty(self) -> None:
        """
        Test that get_edge() on an empty Graph returns None.
        """
        with self.assertWarns(RuntimeWarning, msg="WARNING: Edge with index 1 is out of bounds for graph TestGraph with 0 total edges.") as cm:
            edge_object = self.graph.get_edge(1)
        self.assertIsNone(
            edge_object,
            "Graph -- get_edge() is returning a GraphEdge from an empty Graph",
        )

    def test_update_edge_indices_empty(self) -> None:
        """
        Test that update_edge_indices() on an empty Graph does nothing.
        """
        self.graph.update_edge_indices()
        self.assertEqual(
            self.graph.edge_count,
            0,
            "Graph -- update_edge_indices() on an empty Graph is setting a non-zero edge_count",
        )

    def test_update_node_indices_empty(self) -> None:
        """
        Test that update_node_indices() on an empty Graph does nothing.
        """
        self.graph.update_node_indices()
        self.assertEqual(
            self.graph.node_count,
            0,
            "Graph -- update_node_indices() on an empty Graph is setting a non-zero node_count",
        )

    def test_relationship_exists_empty(self) -> None:
        """
        Test that relationship_exists() on an empty Graph returns None.
        """
        edge_index = self.graph.relationship_exists(13, 12)
        self.assertIsNone(
            edge_index,
            "Graph -- relationship_exists() on an empty Graph is returning an edge index",
        )

    def test_get_relationships_empty(self) -> None:
        """
        Test that get_relationships() on an empty Graph returns None.
        """
        relationships_dict = self.graph.get_relationships(13, 12)
        self.assertIsNone(
            relationships_dict,
            "Graph -- get_relationships() on an empty Graph is returning a dictionary object",
        )

    def test_agent_in_graph_empty(self) -> None:
        """
        Test that agent_in_graph() on an empty Graph returns False.
        """
        test_agent: Agent = Agent("TEST0001")
        in_graph: bool = self.graph.agent_in_graph(test_agent)
        self.assertFalse(
            in_graph, "Graph -- agent_in_graph() on an empty Graph is returning True"
        )

    def test_node_from_agent_empty(self) -> None:
        """
        Test that node_from_agent() on an empty Graph returns None.
        """
        test_agent: Agent = Agent("TEST0001")
        agent_node = self.graph.node_from_agent(test_agent)
        self.assertIsNone(
            agent_node,
            "Graph -- node_from_agent() on an empty Graph returns a GraphNode object",
        )

    def test_get_agent_index_empty(self) -> None:
        """
        Test that get_agent_index() on an empty Graph returns None.
        """
        test_agent: Agent = Agent("TEST0001")
        agent_index = self.graph.get_agent_index(test_agent)
        self.assertIsNone(
            agent_index, "Graph -- get_agent_index() on an empty Graph returns an index"
        )

    def test_remove_node_empty(self) -> None:
        """
        Test that remove_node() on an empty Graph produces the appropriate error.
        """
        with self.assertRaises(IndexError, msg="Trying to remove node 4 for hierarchy TestGraph with 0 existing nodes") as cm:
            self.graph.remove_node(4)

    def test_remove_edge_empty(self) -> None:
        """
        Test that remove_edge() on an empty Graph raises the appropriate warning.
        """
        with self.assertWarns(UserWarning, msg="WARNING: Attempted to remove edge (13 -> 12) which does not exist in the graph.") as cm:
            self.graph.remove_edge(13, 12)

    def test_get_neighbours_empty(self) -> None:
        """
        Test that get_neighbours() on an empty Graph returns None and raises an appropriate warning.
        """
        test_agent: Agent = Agent("TEST0001")
        with self.assertWarns(UserWarning, msg="Input Agent does not exist in this hierarchy (TestGraph)") as cm:
            agent_neighbours = self.graph.get_neighbours(test_agent)
        self.assertIsNone(
            agent_neighbours,
            "Graph -- get_neighbours() on an empty Graph is returning a list of GraphNodes",
        )

    def test_neighbour_influences_empty(self) -> None:
        """
        Test that neighbour_influences() on an empty Graph returns None and raises an appropriate warning.
        """
        test_agent: Agent = Agent("TEST0001")
        with self.assertWarns(UserWarning, msg="Input Agent TEST0001 does not exist in this hierarchy (TestGraph)") as cm:
            neighbour_influences = self.graph.neighbour_influences(test_agent)
        self.assertIsNone(
            neighbour_influences,
            "Graph -- neighbour_influences() on an empty Graph is returning a value",
        )

    def test_estimate_neighbour_opinions_empty(self) -> None:
        """
        Test that estimate_neighbour_opinions() on an empty Graph returns an empty dictionary.
        """
        test_agent: Agent = Agent("TEST0001")
        # No msg checked, as this warning should originate from an internal call to a previously tested function
        with self.assertWarns(UserWarning) as cm:
            neighbour_opinions = self.graph.estimate_neighbour_opinions(test_agent)
        self.assertIsInstance(
            neighbour_opinions,
            dict,
            "Graph -- estimate_neighbour_opinions() on an empty Graph is not returning a dictionary",
        )
        self.assertDictEqual(
            neighbour_opinions,
            {},
            "Graph -- estimate_neighbour_opinions() on an empty Graph is not returning an empty dictionary",
        )

    def test_estimate_opinion_climate_empty(self) -> None:
        """
        Test that estimate_opinion_climate() on an empty Graph returns 0.0.
        """
        test_agent: Agent = Agent("TEST0001")
        with self.assertWarns(UserWarning) as cm:
            opinion_climate = self.graph.estimate_opinion_climate(test_agent)
        self.assertIsInstance(
            opinion_climate,
            float,
            "Graph -- estimate_opinion_climate() on an empty Graph is not returning a float value",
        )
        self.assertEqual(
            opinion_climate,
            0.0,
            "Graph -- estimate_opinion_climate() on an empty Graph is not returning 0.0",
        )

    def test_calculate_polarisation_empty(self) -> None:
        """
        Test that calculate_polarisation() on an empty Graph raises an error due to division by zero in the code.
        """
        with self.assertRaises(ZeroDivisionError) as cm:
            polarisation: float = self.graph.calculate_polarisation()
