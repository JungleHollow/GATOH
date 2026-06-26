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

    def test_get_edge(self) -> None:
        """
        Test that get_edge() on a populated Graph works correctly.
        """
        edge_to_add = {
            "from_node": [13],
            "to_node": [12],
            "weighting": [0.45],
        }
        self.graph.add_edges(edge_to_add)
        edge_object = self.graph.get_edge(0)
        self.assertIsNotNone(
            edge_object,
            "Graph -- get_edge() on a valid index is not returning any object",
        )
        self.assertIsInstance(
            edge_object,
            gr.GraphEdge,
            "Graph -- get_edge() on a valid index is not returning a GraphEdge object",
        )
        self.assertEqual(
            edge_object.index,
            0,
            "Graph -- get_edge() is showing a mismatch between the input index and the returned GraphEdge's index",
        )

    def test_update_edge_indices(self) -> None:
        """
        Test that update_edge_indices() will correctly update the GraphEdge index attributes of edges in the Graph.
        """
        edges_to_add = [
            (
                13,
                12,
                gr.GraphEdge(
                    "TestGraph",
                    13,
                    12,
                ),
            ),
            (
                4,
                7,
                gr.GraphEdge(
                    "TestGraph",
                    4,
                    7,
                ),
            ),
            (
                19,
                1,
                gr.GraphEdge(
                    "TestGraph",
                    19,
                    1,
                ),
            ),
        ]
        _ = self.graph.graph.add_edges_from(edges_to_add)
        self.assertEqual(
            self.graph.edge_count,
            0,
            "Graph -- edge_count attribute is being updated unexpectedly",
        )
        self.assertIsNone(
            self.graph.graph.edges()[0].index,
            "Graph -- GraphEdge is being assigned an index without calling update_edge_indices()",
        )
        self.graph.update_edge_indices()
        self.assertEqual(
            self.graph.edge_count,
            3,
            "Graph -- edge_count attribute is not being updated correctly by update_edge_indices()",
        )
        self.assertEqual(
            self.graph.graph.edges()[2].index,
            2,
            "Graph -- update_edge_indices() is not assigning the correct indices to GraphEdges",
        )

    def test_update_node_indices(self) -> None:
        """
        Test that update_node_indices() will correctly update the GraphNode's index attribute in the Graph.
        """
        empty_graph: gr.Graph = gr.Graph("EmptyGraph", (0.0, 0.0))
        agent_graphnodes: list[gr.GraphNode] = [
            gr.GraphNode(agent) for agent in self.agents
        ]
        _ = empty_graph.graph.add_nodes_from(agent_graphnodes)
        self.assertEqual(
            empty_graph.node_count,
            0,
            "Graph -- node_count attribute is being updated unexpectedly",
        )
        self.assertIsNone(
            empty_graph.graph.nodes()[0].index,
            "Graph -- GraphNode is being assigned an index without calling update_node_indices()",
        )
        empty_graph.update_node_indices()
        self.assertEqual(
            empty_graph.node_count,
            22,
            "Graph -- node_count attribute is not being updated correctly by update_node_indices()",
        )
        self.assertEqual(
            empty_graph.graph.nodes()[4].index,
            4,
            "Graph -- update_node_indices() is not assigning the correct indices to GraphNodes",
        )
