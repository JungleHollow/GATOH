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

    def test_relationship_exists(self) -> None:
        """
        Test that relationship_exists() on a valid relationship returns the corresponding edge index.
        """
        edge_to_add = {
            "from_node": [13],
            "to_node": [12],
            "weighting": [0.44],
        }
        self.graph.add_edges(edge_to_add)
        edge_index = self.graph.relationship_exists(13, 12)
        self.assertIsNotNone(
            edge_index,
            "Graph -- relationship_exists() on a valid relationship is returning None",
        )
        self.assertEqual(
            edge_index,
            0,
            "Graph -- relationship_exists() on a valid relationship is not returning an edge index",
        )

    def test_get_relationships_unidirectional(self) -> None:
        """
        Test that get_relationships() on a valid unidirectional relationship produces a valid return.
        """
        edge_to_add = {
            "from_node": [13],
            "to_node": [12],
            "weighting": [0.44],
        }
        self.graph.add_edges(edge_to_add)
        relationship = self.graph.get_relationships(13, 12)
        self.assertIsNotNone(
            relationship,
            "Graph -- get_relationships() on a valid unidirectional relationship is returning None",
        )
        self.assertEqual(
            len(relationship.keys()),
            1,
            "Graph -- get_relationships() on a valid unidirectional relationship is not returning a dictionary with 1 entry",
        )
        self.assertEqual(
            relationship[(13, 12)],
            0.44,
            "Graph -- get_relationships() is not reporting the correct weighting for A -> B (unidirectional)",
        )

    def test_get_relationships_bidirectional(self) -> None:
        """
        Test that get_relationships() on a valid bidirectional relationship produces a valid return.
        """
        edges_to_add = {
            "from_node": [13, 12],
            "to_node": [12, 13],
            "weighting": [0.16, 0.72],
        }
        self.graph.add_edges(edges_to_add)
        relationships = self.graph.get_relationships(13, 12)
        self.assertIsNotNone(
            relationships,
            "Graph -- get_relationships() on a valid bidirectional relationship is returning None",
        )
        self.assertEqual(
            len(relationships.keys()),
            2,
            "Graph -- get_relationships() on a valid bidirectional relationship is not returning a dictionary with 2 entries",
        )
        self.assertEqual(
            relationships[(13, 12)],
            0.16,
            "Graph -- get_relationships() is not reporting the correct weighting for A -> B (bidirectional)",
        )
        self.assertEqual(
            relationships[(12, 13)],
            0.72,
            "Graph -- get_relationships() is not reporting the correct weighting for B -> A (bidirectional)",
        )

    def test_change_weights_existing(self) -> None:
        """
        Test that change_weights on an existing relationship will correctly update the GraphEdge weighting attribute.
        """
        edge_to_add = {
            "from_node": [13],
            "to_node": [12],
            "weighting": [0.12],
        }
        self.graph.add_edges(edge_to_add)
        self.graph.change_weights(13, 12, 0.21)
        self.assertEqual(
            self.graph.edge_count,
            1,
            "Graph -- change_weights() on an existing edge is creating a new GraphEdge in the Graph",
        )
        self.assertEqual(
            self.graph.graph.edges()[0].weighting,
            0.21,
            "Graph -- change_weights() on an existing edge is not updating the weighting attribute correctly",
        )

    def test_change_weights_new(self) -> None:
        """
        Test that change_weights on an unseen relationship will correctly create a new relationship in the Graph.
        """
        self.graph.change_weights(13, 12, 0.21)
        self.assertEqual(
            self.graph.edge_count,
            1,
            "Graph -- change_weights() with a new relationship is not creating a new GraphEdge in the Graph",
        )
        self.assertEqual(
            self.graph.graph.edges()[0].weighting,
            0.21,
            "Graph -- change_weights() with a new relationship is not setting the weighting attribute correctly",
        )

    def test_remove_node_simple(self) -> None:
        """
        Test that remove_node() on a valid node index which has no related relationships works correctly.
        """
        self.graph.remove_node(4)
        # Rustworkx automatically adds future nodes to empty indices in a graph, so node_count in the Graph should remain untouched
        self.assertEqual(
            self.graph.node_count,
            22,
            "Graph -- remove_node() is updating the node_count attribute",
        )
        self.assertIsNone(
            self.graph.graph.nodes()[4],
            "Graph -- remove_node() is not removing the GraphNode object at the correct index",
        )

    def test_remove_edge_simple(self) -> None:
        """
        Test that remove_edge() on a valid relationship works correctly.
        """
        edge_to_add = {
            "from_node": [13],
            "to_node": [12],
            "weighting": [0.44],
        }
        self.graph.add_edges(edge_to_add)
        self.graph.remove_edge(13, 12)
        # Rustworkx automatically adds future edges to empty indices in a graph, so edge_count in the Graph should remain untouched
        self.assertEqual(
            self.graph.edge_count,
            1,
            "Graph -- remove_edge() is updating the edge_count attribute",
        )
        self.assertIsNone(
            self.graph.graph.edges()[0],
            "Graph -- remove_edge() is not removing the GraphEdge object with the corresponding nodes",
        )

    def test_remove_edge_complex(self) -> None:
        """
        Test that remove_edge on a valid bidirectional relationship deletes the specified direction without touching the other.
        """
        edges_to_add = {
            "from_node": [13, 12],
            "to_node": [12, 13],
            "weighting": [0.44, 0.21],
        }
        self.graph.add_edges(edges_to_add)
        self.graph.remove_edge(13, 12)
        self.assertEqual(
            self.graph.edge_count,
            2,
            "Graph -- remove_edge() is updating the edge_count attribute (bidirectional case)",
        )
        self.assertIsNone(
            self.graph.graph.edges()[0],
            "Graph -- remove_edge() is not removing the GraphEdge object with the corresponding direction (bidirectional)",
        )
        self.assertIsNotNone(
            self.graph.graph.edges()[1],
            "Graph -- remove_edge() is removing an edge between nodes in a direction that was not specified",
        )

    def test_agent_in_graph(self) -> None:
        """
        Test that agent_in_graph() on a valid Agent in the Graph returns True.
        """
        in_graph: bool = self.graph.agent_in_graph(self.agents[15])
        self.assertTrue(
            in_graph, "Graph -- agent_in_graph() with a valid Agent is returning False"
        )

    def test_node_from_agent(self) -> None:
        """
        Test that node_from_agent() on a valid Agent in the Graph returns the corresponding GraphNode.
        """
        agent_node = self.graph.node_from_agent(self.agents[9])
        self.assertIsNotNone(
            agent_node,
            "Graph -- node_from_agent() with a valid Agent is returning None",
        )
        self.assertIsInstance(
            agent_node,
            gr.GraphNode,
            "Graph -- node_from_agent() with a valid Agent is not returning a GraphNode",
        )
        self.assertEqual(
            self.agents[9].id,
            agent_node.agent.id,
            "Graph -- node_from_agent() with a valid Agent is not returning a GraphNode which corresponds to the same Agent",
        )

    def test_get_agent_index(self) -> None:
        """
        Test that get_agent_index() on a valid Agent in the Graph returns an index.
        """
        agent_index = self.graph.get_agent_index(self.agents[0])
        self.assertIsNotNone(
            agent_index,
            "Graph -- get_agent_index() with a valid Agent is returning None",
        )
        self.assertEqual(
            agent_index,
            0,
            "Graph -- get_agent_index() with a valid Agent is not returning the correct index",
        )

    def test_get_neighbours_solo(self) -> None:
        """
        Test that get_neighbours() on a valid, but solitary Agent in the Graph returns an empty list.
        """
        agent_neighbours = self.graph.get_neighbours(self.agents[0])
        self.assertIsNotNone(
            agent_neighbours,
            "Graph -- get_neighbours() on a valid, solitary Agent is returning None",
        )
        self.assertEqual(
            agent_neighbours,
            [],
            "Graph -- get_neighbours() on a valid, solitary Agent is not returning an empty list",
        )

    def test_get_neighbours(self) -> None:
        """
        Test that get_neighbours() on a valid Agent with neighbours returns a list of GraphNodes.
        """
