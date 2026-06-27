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

    def test_node_from_agent_invalid(self) -> None:
        """
        Test that node_from_agent() with an invalid Agent in a populated Graph returns None.
        """
        invalid_agent: Agent = Agent("INVLD404")
        agent_node = self.graph.node_from_agent(invalid_agent)
        self.assertIsNone(
            agent_node,
            "Graph -- node_from_agent() with an invalid Agent in a populated Graph is returning an object",
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

    def test_get_agent_index_invalid(self) -> None:
        """
        Test that get_agent_index() with an invalid Agent in a populated Graph returns None.
        """
        invalid_agent: Agent = Agent("INVLD404")
        agent_index = self.graph.get_agent_index(invalid_agent)
        self.assertIsNone(
            agent_index,
            "Graph -- get_agent_index() with an invalid Agent in a populated Graph is returning an object",
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
        edges_to_add = {
            "from_node": [1, 1, 1, 1],
            "to_node": [2, 17, 4, 8],
            "weighting": [0.1, 0.2, 0.3, 0.4],
        }
        self.graph.add_edges(edges_to_add)
        agent_neighbours = self.graph.get_neighbours(self.agents[1])
        self.assertIsNotNone(
            agent_neighbours,
            "Graph -- get_neighbours() on a valid Agent with neighbours is returning None",
        )
        self.assertEqual(
            len(agent_neighbours),
            4,
            "Graph -- get_neighbours() on a valid Agent with neighbours is not reporting the correct number of neighbours",
        )
        for neighbour in agent_neighbours:
            self.assertIn(
                neighbour.index,
                edges_to_add["to_node"],
                "Graph -- get_neighbours() on a valid Agent with neighbours is not returning the correct neighbour GraphNodes",
            )

    def test_agent_previous_opinion(self) -> None:
        """
        Test that agent_previous_opinion() correctly stores the previous opinion.
        """
        self.graph.graph.nodes()[0].agent.previous_opinion = 0.0
        self.graph.graph.nodes()[0].agent.opinion = 0.44
        self.graph.agent_previous_opinion(self.graph.graph.nodes()[0].agent)
        self.assertEqual(
            self.graph.graph.nodes()[0].agent.previous_opinion,
            0.44,
            "Graph -- agent_previous_opinion() is not updating the Agent previous_opinion correctly",
        )

    def test_agent_opinion_change_increment(self) -> None:
        """
        Test that agent_opinion_change() correctly increments the current opinion.
        """
        self.graph.graph.nodes()[0].agent.opinion = 0.4
        self.graph.agent_opinion_change(self.graph.graph.nodes()[0].agent, 0.12)
        self.assertAlmostEqual(
            self.graph.graph.nodes()[0].agent.opinion,
            0.52,
            5,
            "Graph -- agent_opinion_change() is not incrementing Agent current_opinion correctly",
        )

    def test_agent_opinion_change_decrement(self) -> None:
        """
        Test that agent_opinion_change() correctly decrements the current opinion.
        """
        self.graph.graph.nodes()[0].agent.opinion = 0.4
        self.graph.agent_opinion_change(self.graph.graph.nodes()[0].agent, -0.09)
        self.assertAlmostEqual(
            self.graph.graph.nodes()[0].agent.opinion,
            0.31,
            5,
            "Graph -- agent_opinion_change is not decrementing Agent current_opinion correctly",
        )

    def test_agent_radicalisation_change(self) -> None:
        """
        Test that agent_radicalisation_change() correctly changes the Agent's radicalisation.
        """
        self.graph.graph.nodes()[0].agent.radicalisation = False
        self.graph.agent_radicalisation_change(self.graph.graph.nodes()[0].agent, True)
        self.assertTrue(
            self.graph.graph.nodes()[0].agent.radicalisation,
            "Graph -- agent_radicalisation_change() is not updating the Agent radicalisation correctly",
        )

    def test_neighbour_influences_solo(self) -> None:
        """
        Test that neighbour_influences() on a valid, solitary Agent returns 0.0.
        """
        neighbour_influences = self.graph.neighbour_influences(self.agents[0])
        self.assertIsNotNone(
            neighbour_influences,
            "Graph -- neighbour_influences() on a valid, solitary Agent is returning None",
        )
        self.assertEqual(
            neighbour_influences,
            0.0,
            "Graph -- neighbour_influences() on a valid, solitary Agent is not returning 0.0",
        )

    def test_neighbour_influences_simple(self) -> None:
        """
        Test that neighbour_influences() on a valid Agent with a single neighbour returns the expected value.
        """
        # These values will mean an average opinion of 0.5, with Agent 13 having a distance of -0.1 from the average
        # weighted_delta = distance_from_avg * agent_hierarchy_weighting * relationship_strength
        # therefore weighted_delta = -0.1 * 0.5 * 0.5
        # = -0.025
        # No other neighbours to affect the relative weighting, so the final_change should equal -0.025
        self.graph.graph.nodes()[13].opinion = 0.6
        self.graph.graph.nodes()[12].opinion = 0.4
        edge_to_add = {
            "from_node": [13],
            "to_node": [12],
            "weighting": [0.5],
        }
        self.graph.add_edges(edge_to_add)
        final_change = self.graph.neighbour_influences(self.graph.get_node(13).agent)
        expected_final_change = -0.025
        self.assertIsNotNone(
            final_change,
            "Graph -- neighbour_influences() on a valid Agent with a single neighbour is returning None",
        )
        if final_change is not None:  # Included for type checking warnings
            self.assertAlmostEqual(
                final_change,
                expected_final_change,
                5,
                "Graph -- neighbour_influences() on a valid Agent with a single neighbour is not calculating the correct value",
            )

    def test_neighbour_influences_complex(self) -> None:
        """
        Test that neighbour_influences() on a valid Agent with multiple neighbours returns the expected value.
        """
        self.graph.graph.nodes()[13].opinion = 0.0
        self.graph.graph.nodes()[12].opinion = 0.85
        self.graph.graph.nodes()[4].opinion = 0.44
        self.graph.graph.nodes()[7].opinion = -0.15
        self.graph.graph.nodes()[21].opinion = 0.17
        edges_to_add = {
            "from_node": [13, 13, 13, 13],
            "to_node": [12, 4, 7, 21],
            "weighting": [0.41, 0.1, 0.35, -0.34],
        }
        self.graph.add_edges(edges_to_add)
        final_change = self.graph.neighbour_influences(self.graph.get_node(13).agent)
        # Worked example:
        # 1. (13 -> 12):
        #   - avg_opinion = (0.0 + 0.85) / 2 = 0.425
        #   - dist_from_avg = (0.425 - 0.0) = 0.425
        #   - delta_value = 0.425 * 0.5 * 0.41 = 0.087125
        # 2. (13 -> 4):
        #   - avg_opinion = (0.0 + 0.44) / 2 = 0.22
        #   - dist_from_avg = (0.22 - 0.0) = 0.22
        #   - delta_value = 0.22 * 0.5 * 0.1 = 0.011
        # 3. (13 -> 7):
        #   - avg_opinion = (0.0 + -0.15) / 2 = -0.075
        #   - dist_from_avg = (-0.075 - 0.0) = -0.075
        #   - delta_value = -0.075 * 0.5 * 0.35 = -0.013125
        # 4. (13 -> 21):
        #   - avg_opinion = (0.0 + 0.17) / 2 = 0.085
        #   - dist_from_avg = (0.085 - 0.0) = 0.085
        #   - delta_value = 0.085 * 0.5 * -0.34 = -0.01445
        # ---
        # All are non-radicalised, therefore each delta_value has a relative_weighting = 1.0
        # and total_weightings = 4.0
        # ---
        # final_change = (0.25 * 0.087125) + (0.25 * 0.011) + (0.25 * -0.013125) + (0.25 * -0.01445)
        #   = 0.02178125 + 0.00275 + -0.00328125 + -0.0036125
        #   = 0.0176375
        expected_final_change = 0.0176375
        self.assertIsNotNone(
            final_change,
            "Graph -- neighbour_influences() on a valid Agent with multiple neighbours is returning None",
        )
        if final_change is not None:  # Included for type checking warnings
            self.assertAlmostEqual(
                final_change,
                expected_final_change,
                5,
                "Graph -- neighbour_influences() on a valid Agent with multiple neighbours is not calculating the correct value",
            )

    def test_neighbour_influences_simple_radicalised(self) -> None:
        """
        Test that neighbour_influences() on a valid Agent with a single radicalised neighbour will return the correct value.
        """
        # These values will mean an average opinion of 0.5, with Agent 13 having a distance of -0.1 from the average
        # weighted_delta = distance_from_avg * agent_hierarchy_weighting * relationship_strength
        # therefore weighted_delta = -0.1 * 0.5 * 0.5
        # = -0.025
        # No other neighbours to affect the relative weighting, so the final_change should equal -0.025
        self.graph.graph.nodes()[13].opinion = 0.6
        self.graph.graph.nodes()[12].opinion = 0.4
        self.graph.graph.nodes()[12].radicalised = True
        edge_to_add = {
            "from_node": [13],
            "to_node": [12],
            "weighting": [0.5],
        }
        self.graph.add_edges(edge_to_add)
        final_change = self.graph.neighbour_influences(self.graph.get_node(13).agent)
        # For the solitary case, the expected final change is the same as if the neighbour were non-radicalised
        # (as the relative weighting should have no effect)
        expected_final_change = -0.025
        self.assertIsNotNone(
            final_change,
            "Graph -- neighbour_influences() on a valid Agent with a single radicalised neighbour is returning None",
        )
        if final_change is not None:  # Included for type checking warnings
            self.assertAlmostEqual(
                final_change,
                expected_final_change,
                5,
                "Graph -- neighbour_influences() on a valid Agent with a single radicalised neighbour is not calculating the correct value",
            )

    def test_neighbour_influences_complex_radicalised(self) -> None:
        """
        Test that neighbour_influences() on a valid Agent with multiple, partially radicalised Agents will return the correct value.
        """
        self.graph.graph.nodes()[13].opinion = 0.0
        self.graph.graph.nodes()[13].personality = "rational"
        self.graph.graph.nodes()[12].opinion = 0.85
        self.graph.graph.nodes()[12].radicalised = True
        self.graph.graph.nodes()[4].opinion = 0.44
        self.graph.graph.nodes()[7].opinion = -0.15
        self.graph.graph.nodes()[21].opinion = 0.17
        edges_to_add = {
            "from_node": [13, 13, 13, 13],
            "to_node": [12, 4, 7, 21],
            "weighting": [0.41, 0.1, 0.35, -0.34],
        }
        self.graph.add_edges(edges_to_add)
        final_change = self.graph.neighbour_influences(self.graph.get_node(13).agent)
        # Worked example:
        # 1. (13 -> 12):
        #   - avg_opinion = (0.0 + 0.85) / 2 = 0.425
        #   - dist_from_avg = (0.425 - 0.0) = 0.425
        #   - delta_value = 0.425 * 0.5 * 0.41 = 0.087125
        # 2. (13 -> 4):
        #   - avg_opinion = (0.0 + 0.44) / 2 = 0.22
        #   - dist_from_avg = (0.22 - 0.0) = 0.22
        #   - delta_value = 0.22 * 0.5 * 0.1 = 0.011
        # 3. (13 -> 7):
        #   - avg_opinion = (0.0 + -0.15) / 2 = -0.075
        #   - dist_from_avg = (-0.075 - 0.0) = -0.075
        #   - delta_value = -0.075 * 0.5 * 0.35 = -0.013125
        # 4. (13 -> 21):
        #   - avg_opinion = (0.0 + 0.17) / 2 = 0.085
        #   - dist_from_avg = (0.085 - 0.0) = 0.085
        #   - delta_value = 0.085 * 0.5 * -0.34 = -0.01445
        # ---
        # All are non-radicalised, except for neighbour 12, and node 13 has personality "rational",
        # therefore (13 -> 12) is given relative weighting 0.5, and all others are given relative weighting 1.0
        # for a final total_weightings = 3.5
        # ---
        # let:
        # normal_weighting (n_w) = 1.0 / 3.5 = 0.285714285714
        # radical_weighting (r_w) = 0.5 / 3.5 = 0.142857142857
        # ---
        # final_change = (r_w * 0.087125) + (n_w * 0.011) + (n_w * -0.013125) + (n_w * -0.01445)
        #   = 0.0124464285714 + 0.00314285714285 + -0.00375 + -0.00412857142857
        #   = 0.00771071428568
        expected_final_change = 0.00771071428568
        self.assertIsNotNone(
            final_change,
            "Graph -- neighbour_influences() on a valid Agent with multiple partially radicalised neighbours is returning None",
        )
        if final_change is not None:  # Included for type checking warnings
            self.assertAlmostEqual(
                final_change,
                expected_final_change,
                5,
                "Graph -- neighbour_influences() on a valid Agent with multiple partially radicalised neighbours is not calculating the correct value",
            )

    def test_estimate_neighbour_opinions_solo(self) -> None:
        """
        Test that estimate_neighbour_opinions() on a valid, solitary Agent returns an empty dictionary.
        """
        estimated_opinions = self.graph.estimate_neighbour_opinions(self.agents[0])
        self.assertEqual(
            estimated_opinions,
            {},
            "Graph -- estimate_neighbour_opinions() on a valid, solitary Agent is not returning an empty dictionary",
        )

    def test_estimate_neighbour_opinions_complex(self) -> None:
        """
        Test that estimate_neighbour_opinions() on a valid Agent with multiple neighbours returns the expected output.
        """
        self.graph.graph.nodes()[12].opinion = 0.1
        self.graph.graph.nodes()[4].opinion = 0.93
        self.graph.graph.nodes()[2].opinion = 0.44
        self.graph.graph.nodes()[9].opinion = -0.99
        edges_to_add = {
            "from_node": [13, 13],
            "to_node": [12, 4],
            "weighting": [0.1, 0.1],
        }
        # 12 and 4 are direct neighbours with the node of interest; 2 and 9 are not
        self.graph.add_edges(edges_to_add)
        estimated_opinions = self.graph.estimate_neighbour_opinions(
            self.graph.get_node(13).agent
        )
        # Indirect neighbour 2 should not have a strong enough opinion to be passively detected by 13
        self.assertNotIn(
            self.graph.get_node(2).agent.id,
            estimated_opinions.keys(),
            "Graph -- estimate_neighbour_opinions() on a valid Agent with multiple neighbours is including a 'weak' indirect opinion in the estimate",
        )
        for neighbour_index in [12, 4]:
            direct_id = self.graph.get_node(neighbour_index).agent.id
            self.assertIn(
                direct_id,
                estimated_opinions.keys(),
                "Graph -- estimate_neighbour_opinions() on a valid Agent with multiple neighbours is not including the direct opinions in the estimate",
            )
            if neighbour_index == 12:
                self.assertEqual(
                    estimated_opinions[direct_id],
                    0.1,
                    "Graph -- estimate_neighbour_opinions() on a valid Agent with multiple neighbours is not reporting the correct weak direct opinions",
                )
            elif neighbour_index == 4:
                self.assertEqual(
                    estimated_opinions[direct_id],
                    0.93,
                    "Graph -- estimate_neighbour_opinions() on a valid Agent with multiple neighbours is not reporting the correct strong direct opinions",
                )
        strong_indirect_id = self.graph.get_node(9).agent.id
        self.assertIn(
            strong_indirect_id,
            estimated_opinions.keys(),
            "Graph -- estimate_neighbour_opinions() on a valid Agent with multiple neighbours is not including strong indirect opinions in the estimate",
        )
        self.assertEqual(
            estimated_opinions[strong_indirect_id],
            -0.99,
            "Graph -- estimate_neighbour_opinions() on a valid Agent with multiple neighbours is not reporting the correct strong indirect opinions",
        )

    def test_estimate_opinion_climate_solo(self) -> None:
        """
        Test that estimate_opinion_climate() on a valid, solitary Agent will return 0.0.
        """
        opinion_climate = self.graph.estimate_opinion_climate(self.agents[13])
        self.assertIsInstance(
            opinion_climate,
            float,
            "Graph -- estimate_opinion_climate() on a valid, solitary Agent is not returning a float value",
        )
        self.assertEqual(
            opinion_climate,
            0.0,
            "Graph -- estimate_opinion_climate() on a valid, solitary Agent is not returning 0.0",
        )

    def test_estimate_opinion_climate_simple(self) -> None:
        """
        Test that estimate_opinion_climate() on a vaild Agent with a single neighbour will return the expected value.
        """
        self.graph.graph.nodes()[12].opinion = 0.21
        edge_to_add = {
            "from_node": [13],
            "to_node": [12],
            "weighting": [0.75],
        }
        self.graph.add_edges(edge_to_add)
        opinion_climate = self.graph.estimate_opinion_climate(
            self.graph.get_node(13).agent
        )
        self.assertIsInstance(
            opinion_climate,
            float,
            "Graph -- estimate_opinion_climate() on a valid Agent with a single neighbour is not returning a float value",
        )
        self.assertEqual(
            opinion_climate,
            0.21,
            "Graph -- estimate_opinion_climate() on a valid Agent with a single neighbour is not calculating the expected value",
        )

    def test_estimate_opinion_climate_indirect(self) -> None:
        """
        Test that estimate_opinion_climate() on a valid Agent with no neighbours, but strong indirect opinions will return the expected value.
        """
        self.graph.graph.nodes()[12].opinion = 0.99
        opinion_climate = self.graph.estimate_opinion_climate(
            self.graph.get_node(13).agent
        )
        self.assertIsInstance(
            opinion_climate,
            float,
            "Graph -- estimate_opinion_climate() on a valid Agent with strong indirect opinions is not returning a float value",
        )
        self.assertEqual(
            opinion_climate,
            0.99,
            "Graph -- estimate_opinion_climate() on a valid Agent with strong indirect opinions is not returning the expected value",
        )

    def test_estimate_opinion_climate_complex(self) -> None:
        """
        Test that estimate_opinion_climate on a valid Agent with direct neighbours and a strong indirect opinion will return the expected value.
        """
        self.graph.graph.nodes()[12].opinion = 0.99
        self.graph.graph.nodes()[4].opinion = 0.44
        self.graph.graph.nodes()[15].opinion = -0.12
        edges_to_add = {
            "from_node": [13, 13],
            "to_node": [4, 15],
            "weighting": [0.3, 0.74],
        }
        self.graph.add_edges(edges_to_add)
        opinion_climate = self.graph.estimate_opinion_climate(
            self.graph.get_node(13).agent
        )
        self.assertIsInstance(
            opinion_climate,
            float,
            "Graph -- estimate_opinion_climate() on a valid Agent (complex case) is not returning a float value",
        )
        # Worked example:
        # observed_opinions will include [0.99, 0.44, -0.12]
        # therefore summed_opinions = 1.31
        # expected opinion climate = summed_opinions / len(observed_opinions)
        # -> expected opinion climate = 0.436666666
        expected_opinion_climate = 0.4366666
        self.assertAlmostEqual(
            opinion_climate,
            expected_opinion_climate,
            5,
            "Graph -- estimate_opinion_climate() on a valid Agent (complex case) is not returning the expected value",
        )
