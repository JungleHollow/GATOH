from __future__ import annotations

import unittest as ut

import gatoh.graphs as gr


class TestGraphSet(ut.TestCase):
    def test_init(self) -> None:
        """
        Test that an empty initialisation of a GraphSet is returning the expected value.
        """
        empty_graphset: gr.GraphSet = gr.GraphSet()
        self.assertIsInstance(
            empty_graphset,
            gr.GraphSet,
            "GraphSet -- Empty initialisation is not returning a GraphSet object",
        )
        self.assertEqual(
            empty_graphset.graphs,
            [],
            "GraphSet -- Empty initialisation is not initialising an empty graphs list",
        )
        self.assertEqual(
            empty_graphset.stochastic_relationships,
            {},
            "GraphSet -- Empty initialisation is not initialising an empty stochastic_relationships dictionary",
        )
        self.assertEqual(
            empty_graphset.stochastic_rels_flags,
            {},
            "GraphSet -- Empty initialisation is not initialising an empty stochastic_rels_flags dictionary",
        )

    def test_add_graph(self) -> None:
        """
        Test that add_graph() is working as intended.
        """
        graphset: gr.GraphSet = gr.GraphSet()
        new_graph: gr.Graph = gr.Graph("FooBar", (0.0, 0.0))
        graphset.add_graph(new_graph)
        self.assertEqual(
            len(graphset.graphs),
            1,
            "GraphSet -- add_graph() is not adding the graph object to the graphs list",
        )
        self.assertTrue(
            new_graph in graphset.graphs,
            "GraphSet -- add_graph() is not adding the graph object correctly to the graphs list",
        )
