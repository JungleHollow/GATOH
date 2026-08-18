from __future__ import annotations

import os
import unittest as ut
import pickle
import zipfile
from typing import override

from gatoh.graphs import GraphNode, GraphEdge, Graph, GraphSet


class TestGraphSetPersistence(ut.TestCase):
    @classmethod
    @override
    def setUpClass(cls) -> None:
        cls._graphset: GraphSet = GraphSet()
        for i in range(4):
            new_graph: Graph = Graph(f"{i}", (0.0, 0.0))
            cls._graphset.add_graph(new_graph)
        cls._subdir_path: str = "./tests/test_saves/graphset_persistence"

    def test_write_node_pickle(self) -> None:
        """
        Test that graph node pickles are correctly written.
        """

    def test_write_edge_pickle(self) -> None:
        """
        Test that graph edge pickles are correctly written.
        """

    def test_save_graphset(self) -> None:
        """
        Test that save_graphset() is working as intended.
        """

    def test_load_graphset(self) -> None:
        """
        Test that load_graphset() is working as intended.
        """

    @classmethod
    @override
    def tearDownClass(cls) -> None:
        del cls._graphset
