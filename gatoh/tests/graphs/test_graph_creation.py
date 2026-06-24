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

    @override
    def tearDown(self) -> None:
        """
        Reset the Graph object to run subsequent tests.
        """
        del self.graph
