import argparse

from gatoh.agents import Agent as Agent
from gatoh.agents import AgentSet as AgentSet
from gatoh.graphs import Graph as Graph
from gatoh.graphs import GraphEdge as GraphEdge
from gatoh.graphs import GraphNode as GraphNode
from gatoh.graphs import GraphSet as GraphSet
from gatoh.logging import GATOHLogger as GATOHLogger
from gatoh.model import ABModel as ABModel
from gatoh.visualisation import ABVisualiser as ABVisualiser

__version__ = "2026.07.0"
__authors__ = "Manuel Munizaga Sepulveda"
__license__ = "MIT License"
__year__ = "2026"
__repo__ = "https://www.github.com/JungleHollow/GATOH"

###
# This may be turned into a CLI entry point or extended context manager in the future...
###

parser = argparse.ArgumentParser(
    prog="gatoh",
    usage="",
    description="Generalised Agent Transformations of Opinions in Hierarchies -- An open-source Python package to model social unrest in small yet complex communities",
    epilog="",
)
