import argparse

from gatoh.agents.agents import Agent as Agent
from gatoh.agents.agents import AgentSet as AgentSet
from gatoh.graphs.graphs import Graph as Graph
from gatoh.graphs.graphs import GraphEdge as GraphEdge
from gatoh.graphs.graphs import GraphNode as GraphNode
from gatoh.graphs.graphs import GraphSet as GraphSet
from gatoh.logging.logging import GATOHLogger as GATOHLogger
from gatoh.model.model import ABModel as ABModel

__version__ = "0.1"
__authors__ = "Manuel Munizaga Sepulveda"
__license__ = "MIT License"
__year__ = "2025"
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
