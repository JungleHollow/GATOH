import argparse

from .agents import Agent as Agent
from .agents import AgentSet as AgentSet
from .graphs import Graph as Graph
from .graphs import GraphEdge as GraphEdge
from .graphs import GraphNode as GraphNode
from .graphs import GraphSet as GraphSet
from .logging import ABMOSLogger as ABMOSLogger
from .model import ABModel as ABModel

__version__ = "0.1"
__authors__ = "Manuel Munizaga Sepulveda"
__license__ = "MIT License"
__year__ = "2025"
__repo__ = "https://www.github.com/JungleHollow/ABMOS"

###
# This may be turned into a CLI entry point or extended context manager in the future...
###

parser = argparse.ArgumentParser(
    prog="ABMOS",
    usage="",
    description="An open-source Python package to model social unrest in small yet complex communities",
    epilog="",
)
