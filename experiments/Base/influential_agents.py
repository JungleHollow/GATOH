from __future__ import annotations

import random as rd
from typing import Any

import src.GATOH.agents as agt
import src.GATOH.graphs as gr
import src.GATOH.logging as lg
import src.GATOH.model as md


class InfluentialTester:
    """
    A test class that sets up and runs models for a low-influence (li) scenario and a high-influence (hi) scenario for
    an experimental comaprison.

    In both cases, the models are composed of 5 hierarchies each with unique random walk distributions for their dynamic
    relationships. Additionally, both models will have 20 Agents within them, and run for a total of 100 iterations.

    In the case of the li model, all 20 Agents are "low influence" Agents with somewhat weaker relationships, and with
    hierarchy graphs having significantly less density of relationships between agents.

    In the case of the hi model, 18 of the Agents are "low influence" whilst 2 are replace with "high influence" Agents.
    These high influence Agents have somewhat stronger relationships with their neighbours, and a significantly higher
    density of relationships between them and other Agents in the hierarchies.
    """

    # The social hierarchies that will exist in the models
    HIERARCHY_NAMES: list[str] = [
        "family",
        "friends",
        "religion",
        "neighbours",
        "cultural",
    ]

    # The random walk distribution parameters for each hierarchy
    HIERARCHY_RW_DISTRIBUTIONS: list[tuple[float, float]] = [
        (0.0, 0.01),  # Family
        (0.0, 0.05),  # Friends
        (0.0, 0.15),  # Religion
        (0.0, 0.08),  # Neighbours
        (0.0, 0.2),  # Cultural
    ]

    # The hierarchy weightings that will be given to each hierarchy (shared amongst all agents)
    HIERARCHY_WEIGHTINGS: dict[str, float] = {
        "family": 0.9,
        "friends": 0.7,
        "religion": 0.5,
        "neighbours": 0.55,
        "cultural": 0.25,
    }

    # Defining the Agents that will be created across both models
    AGENT_CHARACTERISTICS: dict[str, dict[str, Any]] = {
        # Low influence Agents that make up the entire graph in the li case, and the majority in the hi case
        "LOWI0001": {"opinion": rd.uniform(0.0, 1.0)},
        "LOWI0002": {"opinion": rd.uniform(0.0, 1.0)},
        "LOWI0003": {"opinion": rd.uniform(0.0, 1.0)},
        "LOWI0004": {"opinion": rd.uniform(0.0, 1.0)},
        "LOWI0005": {"opinion": rd.uniform(0.0, 1.0)},
        "LOWI0006": {"opinion": rd.uniform(0.0, 1.0)},
        "LOWI0007": {"opinion": rd.uniform(0.0, 1.0)},
        "LOWI0008": {"opinion": rd.uniform(0.0, 1.0)},
        "LOWI0009": {"opinion": rd.uniform(0.0, 1.0)},
        "LOWI0010": {"opinion": rd.uniform(0.0, 1.0)},
        "LOWI0011": {"opinion": rd.uniform(0.0, 1.0)},
        "LOWI0012": {"opinion": rd.uniform(0.0, 1.0)},
        "LOWI0013": {"opinion": rd.uniform(0.0, 1.0)},
        "LOWI0014": {"opinion": rd.uniform(0.0, 1.0)},
        "LOWI0015": {"opinion": rd.uniform(0.0, 1.0)},
        "LOWI0016": {"opinion": rd.uniform(0.0, 1.0)},
        "LOWI0017": {"opinion": rd.uniform(0.0, 1.0)},
        "LOWI0018": {"opinion": rd.uniform(0.0, 1.0)},
        # Low influence Agents with a strong negative opinion
        "LOWI0019": {"opinion": rd.uniform(-1.0, -0.5)},
        "LOWI0020": {"opinion": rd.uniform(-1.0, -0.5)},
        # High influence Agents that replace the negative opinion Agents above
        "INFL0001": {"opinion": rd.uniform(-1.0, -0.5)},
        "INFL0002": {"opinion": rd.uniform(-1.0, -0.5)},
    }

    def __init__(self):
        self.li_agents: list[agt.Agent] = []
        self.create_li_agents()

        self.hi_agents: list[agt.Agent] = []
        self.create_hi_agents()

        self.li_graphs: list[gr.Graph] = []
        self.create_li_graphs()

        self.hi_graphs: list[gr.Graph] = []
        self.create_hi_graphs()

        self.li_model: md.ABModel = md.ABModel(
            InfluentialTester.HIERARCHY_NAMES,
            InfluentialTester.HIERARCHY_RW_DISTRIBUTIONS,
        )
        self.hi_model: md.ABModel = md.ABModel(
            InfluentialTester.HIERARCHY_NAMES,
            InfluentialTester.HIERARCHY_RW_DISTRIBUTIONS,
        )

    def create_li_agents(self):
        pass

    def create_hi_agents(self):
        pass

    def create_li_graphs(self):
        pass

    def create_hi_graphs(self):
        pass

    def run_model_li(self):
        pass

    def run_model_hi(self):
        pass


if __name__ == "__main__":
    tester: InfluentialTester = InfluentialTester()
