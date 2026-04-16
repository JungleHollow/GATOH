from __future__ import annotations

import src.GATOH.agents as agt
import src.GATOH.graphs as gr
import src.GATOH.logging as lg
import src.GATOH.model as md


class InfluentialTester:
    HIERARCHY_NAMES: list[str] = [
        "family",
        "friends",
        "religion",
        "neighbours",
        "cultural",
    ]
    HIERARCHY_RW_DISTRIBUTIONS: list[tuple[float, float]] = [
        (0.0, 0.01),  # Family
        (0.0, 0.05),  # Friends
        (0.0, 0.15),  # Religion
        (0.0, 0.08),  # Neighbours
        (0.0, 0.2),  # Cultural
    ]

    def __init__(self):
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
