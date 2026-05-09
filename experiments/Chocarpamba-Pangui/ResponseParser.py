from __future__ import annotations

import csv
from typing import Any


class ResponseParser:
    def __init__(self) -> None:
        self.agents: dict[str, dict[str, list[Any]]] = {}
        self.graphs: dict[str, dict[str, list[Any]]] = {}

        for key in RESPONSE_CSV.keys():
            self.agents[key] = {}
            self.graphs[key] = {}

    def calculate_weightings(self) -> None:
        pass

    def create_base_hierarchies(self) -> None:
        pass

    def write_agents(self) -> None:
        pass

    def write_graphs(self) -> None:
        pass


if __name__ == "__main__":
    RESPONSE_CSV: dict[str, str] = {
        "CHPBA": "./data/Chocarpamba.csv",
        "PANGI": "./data/Pangui.csv",
    }

    AGENT_PATHS: dict[str, str] = {
        "CHPBA": "./experiments/Chocarpamba-Pangui/CHPBA_Agents.csv",
        "PANGI": "./experiments/Chocarpamba-Pangui/PANGI_Agents.csv",
    }

    GRAPH_PATHS: dict[str, str] = {
        "CHPBA": "./experiments/Chocarpamba-Pangui/CHPBA_Graphs.csv",
        "PANGI": "./experiments/Chocarpamba-Pangui/PANGI_Graphs.csv",
    }
