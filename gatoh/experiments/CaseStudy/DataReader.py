from __future__ import annotations

from typing import Any

import polars as pl

from gatoh.agents.agents import Agent, AgentSet
from gatoh.graphs.graphs import Graph, GraphEdge, GraphNode, GraphSet
from gatoh.model.model import ABModel


class DataReader:
    """
    A class that will be used to create and handle an appropriate GATOH model from the given agent and
    social hierarchy CSV file paths
    """

    def __init__(
        self,
        agents_path: str,
        initial_hierarchies: list[str],
        hierarchy_matrices: list[str],
        social_path: str,
        base_hierarchies: list[str] | None = None,
        opinions_path: str | None = None,
        iterations: int = 100,
    ) -> None:
        """
        :param agents_path: A relative or absolute file path to a CSV file containing relevant data on model agent characteristics.
        :param initial_hierarchies: A list of the social hierarchies that will be present in the initial data passed to the reader.
        :param hierarchy_matrices: A list of paths to .csv files containing adjacency matrices that outline the presence and relative strength of relationships in each social hierarchy.
        :param social_path: A relative or absolute file path to a CSV file containing relevant data on the relative influence of the existing social hierarchies in the community.
        :param base_hierarchies: A list containing the most basic hierarchies that exist; will be created before <initial_hierarchies>.
        :optional param opinions_path: A relative or absolute file path to a CSV file containing the dependant variables of actual agent opinions; used to compare model accuracy after execution.
        :optional param iterations: The number of iterations that the ABModel will be run for.
        """
        self.agents_path: str = agents_path
        self.agents_df: pl.DataFrame

        self.base_hierarchies: list[str]
        if base_hierarchies:
            self.base_hierarchies = base_hierarchies
        else:
            self.base_hierarchies = [
                "Age",
                "Gender",
            ]
        self.initial_hierarchies: list[str] = initial_hierarchies
        self.hierarchy_influences: dict[str, dict[str, Any]] = {}

        self.social_path: str = social_path
        self.social_df: pl.DataFrame

        self.opinions_path: str | None = opinions_path
        self.opinions_df: pl.DataFrame

        self.agent_objects: list[Agent]
        self.graph_objects: list[Graph]

        with open(self.agents_path, "r") as file:
            self.agents_df = pl.read_csv(file)
        with open(self.social_path, "r") as file:
            self.social_df = pl.read_csv(file)

        if self.opinions_path:
            with open(self.opinions_path, "r") as file:
                self.opinions_df = pl.read_csv(file)

        self.model: ABModel = ABModel(iterations=iterations, xlims=xlims, ylims=ylims)

    def extract_hierarchy_influences(self):
        """
        Calculates the influence of each social hierarchy for each agent
        """
        general_column: pl.Series | None = self.social_df.get_column(
            "General", default=None
        )
        if general_column is not None:
            for hierarchy in set(
                list(general_column)
            ):  # set() to reduce the number that will be iterated over
                if hierarchy not in self.initial_hierarchies:
                    self.initial_hierarchies.append(hierarchy)

        for agent_row in self.social_df.iter_rows(named=True):
            hierarchy_effects: dict[str, Any] = {}
            for hierarchy in self.initial_hierarchies:
                hierarchy_effects[hierarchy] = (
                    0.0  # Initialise each hierarchy effect, even if not explicitly seen in the current agent row
                )

            raw_hierarchy_values: dict[str, list[int]] = {}
            for key, value in agent_row:
                if key == "AgentId":
                    continue
                elif key == "General":
                    hierarchy_effects[value] = (
                        1.0  # Placeholder of 1.0 strength for the moment
                    )
                else:
                    hierarchy_name: str = key.split("_")[0]
                    if hierarchy_name not in raw_hierarchy_values.keys():
                        raw_hierarchy_values[hierarchy_name] = []
                    raw_hierarchy_values[hierarchy_name].append(abs(int(value)))

            for key, value in raw_hierarchy_values.items():
                sum_values: int = sum(value)
                averaged_sum: float = sum_values / len(value)
                final_effect: float = averaged_sum / 10.0
                hierarchy_effects[key] = final_effect

            self.hierarchy_influences[agent_row["AgentId"]] = hierarchy_effects

    def create_model_agents(self):
        """
        Uses agents_df and the extracted hierarchy influences to create Agent objects for the ABModel
        """
        self.agent_objects = []
        for key, value in self.hierarchy_influences.items():
            new_agent: Agent = Agent(
                key, value
            )  # This should create an Agent with id <key> and social_weightings <value>
            self.agent_objects.append(new_agent)
        _ = self.model.add_agents(self.agent_objects)
        self.create_initial_graphs()

    def create_initial_graphs(self):
        """
        Uses the available agent information to generate the initial basic graphs for the model (Age, Gender, Time in community, etc...)
        """
        self.graph_objects = []
        for hierarchy in self.base_hierarchies:
            new_graph = Graph(hierarchy)
            new_graph.add_nodes(self.agent_objects)
            graph_edges: dict[str, list[Any]] = {
                "from_node": [],
                "to_node": [],
                "weighting": [],
            }
            for i in range(new_graph.node_count):
                for j in range(new_graph.node_count):
                    if i == j:
                        continue
                    else:
                        node_i = new_graph.get_node(i)
                        node_j = new_graph.get_node(j)
                        # TODO: FINISH THIS FUNCTION

    def create_model_graphs(self):
        """
        Uses initial_hierarchies and the extracted hierarchy influences to create Graph objects with the appropriate GraphNodes
        """
        for hierarchy in self.initial_hierarchies:
            new_graph = Graph(hierarchy)
            new_graph.add_nodes(self.agent_objects)
            graph_edges: dict[str, list[Any]] = {
                "from_node": [],
                "to_node": [],
                "weighting": [],
            }
            for i in range(new_graph.node_count):
                for j in range(new_graph.node_count):
                    if i == j:
                        continue
                    else:
                        # Check for and create all appropriate graph edges between nodes
                        node_i = new_graph.get_node(i)
                        node_j = new_graph.get_node(j)
                        # TODO: FINISH THIS FUNCTION


if __name__ == "__main__":
    SAVEDIRS: dict[str, str] = {
        "NONMN": "./gatoh/experiments/CaseStudy/Results/NONMN",
        "MINNG": "./gatoh/experiments/CaseStudy/Results/MINNG",
    }

    SAVEFILES: dict[str, str] = {
        "NONMN": "./gatoh/experiments/CaseStudy/Results/NONMN_model_variables.csv",
        "MINNG": "./gatoh/experiments/CaseStudy/Results/MINNG_model_variables.csv",
    }

    AGENT_PATHS: dict[str, str] = {
        "NONMN": "./data/NONMN/NonMining.csv",
        "MINNG": "./data/MINNG/Mining.csv",
    }

    BASE_HIERARCHIES: list[str] = [
        "Friends",
        "Family",
        "Cultural",
        "Religion",
        "Geographical",
        "Social",
    ]

    HIERARCHY_PATHS: dict[str, dict[str, str]] = {
        "NONMN": {
            "Friends": "./data/NONMN/NONMN_Friends.csv",
            "Family": "./data/NONMN/NONMN_Family.csv",
            "Cultural": "./data/NONMN/NONMN_Cultural.csv",
            "Religion": "./data/NONMN/NONMN_Religion.csv",
            "Geographical": "./data/NONMN/NONMN_Geographical.csv",
            "Social": "./data/NONMN/NONMN_Social.csv",
        },
        "MINNG": {
            "Friends": "./data/MINNG/MINNG_Friends.csv",
            "Family": "./data/MINNG/MINNG_Family.csv",
            "Cultural": "./data/MINNG/MINNG_Cultural.csv",
            "Religion": "./data/MINNG/MINNG_Religion.csv",
            "Geographical": "./data/MINNG/MINNG_Geographical.csv",
            "Social": "./data/MINNG/MINNG_Social.csv",
        },
    }

    data_reader: DataReader = DataReader()
