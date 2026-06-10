from __future__ import annotations

import json
import os
import pickle
import random as rd
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

import gatoh.agents.agents as agt
import gatoh.graphs.graphs as gr


@dataclass
class AgentAttributes:
    """
    A dataclass used to collect all relevant agent attributes that will be saved for Agent creation,
    and used to determine relationships during hierarchy creation.
    """

    id: str
    age: str
    gender: str
    weightings: dict[str, float]
    opinion: float
    benefit: bool
    sociability: tuple[str, float]

    def __init__(
        self,
        id: str,
        age: str,
        gender: str,
        weightings: dict[str, float],
        opinion: float,
        benefit: bool,
        sociability: tuple[str, float],
    ) -> None:
        self.id = id
        self.age = age
        self.gender = gender
        self.weightings = weightings
        self.opinion = opinion
        self.benefit = benefit
        self.sociability = sociability


class ResponseParser:
    def __init__(self) -> None:
        self.agents: dict[str, list[AgentAttributes]] = {}
        self.agent_objects: dict[str, dict[str, agt.Agent]] = {}
        self.graphs: dict[str, dict[str, list[Any]]] = {}
        self.graph_objects: dict[str, list[gr.Graph]] = {}
        self.hierarchy_clusters: dict[str, dict[str, dict[str, list[str]]]] = {}

        for key in RESPONSE_CSV.keys():
            self.agents[key] = []
            self.agent_objects[
                key
            ] = {}  # <ID: Agent> mapping to simplify lookups later
            self.graphs[key] = {
                "hierarchies": [],  # String names
                "adj_matrices": [],  # 2D matrices indicating presence and strength of graph relationships (ranges [-1, 1])
                "rw_params": [],  # (mean, variance) defining the dynamic relationship random walk distributions
                "agents": [],  # List of IDs of all agents contained within the hierarchy
            }
            self.graph_objects[key] = []
            self.hierarchy_clusters[
                key
            ] = {  # All values are dictionaries that will represent <cluster, list[agent ids]> within each hierarchy
                "Age": {"18-25": [], "26-35": [], "36-45": [], "46-60": [], "60+": []},
                "Gender": {"Male": [], "Female": [], "Other": []},
                "Religious": {"Yes": [], "No": []},
                "Cultural_1": {
                    "Yes": [],  # Participates in community dialogue
                    "No": [],  # Does not participate
                    "Maybe": [],  # May participate
                },
                "Cultural_2": {
                    # Favourite sports
                    "Football": [],
                    "Basketball": [],
                    "Volleyball": [],
                    "Athletics": [],
                    "Other": [],
                },
                "Cultural_3": {
                    # Favourite music genres
                    "Traditional": [],
                    "Popular": [],
                    "Relgiious": [],
                    "International": [],
                    "Varied": [],
                },
                "General": {  # "Wildcard" groupings that will affect existing relationship strengths across all hierarchies
                    "Environmental": [],
                    "Economic": [],
                },
            }

        self.survey_values: dict[str, Any]
        self.survey_question_types: dict[str, list[str]]
        self.adjacency_matrix_values: dict[str, Any]

        with open(SURVEY_VALUES, "r") as survey_values:
            self.survey_values = json.load(survey_values)

        with open(SURVEY_TYPES, "r") as survey_types:
            self.survey_question_types = json.load(survey_types)

        with open(HIERARCHY_MATRIX_VALUES, "r") as hierarchy_values:
            self.adjacency_matrix_values = json.load(hierarchy_values)

        self.surveys: dict[str, pl.DataFrame] = {}

        for key, value in RESPONSE_CSV.items():
            with open(value, "r") as file:
                self.surveys[key] = pl.read_csv(file)

    def generate_agents(self) -> None:
        """
        Calculates the weighting that each Agent gives to each social hierarchy. Also generates random weightings
        for those without preferential information.

        Additionally calculates or stochastically generates other important agent attributes from the given information.
        """
        dependant_total: float = float(self.survey_values["DependantTotal"])
        religious_total: float = float(self.survey_values["ReligiousTotal"])
        cultural_total: float = float(self.survey_values["CulturalTotal"])

        for community, survey in self.surveys.items():
            for agent_row in survey.iter_rows(named=True):
                agent_weightings: dict[str, float] = {}
                for hierarchy in HIERARCHIES:
                    agent_weightings[hierarchy] = 0.0

                agent_values: dict[str, Any] = {
                    "dependant_sum": 0.0,
                    "religious_sum": 0.0,
                    "cultural_sum": 0.0,
                }

                for column, value in agent_row.items():
                    if column == "AgentId":
                        agent_values["id"] = str(value)
                    elif column in self.survey_question_types["Dependant"]:
                        agent_values["dependant_sum"] += float(
                            self.survey_values[column][value]
                        )
                    elif column in self.survey_question_types["Age"]:
                        agent_values["age"] = str(value)
                    elif column in self.survey_question_types["Gender"]:
                        agent_values["gender"] = str(value)
                    elif column in self.survey_question_types["Religious"]:
                        agent_values["religious"] = str(value)
                    elif column in self.survey_question_types["ReligiousStrength"]:
                        agent_values["religious_sum"] += float(
                            self.survey_values[column][value]
                        )
                    elif column in self.survey_question_types["Cultural"]:
                        match column:
                            case "Q20":  # Willing to participate in dialogue
                                agent_values["dialogue"] = str(value)
                            case "Q22":  # Preferred sport
                                agent_values["sport"] = str(value)
                            case "Q29":  # Favourite music genre
                                agent_values["music"] = str(value)
                            case _:
                                pass
                    elif column in self.survey_question_types["CulturalStrength"]:
                        agent_values["cultural_sum"] += float(
                            self.survey_values[column][value]
                        )
                    elif column in self.survey_question_types["Time"]:
                        agent_values["time"] = str(value)
                    else:  # "General"
                        match column:
                            case "Q17":  # Specific influencing factor
                                agent_values["influencing_factor"] = str(value)
                            case "Q28":  # Preferred activity to participate in
                                agent_values["activity"] = str(value)
                            case _:
                                pass

                # Assign the Agent to all the relevant clusters
                self.hierarchy_clusters[community]["Age"][agent_values["age"]].append(
                    agent_values["id"]
                )
                self.hierarchy_clusters[community]["Gender"][
                    agent_values["gender"]
                ].append(agent_values["id"])
                self.hierarchy_clusters[community]["Religious"][
                    agent_values["religious"]
                ].append(agent_values["id"])
                self.hierarchy_clusters[community]["Cultural_1"][
                    agent_values["dialogue"]
                ].append(agent_values["id"])
                self.hierarchy_clusters[community]["Cultural_2"][
                    agent_values["sport"]
                ].append(agent_values["id"])
                self.hierarchy_clusters[community]["Cultural_3"][
                    agent_values["music"]
                ].append(agent_values["id"])

                # Calculate the relevant hierarchy weightings
                agent_weightings["Age"] = rd.uniform(-1.0, 1.0)
                agent_weightings["Gender"] = rd.uniform(-1.0, 1.0)
                agent_weightings["Family"] = rd.uniform(-1.0, 1.0)
                agent_weightings["Friends"] = rd.uniform(-1.0, 1.0)
                if agent_values["religious"] == "Yes":
                    # Rescale the values from [0, 1] to [-1, 1]
                    agent_weightings["Religious"] = 2.0 * (
                        agent_values["religious_sum"] / religious_total - 0.5
                    )
                else:
                    pass  # This agent does not "exist" in the religious hierarchy
                if agent_values["dialogue"] == "Yes":
                    # Rescale the values from [0, 1] to [-1, 1]
                    agent_weightings["Cultural"] = 2.0 * (
                        agent_values["cultural_sum"] / cultural_total - 0.5
                    )
                elif agent_values["dialogue"] == "Maybe":
                    # Weaker range of [-0.75, 0.75]
                    agent_weightings["Cultural"] = 1.5 * (
                        agent_values["cultural_sum"] / cultural_total - 0.5
                    )
                else:
                    # Weakest range of [-0.5, 0.5]
                    agent_weightings["Cultural"] = (
                        agent_values["cultural_sum"] / cultural_total - 0.5
                    )

                # Generate the opinion, benefit, and sociability attributes
                agent_values["opinion"] = 2.0 * (
                    agent_values["dependant_sum"] / dependant_total - 0.5
                )
                agent_values["benefit"] = rd.choice([True, False])
                agent_personality: str = agt.draw_personality()
                agent_susceptibility: float = rd.uniform(-1.0, 1.0)
                agent_values["sociability"] = (agent_personality, agent_susceptibility)

                # Append all the relevant attributes to self.agents
                agent_attributes: AgentAttributes = AgentAttributes(
                    agent_values["id"],
                    agent_values["age"],
                    agent_values["gender"],
                    agent_weightings,
                    agent_values["opinion"],
                    agent_values["benefit"],
                    agent_values["sociability"],
                )

                self.agents[community].append(agent_attributes)
        return None

    def create_base_hierarchies(self) -> None:
        """
        Creates the hierarchy graphs for the elemental relationships (Age and Gender)
        """
        for community in self.surveys.keys():
            num_agents: int = len(self.agents[community])
            age_matrix: np.ndarray = np.zeros(
                (num_agents, num_agents), dtype=np.float32
            )
            gender_matrix: np.ndarray = np.zeros(
                (num_agents, num_agents), dtype=np.float32
            )

            # Collect all the agent IDs in a list to add to the graph later
            agent_ids: list[str] = []

            # Populate the age and gender matrices
            for i, agent_i in enumerate(self.agents[community]):
                for j, agent_j in enumerate(self.agents[community]):
                    if i == j:
                        continue

                    age_difference: int = abs(
                        self.survey_values["Q01"][agent_i.age]
                        - self.survey_values["Q01"][agent_j.age]
                    )
                    age_weighting: float = (
                        1.0 - age_difference * 0.25
                    )  # 1.0 at no difference, and -0.25 at maximum difference
                    # Set the weighting in the age matrix
                    age_matrix[i, j] = age_weighting

                    # Set the weighting in the gender matrix
                    if agent_i.gender == agent_j.gender:
                        gender_matrix[i, j] = 1.0
                    else:  # Explictly defining that the weighting should remain 0.0
                        gender_matrix[i, j] = 0.0

                # Append the Agent ID
                agent_ids.append(agent_i.id)

            # After both matrices are populated, add them to self.graphs
            self.graphs[community]["hierarchies"].append("Age")
            self.graphs[community]["adj_matrices"].append(age_matrix)
            self.graphs[community]["rw_params"].append(HIERARCHY_RW["Age"])
            self.graphs[community]["agents"].append(agent_ids)

            self.graphs[community]["hierarchies"].append("Gender")
            self.graphs[community]["adj_matrices"].append(gender_matrix)
            self.graphs[community]["rw_params"].append(HIERARCHY_RW["Gender"])
            self.graphs[community]["agents"].append(agent_ids)
        return None

    def generate_relationship_strengths(
        self, adj_matrix: np.ndarray, hierarchy: str | None = None
    ) -> np.ndarray:
        """
        A helper function that transforms the codified values of an adjacency matrix to float values representing
        the relationship strengths.

        :param adj_matrix: The codified social hierarchy adjacency matrix.
        :param hierarchy: An optional string indicating that explicit conversion values are used for this hierarchy.
        :return: The modified adjacency matrix.
        """
        if hierarchy:
            for i in range(adj_matrix.shape[0]):
                for j in range(adj_matrix.shape[1]):
                    if i == j:
                        # Ensure that the diagonal is always zero
                        adj_matrix[i, j] = 0

                    matrix_value = self.adjacency_matrix_values[hierarchy][
                        f"{adj_matrix[i, j]}"
                    ]

                    # The value is "false"
                    if not matrix_value:
                        adj_matrix[i, j] = 0
                    # The value is a [min, max] generation range
                    else:
                        adj_matrix[i, j] = rd.uniform(matrix_value[0], matrix_value[1])
        else:
            min_value = np.min(adj_matrix)
            max_value = np.max(adj_matrix)
            mid_value: float = (max_value + min_value) / 2

            for i in range(adj_matrix.shape[0]):
                for j in range(adj_matrix.shape[1]):
                    if i == j:
                        continue
                    cell_value = adj_matrix[i, j]
                    if cell_value < mid_value:
                        adj_matrix[i, j] = -1.0 * ((mid_value - cell_value) / mid_value)
                    else:
                        adj_matrix[i, j] = (cell_value - mid_value) / mid_value
        return adj_matrix

    def generate_hierarchies(self) -> None:
        """
        Creates the remaining hierarchy graphs using the available information and the input adjacency matrices.
        """
        for community, adj_matrix_paths in ADJ_MATRIX_PATHS.items():
            num_agents: int = len(self.agents[community])

            for hierarchy, adj_matrix_path in adj_matrix_paths.items():
                adj_matrix = pl.read_csv(adj_matrix_path)

                # Remove the ID column, leaving only the codified adjacency matrix
                adj_matrix = adj_matrix.drop("Persona")
                adj_matrix = adj_matrix.to_numpy()

                # Convert the adjacency matrix from coded values to float strengths
                adj_matrix = self.generate_relationship_strengths(
                    adj_matrix, hierarchy=hierarchy
                )

                # In this case study, some matrices are sparse, but it is rare for any individual to not have
                # at least one relationship in a hierarchy
                agent_ids: list[str] = []
                for agent in self.agents[community]:
                    agent_ids.append(agent.id)

                self.graphs[community]["hierarchies"].append(hierarchy)
                self.graphs[community]["adj_matrices"].append(adj_matrix)
                self.graphs[community]["rw_params"].append(HIERARCHY_RW[hierarchy])
                self.graphs[community]["agents"].append(agent_ids)
        return None

    def write_agents(self) -> None:
        """
        Writes Agent objects for the models, serliases them to Pickle objects, and then saves them to
        the appropriate subdirectory.

        One subdirectory is created per community.
        """
        for community, agents_dir in AGENT_PATHS.items():
            # Create the agents subdirectory if needed
            if not os.path.exists(agents_dir):
                os.mkdir(agents_dir)

            for agent_attributes in self.agents[community]:
                new_agent: agt.Agent = agt.Agent(
                    agent_attributes.id,
                    agent_attributes.weightings,
                    agent_attributes.opinion,
                    agent_attributes.benefit,
                    agent_attributes.sociability,
                    age=agent_attributes.age,
                    gender=agent_attributes.gender,
                )

                self.agent_objects[community][new_agent.id] = deepcopy(new_agent)

                agent_path: str = f"{agents_dir}/agent_{new_agent.id}.pkl"

                with open(agent_path, "wb") as pickle_file:
                    pickle.dump(new_agent, pickle_file)

                # Manual garbage collection
                del new_agent, agent_path

        return None

    def write_graphs(self) -> None:
        """
        Writes Graph objects for the models, serialises their GraphNode and GraphEdge objects, and then
        saves these along with the graph's graphml file to the appropriate subdirectory.

        One subdirectory is created per hierarchy, per community.
        """
        for community, graphs_dir in GRAPH_PATHS.items():
            # Create the Graphs subdirectory if needed
            if not os.path.exists(graphs_dir):
                os.mkdir(graphs_dir)

            for idx, hierarchy in enumerate(self.graphs[community]["hierarchies"]):
                hierarchy_subdir: str = f"{graphs_dir}/{hierarchy}"

                if not os.path.exists(hierarchy_subdir):
                    os.mkdir(hierarchy_subdir)

                new_graph: gr.Graph = gr.Graph(
                    hierarchy,
                    self.graphs[community]["rw_params"][idx],
                )

                included_agents: list[agt.Agent] = []

                # Only select the Agents from the population which have been included in this hierarchy's agents list
                for agent_id, agent_obj in self.agent_objects[community].items():
                    if agent_id in self.graphs[community]["agents"][idx]:
                        included_agents.append(deepcopy(agent_obj))

                # Add the relevant Agent object as GraphNodes
                new_graph.add_nodes(included_agents)

                edges_dict: dict[str, list[int | float]] = {}
                edges_dict["to_node"] = []
                edges_dict["from_node"] = []
                edges_dict["weighting"] = []

                for i in range(len(included_agents)):
                    for j in range(len(included_agents)):
                        adj_matrix_value: float = float(
                            self.graphs[community]["adj_matrices"][idx][i, j]
                        )

                        # No valid relationship exists from i to j
                        if adj_matrix_value == 0.0:
                            continue

                        edges_dict["from_node"].append(i)
                        edges_dict["to_node"].append(j)
                        edges_dict["weighting"].append(adj_matrix_value)

                # Create the GraphEdge relationships using the information from the adjacency matrix
                new_graph.add_edges(edges_dict)

                # Serialise and save the graph object
                self.save_graph(new_graph, hierarchy_subdir)

                self.graph_objects[community].append(new_graph)

                # Manual garbage collection
                del new_graph
        return None

    def save_graph(self, graph_obj: gr.Graph, hierarchy_dir: str) -> None:
        """
        A helper function that handles the serialising and saving of a Graph object to a subdirectory within the input
        hierarchy subdirectory.

        :param graph_obj: The Graph object that is being serialised and saved.
        :param hierarchy_dir: The directory for this Graph's hierarchy within the community's `graphs' subdirectory.
        """
        graphml_path: str = f"{hierarchy_dir}/graph_{graph_obj.name}.graphml"

        # Begin by saving the Graph structure to a graphml format
        graph_obj.save_graph(graphml_path)

        # Next, serialise and save all GraphNodes
        nodes_subdir: str = f"{hierarchy_dir}/nodes"
        if not os.path.exists(nodes_subdir):
            os.mkdir(nodes_subdir)

        node_save_paths: list[str] = []

        for idx, node in enumerate(graph_obj.graph.nodes()):
            node_save_path: str = f"{nodes_subdir}/node_{idx}.pkl"
            with open(node_save_path, "wb") as node_file:
                pickle.dump(node, node_file)
            node_save_paths.append(node_save_path)

        # Finally, serialise and save all GraphEdges
        edges_subdir: str = f"{hierarchy_dir}/edges"
        if not os.path.exists(edges_subdir):
            os.mkdir(edges_subdir)

        edge_save_paths: list[str] = []

        for idx, edge in enumerate(graph_obj.graph.edges()):
            edge_save_path: str = f"{edges_subdir}/edge_{idx}.pkl"
            with open(edge_save_path, "wb") as edge_file:
                pickle.dump(edge, edge_file)
            edge_save_paths.append(edge_save_path)

        return None


if __name__ == "__main__":
    RESPONSE_CSV: dict[str, str] = {
        "NONMN": "./data/NONMN/NonMining.csv",
        "MINNG": "./data/MINNG/Mining.csv",
    }

    AGENT_PATHS: dict[str, str] = {
        "NONMN": "./gatoh/experiments/CaseStudy/Agents/NONMN_Agents",
        "MINNG": "./gatoh/experiments/CaseStudy/Agents/MINNG_Agents",
    }

    GRAPH_PATHS: dict[str, str] = {
        "NONMN": "./gatoh/experiments/CaseStudy/Graphs/NONMN_Graphs",
        "MINNG": "./gatoh/experiments/CaseStudy/Graphs/MINNG_Graphs",
    }

    HIERARCHIES: list[str] = [
        "Age",
        "Gender",
        "Friends",
        "Family",
        "Religious",
        "Cultural",
        "Geographical",
        "Social",
    ]

    HIERARCHY_RW: dict[str, tuple[float, float]] = {
        "Age": (0.0, 0.04),
        "Gender": (0.0, 0.02),
        "Friends": (0.0, 0.05),
        "Family": (0.0, 0.01),
        "Religious": (0.0, 0.1),
        "Cultural": (0.0, 0.15),
        "Geographical": (
            0.0,
            0.0,
        ),  # 0.0 var will be treated as hierarchy without dynamic relationships
        "Social": (0.0, 0.1),
    }

    ADJ_MATRIX_PATHS: dict[str, dict[str, str]] = {
        "NONMN": {
            "Friends": "./data/NONMN/NONMN_Friends.csv",
            "Family": "./data/NONMN/NONMN_Family.csv",
            "Cultural": "./data/NONMN/NONMN_Cultural.csv",
            "Religious": "./data/NONMN/NONMN_Religion.csv",
            "Geographical": "./data/NONMN/NONMN_Geographical.csv",
            "Social": "./data/NONMN/NONMN_Social.csv",
        },
        "MINNG": {
            "Friends": "./data/MINNG/MINNG_Friends.csv",
            "Family": "./data/MINNG/MINNG_Family.csv",
            "Cultural": "./data/MINNG/MINNG_Cultural.csv",
            "Religious": "./data/MINNG/MINNG_Religion.csv",
            "Geographical": "./data/MINNG/MINNG_Geographical.csv",
            "Social": "./data/MINNG/MINNG_Social.csv",
        },
    }

    SURVEY_VALUES: str = "./data/SurveyValues.json"
    SURVEY_TYPES: str = "./data/SurveyQuestionTypes.json"
    HIERARCHY_MATRIX_VALUES: str = "./data/MatrixValues.json"
