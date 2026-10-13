from __future__ import annotations

import json
import gc
import os
import pickle
import random as rd
from copy import deepcopy
from dataclasses import dataclass
from re import L
from typing import Any, TypedDict

import numpy as np
import polars as pl

import gatoh.agents as agt
import gatoh.graphs as gr
import gatoh.groups as grp


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


class GraphsDict(TypedDict):
    """
    A helper class for typechecking ResponseParser.graphs.
    """
    hierarchies: list[str]
    adj_matrices: list[np.ndarray]
    rw_params: list[tuple[float, float]]
    agents: list[list[str]]


class ResponseParser:
    """
    A class that is used to generate GATOH agents, graphs, and groups for the synthetic case study.

    The data used corresponds to the synthetically generated `survey responses' based on the distributions
    of the specific, anonymised surveys that were carried out by research collaborators.
    """
    def __init__(self) -> None:
        self.agents: dict[str, list[AgentAttributes]] = {}
        self.agent_objects: dict[str, dict[str, agt.Agent]] = {}
        self.graphs: dict[str, GraphsDict] = {}
        self.graph_objects: dict[str, list[gr.Graph]] = {}
        self.group_objects: dict[str, list[grp.Group]] = {}
        self.group_rels: dict[str, list[tuple[int, int]]] = {}
        self.hierarchy_clusters: dict[str, dict[str, dict[str, list[str]]]] = {}

        for key in RESPONSE_CSV:
            self.agents[key] = []
            # <ID: Agent> mapping to simplify lookups later
            self.agent_objects[key] = {}
            self.graphs[key] = {
                "hierarchies": [],  # String names
                "adj_matrices": [],  # 2D matrices indicating presence and strength of graph relationships (range [-1, 1])
                "rw_params": [],  # (mean, variance) defining the dynamic relationship random walk distributions
                "agents": [],  # List of IDs of all agents contained within the hierarchy
            }
            self.graph_objects[key] = []
            self.group_objects[key] = []
            self.group_rels[key] = []
            # All values are dictionaries that will represent <cluster : list[agent IDs]> within each hierarchy
            self.hierarchy_clusters[key] = {
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
                    "Voleyball": [],
                    "Athletics": [],
                    "Other": [],
                },
                "Cultural_3": {
                    # Favourite music genres
                    "Traditional": [],
                    "Popular": [],
                    "Religious": [],
                    "International": [],
                    "Varied": [],
                },
                "General": {
                    # "Wildcard" groupings that will affect existing relationships strengths across all hierarchies
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
        Calculates the weighting that each Agent gives to each social hierarchy.

        Also generates random weightings for those without preferential information.

        Additionally calculates or stochastically generates other important agent attributes from the given information.
        """
        dependant_total: float = float(self.survey_values["DependantTotal"])
        religious_total: float = float(self.survey_values["ReligiousTotal"])
        cultural_total: float = float(self.survey_values["CulturalTotal"])

        class AgentValues(TypedDict):
            """
            A helper class for typechecking agent_values.
            """
            dependant_sum: float
            religious_sum: float
            cultural_sum: float
            age: str
            id: str
            gender: str
            religious: str
            dialogue: str
            sport: str
            music: str
            time: str
            influencing_factor: str
            activity: str
            opinion: float
            benefit: bool
            sociability: tuple[str, float]

        for community, survey in self.surveys.items():
            for agent_row in survey.iter_rows(named=True):
                agent_weightings: dict[str, float] = {}
                for hierarchy in HIERARCHIES:
                    agent_weightings[hierarchy] = 0.0

                # Define all of the necessary Agent values and initialise them as empty
                agent_values: AgentValues = {
                    "dependant_sum": 0.0,
                    "religious_sum": 0.0,
                    "cultural_sum": 0.0,
                    "age": "",
                    "id": "",
                    "gender": "",
                    "religious": "",
                    "dialogue": "",
                    "sport": "",
                    "music": "",
                    "time": "",
                    "influencing_factor": "",
                    "activity": "",
                    "opinion": 0.0,
                    "benefit": False,
                    "sociability": ("", 0.0),
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
                        if str(value) == "No response":
                            agent_values["gender"] = "Other"
                        else:
                            agent_values["gender"] = str(value)
                    elif column in self.survey_question_types["Religious"]:
                        agent_values["religious"] = str(value)
                    elif column in self.survey_question_types["ReligiousStrength"]:
                        agent_values["religious_sum"] += float(
                            self.survey_values[column][value]
                        )
                    elif column in self.survey_question_types["Cultural"]:
                        match column:
                            case "Q20":
                                # Willing to participate in dialogue
                                agent_values["dialogue"] = str(value)
                            case "Q22":
                                # Preferred sport
                                agent_values["sport"] = str(value)
                            case "Q29":
                                # Favourite music genre
                                agent_values["music"] = self.survey_values["Q29"][value]
                            case _:
                                pass
                    elif column in self.survey_question_types["CulturalStrength"]:
                        agent_values["cultural_sum"] += float(
                            self.survey_values[column][value]
                        )
                    elif column in self.survey_question_types["Time"]:
                        agent_values["time"] = str(value)
                    else:
                        # "General"
                        match column:
                            case "Q17":
                                # Specific influencing factor
                                agent_values["influencing_factor"] = str(value)
                            case "Q28":
                                # Preferred type of activity to participate in
                                agent_values["activity"] = str(value)
                            case _:
                                pass

                # Assign the Agent to all the relevant clusters
                self.hierarchy_clusters[community]["Age"][agent_values["age"]].append(agent_values["id"])
                self.hierarchy_clusters[community]["Gender"][agent_values["gender"]].append(agent_values["id"])
                self.hierarchy_clusters[community]["Religious"][agent_values["religious"]].append(agent_values["id"])
                self.hierarchy_clusters[community]["Cultural_1"][agent_values["dialogue"]].append(agent_values["id"])
                self.hierarchy_clusters[community]["Cultural_2"][agent_values["sport"]].append(agent_values["id"])
                self.hierarchy_clusters[community]["Cultural_3"][agent_values["music"]].append(agent_values["id"])

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
                    # The agent does not "exist" in the religious hierarchy
                    pass
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
                    agent_weightings["Cultural"] = agent_values["cultural_sum"] / cultural_total - 0.5

                # Generate the opinion, benefit, and sociability attributes
                agent_values["opinion"] = 2.0 * (agent_values["dependant_sum"] / dependant_total - 0.5)
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

                self.agents[community].append(deepcopy(agent_attributes))

                # Manual garbage collection
                del agent_personality, agent_susceptibility, agent_attributes
                _ = gc.collect()

        return None

    def create_base_hierarchies(self) -> None:
        """
        Creates the hierarchy graphs for the elemental relationships (Age and Gender).
        """
        for community in self.surveys:
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
                    # 1.0 at no difference, and -0.25 at maximum difference
                    age_weighting: float = 1.0 - age_difference * 0.25
                    age_matrix[i, j] = age_weighting

                    # Set the weighting in the gender matrix
                    if agent_i.gender == agent_j.gender:
                        gender_matrix[i, j] = 1.0
                    else:
                        # Explicitly defining that the weighting should remain 0
                        gender_matrix[i, j] = 0.0

                # Append the Agent ID
                agent_ids.append(agent_i.id)

            # After both matrices are populated, add them to self.graphs
            self.graphs[community]["hierarchies"].append("Age")
            self.graphs[community]["adj_matrices"].append(deepcopy(age_matrix))
            self.graphs[community]["rw_params"].append(HIERARCHY_RW["Age"])
            self.graphs[community]["agents"].append(deepcopy(agent_ids))

            self.graphs[community]["hierarchies"].append("Gender")
            self.graphs[community]["adj_matrices"].append(deepcopy(gender_matrix))
            self.graphs[community]["rw_params"].append(HIERARCHY_RW["Gender"])
            self.graphs[community]["agents"].append(deepcopy(agent_ids))
        return None

    def generate_relationship_strengths(
        self,
        adj_matrix: np.ndarray,
        hierarchy: str | None = None,
        community: str | None = None,
    ) -> np.ndarray:
        """
        A helper function that transforms the codified values of an adjacency matrix to float values representing
        the relationship strengths.

        :param adj_matrix: The codified social hierarchy adjacency matrix.
        :type adj_matrix: :class:`~numpy.ndarray`
        :param hierarchy: An indication that explicit conversion values are used for the hierarchy with the input name.
        :type hierarchy: str, optional
        :param community: The community that these strengths are being generated for.
        :type community: str, optional
        :return: The modified adjacency matrix.
        :rtype: :class:`~numpy.ndarray`
        """
        new_adj_matrix = adj_matrix.copy()
        if hierarchy is not None:
            for i in range(adj_matrix.shape[0]):
                for j in range(adj_matrix.shape[1]):
                    if i == j:
                        # Ensure that the diagonal is always zero
                        new_adj_matrix[i, j] = 0.0

                    matrix_value = self.adjacency_matrix_values[hierarchy][f"{int(adj_matrix[i, j])}"]

                    if not matrix_value:
                        new_adj_matrix[i, j] = 0.0
                    else:
                        # The value is a [min, max] generation range
                        inclusion_threshold: float = rd.random()
                        # Somewhat restrict the relationship density to maintain reasonable model runtimes
                        # (No need to account for different community sizes as the synthetic data are the same size)
                        if inclusion_threshold <= 0.05:
                            new_value = rd.uniform(matrix_value[0], matrix_value[1])
                            new_adj_matrix[i, j] = new_value
                        else:
                            new_adj_matrix[i, j] = 0.0
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
                        new_adj_matrix[i, j] = -1.0 * ((mid_value - cell_value) / mid_value)
                    else:
                        new_adj_matrix[i, j] = (cell_value - mid_value) / mid_value
        return new_adj_matrix

    def generate_hierarchies(self) -> None:
        """
        Creates the remaining hierarchy graphs using the available information and the input adjacency matrices.
        """
        for community, adj_matrix_paths in ADJ_MATRIX_PATHS.items():
            for hierarchy in adj_matrix_paths:
                adj_matrix: np.ndarray
                match hierarchy:
                    case "Friends":
                        # Values of 0 to 4
                        adj_matrix = self.generate_matrix(4, len(self.agents[community]))
                    case "Family":
                        # Values of 0 to 4
                        adj_matrix = self.generate_matrix(4, len(self.agents[community]))
                    case "Cultural":
                        # Values of 0 to 2
                        adj_matrix = self.generate_matrix(2, len(self.agents[community]))
                    case "Religious":
                        # Values of 0 to 2
                        adj_matrix = self.generate_matrix(2, len(self.agents[community]))
                    case "Geographical":
                        # Values of 0 to 3
                        adj_matrix = self.generate_matrix(3, len(self.agents[community]))
                    case "Social":
                        # Values of 0 to 2
                        adj_matrix = self.generate_matrix(2, len(self.agents[community]))
                    case _:
                        adj_matrix = np.zeros((len(self.agents[community]), len(self.agents[community])))

                adj_matrix = self.generate_relationship_strengths(
                    adj_matrix, hierarchy=hierarchy, community=community,
                )

                agent_ids: list[str] = []
                for agent in self.agents[community]:
                    agent_ids.append(agent.id)

                self.graphs[community]["hierarchies"].append(hierarchy)
                self.graphs[community]["adj_matrices"].append(deepcopy(adj_matrix))
                self.graphs[community]["rw_params"].append(HIERARCHY_RW[hierarchy])
                self.graphs[community]["agents"].append(deepcopy(agent_ids))
        return None

    def generate_matrix(self, max_value: int, matrix_size: int) -> np.ndarray:
        """
        A helper function that generates random NxN matrices with zeros along the diagonal, and values
        [0, max_value] everywhere else.

        :param max_value: The maximum value that non-diagonals can take (inclusive).
        :type max_value: int
        :param matrix_size: The size of the square matrix to be generated.
        :type matrix_size: int
        :return: The generated matrix
        :rtype: ndarray
        """
        new_matrix: np.ndarray = np.zeros((matrix_size, matrix_size), dtype=np.float32)
        for i in range(matrix_size):
            for j in range(matrix_size):
                if i == j:
                    continue
                new_matrix[i, j] = round(rd.uniform(0, max_value))
        return new_matrix

    def generate_groups(self) -> None:
        """
        Clusters the generated hierarchies to create groups of agents for each instance.
        """
        for community, hierarchies in self.graph_objects.items():
            created_groups: list[grp.Group] = []
            group_relationships: list[tuple[int, int]] = []

            group_count: int = 0

            for graph in hierarchies:
                clustered_nodes: dict[gr.GraphNode, int] = graph.cluster_nodes(k=N_CLUSTERS)

                # Re-organise the nodes in clusters into clustered agents
                group_members: dict[int, list[agt.Agent]] = {}
                for node, cluster in clustered_nodes.items():
                    group_members.setdefault(cluster, []).append(node.agent)

                graph_groups: list[grp.Group] = []

                for cluster, members in group_members.items():
                    new_group: grp.Group = grp.Group()
                    _ = new_group.generate_group(
                        f"G{community}{group_count + 1:03}",
                        cluster,
                        hierarchy=graph.name,
                        members=members,
                    )
                    new_group.set_index(group_count)
                    group_count += 1
                    graph_groups.append(new_group)

                created_groups.extend(deepcopy(graph_groups))
                group_relationships.extend(graph.generate_group_edges(graph_groups))

                # Manual garbage collection
                del clustered_nodes, group_members, graph_groups
                _ = gc.collect()

            self.group_objects[community] = deepcopy(created_groups)
            self.group_rels[community] = deepcopy(group_relationships)

            # Manual garbage collection
            del group_relationships, created_groups
            _ = gc.collect()
        return None

    def write_agents(self) -> None:
        """
        Writes the Agent objects for the models, serialises them to Pickle objects, and then saves them to
        the appropriate subdirectory.

        One subdirectory is created per community.
        """
        if not os.path.exists("./experiments/SyntheticCaseStudy/Agents"):
            os.mkdir("./experiments/SyntheticCaseStudy/Agents")

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
                _ = gc.collect()

        return None

    def write_graphs(self) -> None:
        """
        Writes Graph objects for the models, serialises their GraphNode and GraphEdge objects, and then
        saves these along with the graph's graphml file to the appropriate subdirectory.

        One subdirectory is created per hierarchy, per community.
        """
        if not os.path.exists("./experiments/SyntheticCaseStudy/Graphs"):
            os.mkdir("./experiments/SyntheticCaseStudy/Graphs")

        for community, graphs_dir in GRAPH_PATHS.items():
            # Create the graph subdirectory if needed
            if not os.path.exists(graphs_dir):
                os.mkdir(graphs_dir)

            for idx, hierarchy in enumerate(self.graphs[community]["hierarchies"]):
                if hierarchy in ["Age", "Gender"]:
                    continue

                hierarchy_subdir: str = f"{graphs_dir}/{hierarchy}"

                if not os.path.exists(hierarchy_subdir):
                    os.mkdir(hierarchy_subdir)

                new_graph: gr.Graph = gr.Graph(
                    hierarchy,
                    self.graphs[community]["rw_params"][idx],
                    suppress_warnings=True,
                )

                included_agents: list[agt.Agent] = []

                # Only select the Agents from the population which have been included in this hierarchy's agents list
                for agent_id, agent_obj in self.agent_objects[community].items():
                    if agent_id in self.graphs[community]["agents"][idx]:
                        included_agents.append(deepcopy(agent_obj))

                # Add the relevant Agent objects as GraphNodes
                new_graph.add_nodes(included_agents)

                edges_dict: dict[str, list[int | float]] = {
                    "from_node": [],
                    "to_node": [],
                    "weighting": [],
                }

                for i in range(len(included_agents)):
                    for j in range(len(included_agents)):
                        adj_matrix_value: float = float(self.graphs[community]["adj_matrices"][idx][i, j])

                        if adj_matrix_value == 0.0:
                            # No valid relationship exists from i to j
                            continue
                        else:
                            edges_dict["from_node"].append(i)
                            edges_dict["to_node"].append(j)
                            edges_dict["weighting"].append(adj_matrix_value)

                # Create the GraphEdge relationships using the information from the adjacency matrix
                new_graph.add_edges(edges_dict)

                # Serialise and save the graph object
                self.save_graph(new_graph, hierarchy_subdir)

                self.graph_objects[community].append(deepcopy(new_graph))

                # Manual garbage collection
                del new_graph
                _ = gc.collect()

        return None

    def write_groups(self) -> None:
        """
        Serialises the Group objects for each model, and saves them to the appropriate subdirectory.

        Will also save the needed relationship information for these groups to the root subdirectory of the model.

        One subdirectory is created per community.
        """
        if not os.path.exists("./experiments/SyntheticCaseStudy/Groups"):
            os.mkdir("./experiments/SyntheticCaseStudy/Groups")

        for community, grp_dir in GROUP_PATHS.items():
            # Create the group subdirectory if needed
            if not os.path.exists(grp_dir):
                os.mkdir(grp_dir)

            for group in self.group_objects[community]:
                group_path: str = f"{grp_dir}/group_{group.id}.pkl"

                with open(group_path, "wb") as pickle_file:
                    pickle.dump(group, pickle_file)

            # Additionally, pickle the relationships for each community
            with open(f"{grp_dir}/group_edges.pkl", "wb") as pickle_file:
                pickle.dump(self.group_rels[community], pickle_file)

        return None

    def save_graph(self, graph_obj: gr.Graph, hierarchy_dir: str) -> None:
        """
        A helper function that handles the serialising and saving of a Graph object to a subdirectory within the input
        hierarchy subdirectory.

        :param graph_obj: The graph that is being serialised and saved.
        :type graph_obj: ;class:`~gatoh.graphs.Graph`
        :param hierarchy_dir: The directory for this Graph's hierarchy within the commmunity's "graphs" subdirectory.
        :type hierarchy_dir: str
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
            with open(node_save_path, "wb") as pickle_file:
                pickle.dump(node, pickle_file)
            node_save_paths.append(node_save_path)

        # Finally, serialise and save all GraphEdges
        edges_subdir: str = f"{hierarchy_dir}/edges"
        if not os.path.exists(edges_subdir):
            os.mkdir(edges_subdir)

        edge_save_paths: list[str] = []

        for idx, edge in enumerate(graph_obj.graph.edges()):
            edge_save_path: str = f"{edges_subdir}/edge_{idx}.pkl"
            with open(edge_save_path, "wb") as pickle_file:
                pickle.dump(edge, pickle_file)
            edge_save_paths.append(edge_save_path)

        return None


if __name__ == "__main__":
    RESPONSE_CSV: dict[str, str] = {
        "NONMN": "./experiments/SyntheticCaseStudy/NonMining.csv",
        "MINNG": "./experiments/SyntheticCaseStudy/Mining.csv",
    }

    AGENT_PATHS: dict[str, str] = {
        "NONMN": "./experiments/SyntheticCaseStudy/Agents/NONMN_Agents",
        "MINNG": "./experiments/SyntheticCaseStudy/Agents/MINNG_Agents",
    }

    GRAPH_PATHS: dict[str, str] = {
        "NONMN": "./experiments/SyntheticCaseStudy/Graphs/NONMN_Graphs",
        "MINNG": "./experiments/SyntheticCaseStudy/Graphs/MINNG_Graphs",
    }

    GROUP_PATHS: dict[str, str] = {
        "NONMN": "./experiments/SyntheticCaseStudy/Groups/NONMN_Groups",
        "MINNG": "./experiments/SyntheticCaseStudy/Groups/MINNG_Groups",
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
        # (0.0, 0.0) will be treated as a hierarchy without dynamic relationships
        "Geographical": (0.0, 0.0),
        "Social": (0.0, 0.1),
    }

    # The number of clusters to attempt to split each hierarchy into when generating agent groups for the graphs
    N_CLUSTERS: int = 10

    ADJ_MATRIX_PATHS: dict[str, dict[str, str]] = {
        "NONMN": {
            "Friends": "./experiments/SyntheticCaseStudy/NONMN_Friends.csv",
            "Family": "./experiments/SyntheticCaseStudy/NONMN_Family.csv",
            "Cultural": "./experiments/SyntheticCaseStudy/NONMN_Cultural.csv",
            "Religious": "./experiments/SyntheticCaseStudy/NONMN_Religious.csv",
            "Geographical": "./experiments/SyntheticCaseStudy/NONMN_Geographical.csv",
            "Social": "./experiments/SyntheticCaseStudy/NONMN_Social.csv",
        },
        "MINNG": {
            "Friends": "./experiments/SyntheticCaseStudy/MINNG_Friends.csv",
            "Family": "./experiments/SyntheticCaseStudy/MINNG_Family.csv",
            "Cultural": "./experiments/SyntheticCaseStudy/MINNG_Cultural.csv",
            "Religious": "./experiments/SyntheticCaseStudy/MINNG_Religious.csv",
            "Geographical": "./experiments/SyntheticCaseStudy/MINNG_Geographical.csv",
            "Social": "./experiments/SyntheticCaseStudy/MINNG_Social.csv",
        },
    }

    SURVEY_VALUES: str = "./data/SurveyValues.json"
    SURVEY_TYPES: str = "./data/SurveyQuestionTypes.json"
    HIERARCHY_MATRIX_VALUES: str = "./data/MatrixValues.json"

    response_parser: ResponseParser = ResponseParser()
    response_parser.generate_agents()
    response_parser.create_base_hierarchies()
    response_parser.generate_hierarchies()
    response_parser.write_agents()
    response_parser.write_graphs()
    response_parser.generate_groups()
    response_parser.write_groups()
