from __future__ import annotations

import gc
import os
import random as rd
from copy import deepcopy

from multiprocessing import Pool
# For type checking
from multiprocessing.pool import Pool as WorkerPool

from typing import TypedDict

import numpy as np

import gatoh.agents as agt
import gatoh.graphs as gr
import gatoh.groups as grp
import gatoh.model as md


class GroupsInfo(TypedDict):
    """
    A helper class used for accurate type checking of groups info results.
    """
    groups: list[grp.Group]
    edges: list[tuple[int, int]]


class InfluentialTester:
    """
    A test class that sets up and runs models for a low-influence (LI) scenrario and a high-influence (HI) scenario for
    an experimental comparison.

    In both cases, the models are composed of 5 hierarchies each with unique random walk distributions for their dynamic
    relationships. Additionally, both models will have 40 agents within them, and run for a total of 100 iterations.

    In the case of the LI model, all 40 agents are "low influence" agents with somewhat weaker relationships, and with
    hierarchy graphs having significantly less relationship densitites between agents.

    In the case of the HI model, 36 agents are "low influence" whilst 4 are replaced with "high influence" agents.
    These high influence agents have stronger relationships with their neighbours, and a significantly higher
    relationship density between themselves and their neighbours in the hierarchies.
    """

    def __init__(self, existing: bool = False) -> None:
        """
        :param existing: A flag indicating if the tester is loading existing experiment results.
        :type existing: bool, optional
        """
        # Store the class parameters within the instance
        self.n_agents: int = TEST_PARAMETERS["n_agents"]
        self.n_groups: int = TEST_PARAMETERS["n_groups"]
        self.n_negative: int = TEST_PARAMETERS["n_negative"]

        self.existing: bool = existing

        # Define the data types without assigning values
        self.li_agents: list[agt.Agent]
        self.hi_agents: list[agt.Agent]
        self.li_graphs: list[gr.Graph]
        self.hi_graphs: list[gr.Graph]
        self.li_groups: list[grp.Group]
        self.li_group_edges: list[tuple[int, int]]
        self.hi_groups: list[grp.Group]
        self.hi_group_edges: list[tuple[int, int]]

        # Create the model objects no matter what
        self.li_model: md.ABModel = md.ABModel(
            deepcopy(HIERARCHY_NAMES),
            deepcopy(HIERARCHY_RW_DISTRIBUTIONS),
            iterations=MODEL_PARAMETERS["iterations"],
            silencing_threshold=MODEL_PARAMETERS["silencing_thresh"],
            negation_threshold=MODEL_PARAMETERS["negation_thresh"],
            radicalisation_threshold=MODEL_PARAMETERS["radical_thresh"],
            simulate_groups=True,
            save_dir=LI_SAVEDIR,
            data_file=LI_DATAFILE,
            model_id="LI_MODEL",
        )
        self.hi_model: md.ABModel = md.ABModel(
            deepcopy(HIERARCHY_NAMES),
            deepcopy(HIERARCHY_RW_DISTRIBUTIONS),
            iterations=MODEL_PARAMETERS["iterations"],
            silencing_threshold=MODEL_PARAMETERS["silencing_thresh"],
            negation_threshold=MODEL_PARAMETERS["negation_thresh"],
            radicalisation_threshold=MODEL_PARAMETERS["radical_thresh"],
            simulate_groups=True,
            save_dir=HI_SAVEDIR,
            data_file=HI_DATAFILE,
            model_id="HI_MODEL",
        )

        # Objects are needed to define and run the models
        if not self.existing:
            # Create the agents
            self.li_agents = self.create_li_agents()
            self.hi_agents = self.create_hi_agents()

            # Create the graphs
            self.li_graphs = self.create_li_graphs(
                deepcopy(HIERARCHY_NAMES),
                deepcopy(HIERARCHY_RW_DISTRIBUTIONS),
                self.li_agents,
            )
            self.hi_graphs = self.create_hi_graphs(
                deepcopy(HIERARCHY_NAMES),
                deepcopy(HIERARCHY_RW_DISTRIBUTIONS),
                self.hi_agents,
            )

            # Create the groups
            li_groups_info = self.create_li_groups()
            hi_groups_info = self.create_hi_groups()

            # Separate the relevant information for groups
            self.li_groups = li_groups_info["groups"]
            self.li_group_edges = li_groups_info["edges"]
            self.hi_groups = hi_groups_info["groups"]
            self.hi_group_edges = hi_groups_info["edges"]

            # Manual garbage collection
            del li_groups_info, hi_groups_info
            _ = gc.collect()


    def create_li_agents(self) -> list[agt.Agent]:
        """
        Creates the population of agents to be used in the LI scenario.

        :return: The generated LI agents.
        :rtype: list[Agent]
        """
        created_li_agents: list[agt.Agent] = []

        nn_opinion_range: tuple[float, float] = AGENT_CHARACTERISTICS["non_negative_opinion"]
        n_opinion_range: tuple[float, float] = AGENT_CHARACTERISTICS["negative_opinion"]

        agent_behaviour: tuple[str, float] = (
            AGENT_CHARACTERISTICS["personality"],
            AGENT_CHARACTERISTICS["social_susceptibility"],
        )

        # Define the data types but do not assign any values
        agent_id: str
        agent_opinion: float
        agent: agt.Agent

        created_count: int = 0
        while created_count < self.n_agents - self.n_negative:
            created_count += 1
            agent_id = f"NONN{created_count:04}"

            agent_opinion = rd.uniform(nn_opinion_range[0], nn_opinion_range[1])

            agent = agt.Agent(
                agent_id,
                deepcopy(HIERARCHY_WEIGHTINGS),
                agent_opinion,
                agent_behaviour,
                AGENT_CHARACTERISTICS["personal_benefit"],
            )

            created_li_agents.append(agent)

        created_count = 0
        while created_count < self.n_negative:
            created_count += 1
            agent_id = f"NGTV{created_count:04}"

            agent_opinion = rd.uniform(n_opinion_range[0], n_opinion_range[1])

            agent = agt.Agent(
                agent_id,
                deepcopy(HIERARCHY_WEIGHTINGS),
                agent_opinion,
                agent_behaviour,
                AGENT_CHARACTERISTICS["personal_benefit"],
            )

            created_li_agents.append(agent)

        return created_li_agents

    def create_hi_agents(self) -> list[agt.Agent]:
        """
        Creates the population of agents for the HI scenario.

        :return: The generated agent objects for HI.
        :rtype: list[Agent]
        """
        created_hi_agents: list[agt.Agent] = []

        nn_agents: int = self.n_agents - self.n_negative
        nn_opinion_range: tuple[float, float] = AGENT_CHARACTERISTICS["non_negative_opinion"]

        agent_behaviour: tuple[str, float] = (
            AGENT_CHARACTERISTICS["personality"],
            AGENT_CHARACTERISTICS["social_susceptibility"],
        )

        created_count: int = 0
        while created_count < nn_agents:
            created_count += 1
            nn_agent_id: str = f"NONN{created_count:04}"

            nn_agent_opinion: float = rd.uniform(nn_opinion_range[0], nn_opinion_range[1])

            nn_agent: agt.Agent = agt.Agent(
                nn_agent_id,
                deepcopy(HIERARCHY_WEIGHTINGS),
                nn_agent_opinion,
                agent_behaviour,
                AGENT_CHARACTERISTICS["personal_benefit"],
            )
            created_hi_agents.append(nn_agent)

        n_opinion_range: tuple[float, float] = AGENT_CHARACTERISTICS["negative_opinion"]

        created_count = 0
        while created_count < self.n_negative:
            created_count += 1
            n_agent_id: str = f"INFN{created_count:04}"

            n_agent_opinion: float = rd.uniform(n_opinion_range[0], n_opinion_range[1])

            n_agent: agt.Agent = agt.Agent(
                n_agent_id,
                deepcopy(HIERARCHY_WEIGHTINGS),
                n_agent_opinion,
                agent_behaviour,
                AGENT_CHARACTERISTICS["personal_benefit"],
            )
            created_hi_agents.append(n_agent)

        return created_hi_agents

    def create_li_graphs(
        self,
        hierarchies: list[str],
        rw_distributions: list[tuple[float, float]],
        agents: list[agt.Agent],
    ) -> list[gr.Graph]:
        """
        Creates the graphs for the LI model.

        For the purposes of this experiment, each graph contains all agents in the population,
        and the relationships are created following a customised scale-free methodology.

        :param hierarchies: The hierarchies that all graphs should represent.
        :type hierarchies: list[str]
        :param rw_distributions: The random walk distribution parameters for each social hierarchy.
        :type rw_distributions: list[tuple[float, float]]
        :param agents: The population of agents from which the graphs will be constructed.
        :type agents: list[Agent]
        :return: The created graph objects.
        :rtype: list[Graph]
        """
        created_li_graphs: list[gr.Graph] = []

        # Create a set of the agent indices to be used later in iteration
        agent_indices: set[int] = {i for i in range(len(agents))}

        # The valid range of relationship strengths originating from noninfluential agents
        noninf_rel_range: tuple[float, float] = AGENT_CHARACTERISTICS["relationship"]

        for idx, hierarchy in enumerate(hierarchies):
            graph: gr.Graph = gr.Graph(hierarchy, rw_distributions[idx])

            # Initialise the graph nodes using the population of agents
            graph.add_nodes(deepcopy(agents))

            new_edges: dict[str, list[int | float]] = {
                "from_node": [],
                "to_node": [],
                "weighting": [],
            }

            for agent_node in graph.graph.nodes():
                # Return a filtered list including all agent indices except for the current node
                valid_indices: list[int] = list(agent_indices ^ {agent_node.index})

                selected_indices: list[int] = list(
                    np.random.choice(
                        valid_indices,
                        size=AGENT_CHARACTERISTICS["non_influential_connectivity"],
                        replace=False,
                    )
                )

                for selected_index in selected_indices:
                    edge_weighting: float = rd.uniform(noninf_rel_range[0], noninf_rel_range[1])

                    # Flip the to_ and from_ indices to make the weighting representative of the impact that agent_node has on others.
                    new_edges["from_node"].append(selected_index)
                    new_edges["to_node"].append(agent_node.index)
                    new_edges["weighting"].append(edge_weighting)

            graph.add_edges(new_edges)
            created_li_graphs.append(graph)

        return created_li_graphs

    def create_hi_graphs(
        self,
        hierarchies: list[str],
        rw_distributions: list[tuple[float, float]],
        agents: list[agt.Agent],
    ) -> list[gr.Graph]:
        """
        Creates the graphs for the HI model.

        For the purposes of this experiment, each graph contains all agents in the population,
        and the relationships are created following a customised scale-free methodology.

        :param hierarchies: The hierarchies that all of the created graphs should represent.
        :type hierarchies: list[str]
        :param rw_distributions: The random walk parameters for each social hierarchy.
        :type rw_distributions: list[tuple[float, float]]
        :param agents: The population of agents from which the graphs will be constructed.
        :type agents: list[Agent]
        :return: The created graph objects.
        :rtype: list[Graph]
        """
        created_hi_graphs: list[gr.Graph] = []

        # Calculate the number of noninfluential agents
        n_ni_agents: int = self.n_agents - self.n_negative

        # Create a set of the agent indices to be used later for iteration
        agent_indices: set[int] = {i for i in range(len(agents))}

        noninf_rel_range: tuple[float, float] = AGENT_CHARACTERISTICS["relationship"]
        inf_rel_range: tuple[float, float] = AGENT_CHARACTERISTICS["influential_relationship"]

        for idx, hierarchy in enumerate(hierarchies):
            graph: gr.Graph = gr.Graph(hierarchy, rw_distributions[idx])

            # Initialise the graph nodes using the full agent population
            graph.add_nodes(deepcopy(agents))

            new_edges: dict[str, list[int | float]] = {
                "from_node": [],
                "to_node": [],
                "weighting": [],
            }

            for agent_node in graph.graph.nodes():
                # Return a filtered list including all indices except the current one
                valid_indices: list[int] = list(agent_indices ^ {agent_node.index})

                # Declare data types but assign no values
                selected_indices: list[int]
                edge_weighting: float

                if agent_node.index < n_ni_agents:
                    # The agent is non-influential
                    selected_indices = list(
                        np.random.choice(
                            valid_indices,
                            size=AGENT_CHARACTERISTICS["non_influential_connectivity"],
                            replace=False,
                        )
                    )

                    for selected_index in selected_indices:
                        edge_weighting = rd.uniform(noninf_rel_range[0], noninf_rel_range[1])

                        # Flip the to_ and from_ indices to make the weighting representative of the agent_node's impact on others
                        new_edges["from_node"].append(selected_index)
                        new_edges["to_node"].append(agent_node.index)
                        new_edges["weighting"].append(edge_weighting)
                else:
                    # The agent is influential
                    selected_indices = list(
                        np.random.choice(
                            valid_indices,
                            size=AGENT_CHARACTERISTICS["influential_connectivity"],
                            replace=False,
                        )
                    )

                    for selected_index in selected_indices:
                        edge_weighting = rd.uniform(inf_rel_range[0], inf_rel_range[1])

                        # Flip the to_ and from_ indices to make the weighting representative of the agent_node's impact on others
                        new_edges["from_node"].append(selected_index)
                        new_edges["to_node"].append(agent_node.index)
                        new_edges["weighting"].append(edge_weighting)

            graph.add_edges(new_edges)
            created_hi_graphs.append(graph)

        return created_hi_graphs

    def create_li_groups(self) -> GroupsInfo:
        """
        Runs KMeans clustering and creates the aggregate groups for the LI model.

        :return: A <hierarchy : groups> mapping of the generated agent groups for each social hierarchy.
        :rtype: dict[str, dict[str, list[Group] | list[tuple[int, int]]]]
        """
        created_groups: GroupsInfo = {
            "groups": [],
            "edges": [],
        }

        group_count: int = 0

        for graph in self.li_graphs:
            clustered_nodes: dict[gr.GraphNode, int] = graph.cluster_nodes(k=self.n_groups)

            # Re-organise the nodes in clusters into clustered agents
            group_members: dict[int, list[agt.Agent]] = {}
            for node, cluster in clustered_nodes.items():
                group_members.setdefault(cluster, []).append(node.agent)

            graph_groups: list[grp.Group] = []

            for cluster, members in group_members.items():
                new_group: grp.Group = grp.Group()
                new_group.generate_group(
                    f"LIGRP{group_count + 1:04}",
                    cluster,
                    graph.name,
                    members,
                )
                group_count += 1
                _ = self.li_model.add_group(new_group)
                graph_groups.append(new_group)

            created_groups["groups"].extend(deepcopy(graph_groups))
            created_groups["edges"].extend(graph.generate_group_edges(graph_groups))

            # Manual garbage collection
            del clustered_nodes, group_members, graph_groups
            _ = gc.collect()

        return created_groups

    def create_hi_groups(self) -> GroupsInfo:
        """
        Runs KMeans clustering and creates the aggregate groups for the HI model.

        :return: A <hierarchy : groups> mapping of the generated agent groups for each social hierarchy.
        :rtype: dict[str, list[Group] | list[tuple[int, int]]]
        """
        created_groups: GroupsInfo = {
            "groups": [],
            "edges": [],
        }

        for graph in self.hi_graphs:
            clustered_nodes: dict[gr.GraphNode, int] = graph.cluster_nodes(k=self.n_groups)

            # Re-organise the nodes in clusters into clustered agents
            group_members: dict[int, list[agt.Agent]] = {}
            for node, cluster in clustered_nodes.items():
                group_members.setdefault(cluster, []).append(node.agent)

            graph_groups: list[grp.Group] = []

            for cluster, members in group_members.items():
                new_group: grp.Group = grp.Group()
                new_group.generate_group(
                    f"HIGRP{cluster:04}",
                    cluster,
                    graph.name,
                    members,
                )
                _ = self.hi_model.add_group(new_group)
                graph_groups.append(new_group)

            created_groups["groups"].extend(deepcopy(graph_groups))
            created_groups["edges"].extend(graph.generate_group_edges(graph_groups))

            # Manual garbage collection
            del clustered_nodes, group_members, graph_groups
            _ = gc.collect()

        return created_groups

    def load_models(self) -> None:
        """
        Loads the models that have been previously saved at their respective directories.
        """
        self.li_model.load_model(LI_SAVEDIR)
        self.hi_model.load_model(HI_SAVEDIR)
        return None

    def setup_models(self) -> None:
        """
        Adds the appropriate agent, graph, and group objects to both models.
        """
        _ = self.li_model.add_agents(self.li_agents)
        _ = self.li_model.add_graphs(
            self.li_graphs,
            deepcopy(HIERARCHY_NAMES),
            deepcopy(HIERARCHY_RW_DISTRIBUTIONS),
        )

        _ = self.hi_model.add_agents(self.hi_agents)
        _ = self.hi_model.add_graphs(
            self.hi_graphs,
            deepcopy(HIERARCHY_NAMES),
            deepcopy(HIERARCHY_RW_DISTRIBUTIONS),
        )

        from_group: grp.Group
        to_group: grp.Group
        for connection in self.li_group_edges:
            from_group = self.li_groups[connection[0]]
            to_group = self.li_groups[connection[1]]
            self.li_model.add_group_graph_edge(from_group, to_group)
        for connection in self.hi_group_edges:
            from_group = self.hi_groups[connection[0]]
            to_group = self.hi_groups[connection[1]]
            self.hi_model.add_group_graph_edge(from_group, to_group)

        return None

    def run_model_li(self, worker_pool: WorkerPool | None = None) -> None:
        """
        Runs the LI model.

        :param worker_pool: A pool of workers that can share the processing of the model iteration amongst themselves.
        :type worker_pool: :class:`~multiprocessing.pool.Pool`, optional
        """
        self.li_model.iterate(worker_pool=worker_pool)
        self.li_model.save_model()
        return None

    def run_model_hi(self, worker_pool: WorkerPool | None = None) -> None:
        """
        Runs the HI model.

        :param worker_pool: A pool of workers that can share the processing of the model iteration amongst themselves.
        :type worker_pool: :class:`~multiprocessing.pool.Pool`, optional
        """
        self.hi_model.iterate(worker_pool=worker_pool)
        self.hi_model.save_model()
        return None


if __name__ == "__main__":
    MULTIPROCESSED: bool = True
    WORKER_POOL: WorkerPool | None = Pool() if MULTIPROCESSED else None

    class TestParameters(TypedDict):
        n_agents: int
        n_groups: int
        n_negative: int

    # The parameters set for the tester class itself
    TEST_PARAMETERS: TestParameters = {
        "n_agents": 100,
        "n_negative": 10,
        "n_groups": 10,
    }

    class ModelParameters(TypedDict):
        iterations: int
        silencing_thresh: float
        radical_thresh: float
        negation_thresh: float

    # The model parameters used when creating the ABModel instance
    MODEL_PARAMETERS: ModelParameters = {
        "iterations": 100,
        "silencing_thresh": 0.95,
        "radical_thresh": 0.99,
        "negation_thresh": 0.999,
    }

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

    # The hierarchy weightings that each agent will assign to the social hierarchies.
    HIERARCHY_WEIGHTINGS: dict[str, float] = {
        "family": 0.9,
        "friends": 0.7,
        "religion": 0.5,
        "neighbours": 0.55,
        "cultural": 0.25,
    }

    class AgentCharacteristics(TypedDict):
        non_negative_opinion: tuple[float, float]
        negative_opinion: tuple[float, float]
        non_influential_connectivity: int
        relationship: tuple[float, float]
        influential_connectivity: int
        influential_relationship: tuple[float, float]
        social_susceptibility: float
        personality: str
        personal_benefit: bool

    # Defining the distributions of the agent characteristics for this experiment
    AGENT_CHARACTERISTICS: AgentCharacteristics = {
        "non_negative_opinion": (0.0, 0.8),
        "negative_opinion": (-1.0, -0.5),
        "non_influential_connectivity": 3,
        "relationship": (-0.4, 0.4),
        "influential_connectivity": 20,
        "influential_relationship": (-0.9, 0.9),
        "social_susceptibility": 0.5,
        "personality": "social",
        "personal_benefit": True,
    }

    # The root directory of this experiment
    ROOT_DIR: str = "./experiments/GroupBase/InfluentialAgents"

    # Define the save directories for each model
    LI_SAVEDIR: str = f"{ROOT_DIR}/InfluentialAgents_LI"
    HI_SAVEDIR: str = f"{ROOT_DIR}/InfluentialAgents_HI"

    # Define the data file paths for each model (must point to a .csv)
    LI_DATAFILE: str = f"{ROOT_DIR}/li_model_variables.csv"
    HI_DATAFILE: str = f"{ROOT_DIR}/hi_model_variables.csv"

    tester: InfluentialTester

    # At least one model save directory does not exist
    if not os.path.exists(LI_SAVEDIR) or not os.path.exists(HI_SAVEDIR):
        # Create the tester normally, setup the models, and begin iterations
        tester = InfluentialTester()
        tester.setup_models()
        tester.run_model_li(worker_pool=WORKER_POOL)
        tester.run_model_hi(worker_pool=WORKER_POOL)
    # Both models exist
    else:
        tester = InfluentialTester(existing=True)
        tester.load_models()

    # Ensure that the Pool is closed after all processing is finished
    if WORKER_POOL is not None:
        WORKER_POOL.terminate()
