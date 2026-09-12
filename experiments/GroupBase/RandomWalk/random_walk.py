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


class RandomWalkTester:
    """
    A test class that sets up models which use a variety of random walk distributions for their dynamic
    relationships and hierarchy weighting mechanisms.

    The main scenarios are:
        - A 'base' model with identical random walk parameters for all dynamic relationships and hierarchy weightings
        - 'HIER' -- identical parameters for dynamic relationships and unique parameters for hierarchy weightings
        - 'RELS' -- unique parameters for dynamic relationships and identical parameters for all hierarchy weightings
        - 'BOTH' -- unique parameters for both dynamic relationships and hierarchy weightings

    For each one of these models, there will also be an exploration of the effects that different strengths of parameters
    may have on emergent behaviour; i.e. positive vs. negative distribution means, small vs. large variances,
    parameter strength tied to agent influentiality, etc.

    Each instance will use identical model parameters, graphs, agent, and group populations; with only the random walk
    parameters being different between them.
    """

    def __init__(self, existing: bool = False) -> None:
        """
        :param existing: A flag indicating if the tester is loading existing experiment results.
        :type existing: bool, optional
        """
        self.model_names: list[str] = TEST_PARAMETERS["model_names"]
        self.n_agents: int = AGENT_PARAMETERS["n_agents"]
        self.n_groups: int = GROUP_PARAMETERS["n_groups"]

        self.existing: bool = existing

        # Dynamic model space
        self.models: dict[str, md.ABModel] = {}

        for model_name in self.model_names:
            new_model: md.ABModel = md.ABModel(
                TEST_PARAMETERS["hierarchy_names"],
                # The rw params passed to each model are overriden by the explicit ones set for agents...
                list(TEST_PARAMETERS["shared_hierarchy_rw"].values()),
                save_dir=SAVEDIRS[model_name],
                data_file=SAVEFILES[model_name],
                model_id=model_name,
                simulate_groups=True,
            )
            self.models[model_name] = deepcopy(new_model)

            # Manual garbage collection
            del new_model
            _ = gc.collect()

        # Define the dicts that will contain the populations of Agents, Graphs, and Groups
        self.model_agents: dict[str, list[agt.Agent]] = {}
        self.model_graphs: dict[str, list[gr.Graph]] = {}
        self.model_groups: dict[str, list[grp.Group]] = {}

        self.group_edges: list[tuple[int, int]] = []

        if not self.existing:
            self.model_agents["BASE"] = self.create_agents()
            # No changes in the Agents from the base case
            self.model_agents["RELS"] = deepcopy(self.model_agents["BASE"])

            # HIER model requires all agents to have unique hierarchy rw params
            self.model_agents["HIER"] = deepcopy(self.model_agents["BASE"])
            for agent in self.model_agents["HIER"]:
                agent.rw_distributions = TEST_PARAMETERS["unique_hierarchy_rw"]

            # Agents require unique hierarchy params
            self.model_agents["BOTH"] = deepcopy(self.model_agents["HIER"])

            print("==== All model Agents successfully created ====")

            # Create the BASE graphs
            self.model_graphs["BASE"] = self.create_graphs(self.model_agents["BASE"])

            self.model_graphs["RELS"] = deepcopy(self.model_graphs["BASE"])
            for graph in self.model_graphs["RELS"]:
                # First, update all the nodes with the appropriate Agent objects
                for idx, node in enumerate(graph.graph.nodes()):
                    node.agent = deepcopy(self.model_agents["RELS"][idx])
                # For this case, unique rw parameters must be generated for each edge
                for edge in graph.graph.edges():
                    generated_mean: float = rd.uniform(
                        TEST_PARAMETERS["unique_rel_rw_range"][0][0],
                        TEST_PARAMETERS["unique_rel_rw_range"][0][1],
                    )
                    generated_variance: float = rd.uniform(
                        TEST_PARAMETERS["unique_rel_rw_range"][1][0],
                        TEST_PARAMETERS["unique_rel_rw_range"][1][1],
                    )
                    edge.set_rw_params((generated_mean, generated_variance))

            self.model_graphs["HIER"] = deepcopy(self.model_graphs["BASE"])
            for graph in self.model_graphs["HIER"]:
                # Only the nodes must be updated for this case
                for idx, node in enumerate(graph.graph.nodes()):
                    node.agent = deepcopy(self.model_agents["HIER"][idx])

            self.model_graphs["BOTH"] = deepcopy(self.model_graphs["RELS"])
            for graph in self.model_graphs["BOTH"]:
                # Only the nodes must be updated as unique rels were already generated for RELS
                for idx, node in enumerate(graph.graph.nodes()):
                    node.agent = deepcopy(self.model_agents["BOTH"][idx])

            # Create the groups
            group_info: tuple[list[grp.Group], list[tuple[int, int]]] = self.create_groups(self.model_graphs["BASE"])
            self.model_groups["BASE"] = group_info[0]
            self.group_edges = group_info[1]

            # No changes in the groups from the base case
            self.model_groups["RELS"] = deepcopy(self.model_groups["BASE"])

            # HIER model requires all groups to have unique hierarchy rw params
            self.model_groups["HIER"] = deepcopy(self.model_groups["BASE"])
            for group in self.model_groups["HIER"]:
                group.change_rw_distribution(TEST_PARAMETERS["unique_hierarchy_rw"][group.hierarchy])

            # Groups require unique hierarchy params
            self.model_groups["BOTH"] = deepcopy(self.model_groups["HIER"])

            print("==== All model Graphs successfully created ====")

    def create_agents(self) -> list[agt.Agent]:
        """
        Generates the population of agents that will be used across all models.

        :return: The generated agent population.
        :rtype: list[Agent]
        """
        print("Starting Agent creation")
        created_agents: list[agt.Agent] = []

        personalities: list[str] = list(AGENT_PARAMETERS["personality"].keys())
        personality_p: list[float] = list(AGENT_PARAMETERS["personality"].values())

        benefit_flags: list[bool] = list(AGENT_PARAMETERS["personal_benefit"].keys())
        benefit_p: list[float] = list(AGENT_PARAMETERS["personal_benefit"].values())

        for i in range(self.n_agents):
            agent_id: str = f"{AGENT_PARAMETERS['id_base']}{i + 1:04}"
            agent_opinion: float = rd.uniform(AGENT_PARAMETERS["opinions"][0], AGENT_PARAMETERS["opinions"][1])
            agent_personality: str = str(np.random.choice(personalities, size=1, p=personality_p)[0])
            agent_susceptibility: float = rd.uniform(AGENT_PARAMETERS["social_susceptibility"][0], AGENT_PARAMETERS["social_susceptibility"][1])
            agent_behaviour: tuple[str, float] = (agent_personality, agent_susceptibility)
            agent_benefit: bool = bool(np.random.choice(benefit_flags, size=1, p=benefit_p)[0])

            hierarchy_weightings: dict[str, float] = {}
            for hierarchy_name in TEST_PARAMETERS["hierarchy_names"]:
                generated_weighting: float = rd.uniform(AGENT_PARAMETERS["hierarchy_weighting"][0], AGENT_PARAMETERS["hierarchy_weighting"][1])
                hierarchy_weightings[hierarchy_name] = generated_weighting

            created_agent: agt.Agent = agt.Agent(
                agent_id,
                agent_opinion,
                hierarchy_weightings,
                agent_behaviour,
                agent_benefit,
            )
            # Initialise the base population with shared dynamic hierarchy rw params
            created_agent.rw_distributions = TEST_PARAMETERS["shared_hierarchy_rw"]

            created_agents.append(created_agent)

        print("Finished Agent creation")
        return created_agents

    def create_graphs(self, agents: list[agt.Agent]) -> list[gr.Graph]:
        """
        Generates the social hierarchy graphs that will be used across the models.

        :param agents: The population of agents that will be used to generate the graphs.
        :type agents: list[Agent]
        :return: The generated social hierarchy graphs.
        :rtype: list[Graph]
        """
        print("Starting Graph creation")
        created_graphs: list[gr.Graph] = []

        for hierarchy in TEST_PARAMETERS["hierarchy_names"]:
            graph: gr.Graph = gr.Graph(hierarchy, TEST_PARAMETERS["shared_relationship_rw"])
            _ = graph.generate_graph(
                deepcopy(agents),
                method=TEST_PARAMETERS["graph_generation_alg"],
                relationship_range=AGENT_PARAMETERS["relationships"],
            )

            created_graphs.append(graph)

        print("Finished Graph creation")
        return created_graphs

    def create_groups(self, graphs: list[gr.Graph]) -> tuple[list[grp.Group], list[tuple[int, int]]]:
        """
        Runs KMeans clustering and generates the population of agent groups to be used across the models.

        :param graphs: The social hierarchy graphs which are being clustered for group creation.
        :type graphs: list[Graph]
        :return: The generated agent groups (clusters) and the relationships between them.
        :rtype: tuple[list[Group], list[tuple[int, int]]]
        """
        print("Starting Group creation")
        created_groups: list[grp.Group] = []
        group_relationships: list[tuple[int, int]] = []

        group_count: int = 0

        for graph in graphs:
            clustered_nodes: dict[gr.GraphNode, int] = graph.cluster_nodes(k=self.n_groups)

            # Re-organise the nodes in clusters into clustered agents
            group_members: dict[int, list[agt.Agent]] = {}
            for node, cluster in clustered_nodes.items():
                group_members.setdefault(cluster, []).append(node.agent)

            graph_groups: list[grp.Group] = []

            for cluster, members in group_members.items():
                new_group: grp.Group = grp.Group()
                new_group.generate_group(
                    f"GROUP{group_count + 1:04}",
                    cluster,
                    graph.name,
                    members,
                )
                # Index must be set here so that the group edge tuples have valid indices
                new_group.set_index(group_count)
                group_count += 1
                graph_groups.append(new_group)

            created_groups.extend(deepcopy(graph_groups))
            group_relationships.extend(graph.generate_group_edges(graph_groups))

            # Manual garbage collection
            del clustered_nodes, group_members, graph_groups
            _ = gc.collect()

        return created_groups, group_relationships

    def load_models(self, existing_saves: list[str] | None = None) -> None:
        """
        Loads the model objects that have been previously saved at their respective directories.

        :param existing_saves: A potentially partial list of the model names which have existing saves.
        :type existing_saves: list[str], optional
        """
        if existing_saves is not None:
            for existing_save in existing_saves:
                self.models[existing_save].load_model(SAVEDIRS[existing_save])
            return None

        for model_name in self.model_names:
            self.models[model_name].load_model(SAVEDIRS[model_name])
        return None

    def setup_models(self, missing_saves: list[str] | None = None) -> None:
        """
        Adds the appropriate model objects to all relevant models.

        :param missing_saves: A potentially partial list of the model names which are missing their saves.
        :type missing_saves: list[str], optional
        """
        print("Setting up the model instances")

        from_group: grp.Group
        to_group: grp.Group

        if missing_saves is not None:
            for missing_save in missing_saves:
                _ = self.models[missing_save].add_agents(deepcopy(self.model_agents[missing_save]))
                _ = self.models[missing_save].add_graphs(
                    deepcopy(self.model_graphs[missing_save]),
                    TEST_PARAMETERS["hierarchy_names"],
                    list(TEST_PARAMETERS["shared_hierarchy_rw"].values()),
                )
                _ = self.models[missing_save].add_groups(deepcopy(self.model_groups[missing_save]))

                for edge in self.group_edges:
                    from_group = self.model_groups[missing_save][edge[0]]
                    to_group = self.model_groups[missing_save][edge[1]]
                    self.models[missing_save].add_group_graph_edge(from_group, to_group)

            return None

        for model_name in self.model_names:
            _ = self.models[model_name].add_agents(deepcopy(self.model_agents[model_name]))
            _ = self.models[model_name].add_graphs(
                deepcopy(self.model_graphs[model_name]),
                TEST_PARAMETERS["hierarchy_names"],
                list(TEST_PARAMETERS["shared_hierarchy_rw"].values()),
            )
            _ = self.models[model_name].add_groups(deepcopy(self.model_groups[model_name]))

            for edge in self.group_edges:
                from_group = self.model_groups[model_name][edge[0]]
                to_group = self.model_groups[model_name][edge[1]]
                self.models[model_name].add_group_graph_edge(from_group, to_group)

        return None

    def run_models(self, missing_saves: list[str] | None = None, worker_pool: WorkerPool | None = None) -> None:
        """
        Runs each model in the tester class.

        :param missing_saves: A potentially partial list of the model names which are missing their saves.
        :type missing_saves: list[str], optional
        :param worker_pool: A pool of workers that can share the processing of the model iteration amongst themselves.
        :type worker_pool: :class:`~multiprocessing.pool.Pool`, optional
        """
        print("Beginning model iterations\n\n")
        if missing_saves is not None:
            for missing_save in missing_saves:
                self.models[missing_save].iterate(worker_pool=worker_pool)
                self.models[missing_save].save_model()
            return None

        for model_name in self.model_names:
            self.models[model_name].iterate(worker_pool=worker_pool)
            self.models[model_name].save_model()

        return None


if __name__ == "__main__":
    MULTIPROCESSED: bool = True
    WORKER_POOL: WorkerPool | None = Pool() if MULTIPROCESSED else None

    class TestParameters(TypedDict):
        iterations: int
        model_names: list[str]
        shared_hierarchy_rw: dict[str, tuple[float, float]]
        unique_hierarchy_rw: dict[str, tuple[float, float]]
        shared_relationship_rw: tuple[float, float]
        unique_rel_rw_range: tuple[tuple[float, float], tuple[float, float]]
        hierarchy_names: list[str]
        graph_generation_alg: str

    # The relevant parameters that are being applied in this experiment
    TEST_PARAMETERS: TestParameters = {
        "iterations": 100,
        "model_names": ["BASE", "RELS", "HIER", "BOTH"],
        "shared_hierarchy_rw": {
            "A": (0.0, 1.0),
            "B": (0.0, 1.0),
            "C": (0.0, 1.0),
            "D": (0.0, 1.0),
            "E": (0.0, 1.0),
            "F": (0.0, 1.0),
        },
        "unique_hierarchy_rw": {
            "A": (0.0, 1.5),
            "B": (0.0, 0.3),
            "C": (0.0, 0.1),
            "D": (-0.1, 0.2),
            "E": (0.0, 0.01),
            "F": (0.1, 0.2),
        },
        "shared_relationship_rw": (0.0, 0.1),
        "unique_rel_rw_range": (
            (-0.1, 0.1),  # Mean range
            (0.0, 0.5),  # Variance range
        ),
        "hierarchy_names": ["A", "B", "C", "D", "E", "F"],
        "graph_generation_alg": "small-world",
    }

    class AgentParameters(TypedDict):
        n_agents: int
        opinions: tuple[float, float]
        relationships: tuple[float, float]
        social_susceptibility: tuple[float, float]
        hierarchy_weighting: tuple[float, float]
        personality: dict[str, float]
        personal_benefit: dict[bool, float]
        id_base: str

    # The parameters that will be used to create the shared population of agents across models
    AGENT_PARAMETERS: AgentParameters = {
        "n_agents": 100,
        "opinions": (-0.8, 0.8),
        "relationships": (-0.8, 0.8),
        "social_susceptibility": (0.2, 0.8),
        "hierarchy_weighting": (-0.6, 0.6),
        "personality": {
            "social": 0.4,
            "neutral": 0.4,
            "impulsive": 0.2,
        },
        "personal_benefit": {True: 0.25, False: 0.75},
        "id_base": "GEXRW",
    }

    class GroupParameters(TypedDict):
        n_groups: int

    # The parameters that will be used to create the shared population of groups across models
    GROUP_PARAMETERS: GroupParameters = {
        "n_groups": 20,
    }

    ROOT_DIR: str = "./experiments/GroupBase/RandomWalk"

    # The save directories for each model instance
    SAVEDIRS: dict[str, str] = {
        "BASE": f"{ROOT_DIR}/RandomWalk_BASE",
        "RELS": f"{ROOT_DIR}/RandomWalk_RELS",
        "HIER": f"{ROOT_DIR}/RandomWalk_HIER",
        "BOTH": f"{ROOT_DIR}/RandomWalk_BOTH",
    }

    # The save paths for each model's logger output (must point to a .csv)
    SAVEFILES: dict[str, str] = {
        "BASE": f"{ROOT_DIR}/BASE_model_variables.csv",
        "RELS": f"{ROOT_DIR}/RELS_model_variables.csv",
        "HIER": f"{ROOT_DIR}/HIER_model_variables.csv",
        "BOTH": f"{ROOT_DIR}/BOTH_model_variables.csv",
    }

    tester: RandomWalkTester

    # Check for existing saved models and store the relevant information
    save_dirs: list[str] = list(os.walk(ROOT_DIR))[0][1]

    directory_missing: bool = False
    existing_savedirs: list[str] = []
    missing_savedirs: list[str] = []

    for model, save_dir in SAVEDIRS.items():
        dir_name: str = deepcopy(save_dir).split("/")[-1]
        if dir_name in save_dirs:
            existing_savedirs.append(model)
        else:
            directory_missing = True
            missing_savedirs.append(model)

    if directory_missing:
        tester = RandomWalkTester()

        # At least one model exists
        if len(existing_savedirs) > 0:
            tester.load_models(existing_saves=existing_savedirs)
            tester.setup_models(missing_saves=missing_savedirs)
            tester.run_models(missing_saves=missing_savedirs, worker_pool=WORKER_POOL)
        else:
            tester.setup_models()
            tester.run_models(worker_pool=WORKER_POOL)
    else:
        tester = RandomWalkTester(existing=True)
        tester.load_models()

    # Ensure that the multiprocessing pool is terminated once all processing is finished
    if WORKER_POOL is not None:
        WORKER_POOL.terminate()
