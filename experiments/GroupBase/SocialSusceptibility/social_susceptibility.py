from __future__ import annotations

import gc
import os
import random as rd
from copy import deepcopy
from typing import TypedDict

from multiprocessing import Pool
# For type checking
from multiprocessing.pool import Pool as WorkerPool

import numpy as np

import gatoh.agents as agt
import gatoh.graphs as gr
import gatoh.groups as grp
import gatoh.model as md


class SocialSusceptibilityTester:
    """
    A test class that sets up models which are identical, and with each agent in the population having the same social susceptibility,
    but the shared susceptibility value differs across models.

    The models will contain shared social susceptibility values ranging from 0.0 to 1.0, at intervals of 0.2.

    Each instance will use identical random walk parameters, graph structures, agent hierarchy membership, agent groups,
    and initial conditions; with only the shared social susceptibility value for agents changing between models.
    """

    def __init__(self, existing: bool = False) -> None:
        """
        :param existing: A flag indicating if the tester is loading existing experiment data.
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
                deepcopy(TEST_PARAMETERS["hierarchy_names"]),
                deepcopy(list(TEST_PARAMETERS["hierarchy_rw"].values())),
                save_dir=SAVEDIRS[model_name],
                data_file=SAVEFILES[model_name],
                model_id=model_name,
                simulate_groups=True,
            )
            self.models[model_name] = new_model

        # Define the dicts that will contain the populations of agents, groups, and graphs
        self.model_agents: dict[str, list[agt.Agent]] = {}
        self.model_graphs: dict[str, list[gr.Graph]] = {}
        self.model_groups: dict[str, list[grp.Group]] = {}

        self.group_edges: list[tuple[int, int]] = []

        if not self.existing:
            self.create_agents()
            self.create_graphs()
            self.create_groups()

    def create_agents(self) -> None:
        """
        Generates and sets the shared population of Agent objects that will be used across the instances.
        """
        print("==== Starting Agent creation ====")
        created_agents: list[agt.Agent] = []

        benefit_flags: list[bool] = list(AGENT_PARAMETERS["personal_benefit"].keys())
        benefit_p: list[float] = list(AGENT_PARAMETERS["personal_benefit"].values())

        for i in range(self.n_agents):
            agent_id: str = f"{AGENT_PARAMETERS['id_base']}{i + 1:04}"
            agent_opinion: float = rd.uniform(AGENT_PARAMETERS["opinions"][0], AGENT_PARAMETERS["opinions"][1])
            agent_personality: str = agt.draw_personality()
            # Treat ZERO as the base case
            agent_behaviour: tuple[str, float] = (agent_personality, 0.0)
            agent_benefit: bool = bool(np.random.choice(benefit_flags, size=1, p=benefit_p)[0])

            hierarchy_weightings: dict[str, float] = {}
            for hierarchy_name in TEST_PARAMETERS["hierarchy_names"]:
                generated_weighting: float = rd.uniform(
                    AGENT_PARAMETERS["hierarchy_weighting"][0],
                    AGENT_PARAMETERS["hierarchy_weighting"][1],
                )
                hierarchy_weightings[hierarchy_name] = generated_weighting

            created_agent: agt.Agent = agt.Agent(
                agent_id,
                agent_opinion,
                hierarchy_weightings,
                agent_behaviour,
                agent_benefit,
            )

            created_agents.append(created_agent)

        # Set the agents for ZERO (base case)
        self.model_agents["ZERO"] = deepcopy(created_agents)

        # Update the social susceptibilities to the new value
        for agent in created_agents:
            agent.social_susceptibility += 0.2

        # Set the agents for the next case
        self.model_agents["POINT-TWO"] = deepcopy(created_agents)

        # Repeat...
        for agent in created_agents:
            agent.social_susceptibility += 0.2
        self.model_agents["POINT-FOUR"] = deepcopy(created_agents)

        for agent in created_agents:
            agent.social_susceptibility += 0.2
        self.model_agents["POINT-SIX"] = deepcopy(created_agents)

        for agent in created_agents:
            agent.social_susceptibility += 0.2
        self.model_agents["POINT-EIGHT"] = deepcopy(created_agents)

        for agent in created_agents:
            agent.social_susceptibility += 0.2
        self.model_agents["ONE"] = deepcopy(created_agents)

        print("==== Finished Agent creation ====")
        return None

    def create_graphs(self) -> None:
        """
        Generates and sets the shared collection of social hierarchy Graph objects that will be used across the instances.
        """
        print("==== Starting Graph creation ====")
        created_graphs: list[gr.Graph] = []

        for hierarchy in TEST_PARAMETERS["hierarchy_names"]:
            graph: gr.Graph = gr.Graph(hierarchy, TEST_PARAMETERS["relationship_rw"])
            _ = graph.generate_graph(
                deepcopy(self.model_agents["ZERO"]),
                method=TEST_PARAMETERS["graph_generation_alg"],
                relationship_range=AGENT_PARAMETERS["relationships"],
            )

            created_graphs.append(graph)

        # Set the graphs for all instances
        for model_name in self.model_names:
            self.model_graphs[model_name] = deepcopy(created_graphs)

        # Update the GraphNodes for all instances after ZERO
        self.update_graph_nodes("POINT-TWO")
        self.update_graph_nodes("POINT-FOUR")
        self.update_graph_nodes("POINT-SIX")
        self.update_graph_nodes("POINT-EIGHT")
        self.update_graph_nodes("ONE")

        print("==== Graph creation finished ====")
        return None

    def create_groups(self) -> None:
        """
        Runs KMeans clustering for each social hierarchy to generate and set the shared population of Group objects that will be used across instances.

        :raises RuntimeError: If a valid Agent object does not exist to update a GraphNode.
        """
        print("==== Starting Group creation ====")
        created_groups: list[grp.Group] = []
        group_relationships: list[tuple[int, int]] = []

        group_count: int = 0

        for graph in self.model_graphs["ZERO"]:
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
                    hierarchy=graph.name,
                    members=members,
                )
                group_count += 1
                graph_groups.append(new_group)

            created_groups.extend(deepcopy(graph_groups))
            group_relationships.extend(graph.generate_group_edges(graph_groups))

            # Manual garbage collection
            del clustered_nodes, group_members, graph_groups
            _ = gc.collect()

        # Set the base groups
        self.model_groups["ZERO"] = deepcopy(created_groups)

        # Save the relationships between groups
        self.group_edges = deepcopy(group_relationships)

        # Update the aggregate social susceptibility for the groups
        for group in created_groups:
            _ = group.change_aggregate_susceptibility(0.2)

        # Set the groups for the following model
        self.model_groups["POINT-TWO"] = deepcopy(created_groups)

        # Repeat...
        for group in created_groups:
            _ = group.change_aggregate_susceptibility(0.2)
        self.model_groups["POINT-FOUR"] = deepcopy(created_groups)

        for group in created_groups:
            _ = group.change_aggregate_susceptibility(0.2)
        self.model_groups["POINT-SIX"] = deepcopy(created_groups)

        for group in created_groups:
            _ = group.change_aggregate_susceptibility(0.2)
        self.model_groups["POINT-EIGHT"] = deepcopy(created_groups)

        for group in created_groups:
            _ = group.change_aggregate_susceptibility(0.2)
        self.model_groups["ONE"] = deepcopy(created_groups)

        print("==== Group creation finished ====")
        return None

    def update_graph_nodes(self, model_name: str) -> None:
        """
        A helper function that updates the GraphNodes of all hierarchy graphs within a model instance
        so that they contain their model's respective Agent objects.

        :param model_name: The name of the model instance for which the graph nodes are being updated.
        :type model_name: str
        """
        for hierarchy_graph in self.model_graphs[model_name]:
            # To store AgentIDs in order of appearance
            nodes_in_order: list[str] = []

            for node in hierarchy_graph.graph.nodes():
                nodes_in_order.append(node.agent.id)

            for idx, agent_id in enumerate(nodes_in_order):
                correct_agent: agt.Agent | None = None

                for agent_obj in self.model_agents[model_name]:
                    if agent_obj.id != agent_id
                        continue
                    else:
                        correct_agent = deepcopy(agent_obj)
                        break

                if correct_agent is None:
                    raise RuntimeError("No corresponding Agent object was found -- unable to update hierarchy GraphNode")

                new_graphnode: gr.GraphNode = gr.GraphNode(deepcopy(correct_agent))
                new_graphnode.set_index(idx)

                hierarchy_graph.graph[idx] = new_graphnode

        return None

    def load_models(self, existing_saves: list[str] | None = None) -> None:
        """
        Loads the model objects that have been previously saved in their respective directories.

        :param existing_saves: A potentially partial list of model names for models which have existing experiment data.
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
        Adds the appropriate model objects to all the models.

        :param missing_saves: A potentially partial list of model names for models which do not have existing data.
        :type missing_saves: list[str], optional
        """
        print("==== Setting up the model instances ====")

        from_group: grp.Group
        to_group: grp.Group

        if missing_saves is not None:
            for missing_save in missing_saves:
                _ = self.models[missing_save].add_agents(deepcopy(self.model_agents[missing_save]))
                _ = self.models[missing_save].add_graphs(
                    deepcopy(self.model_graphs[missing_save]),
                    TEST_PARAMETERS["hierarchy_names"],
                    list(TEST_PARAMETERS["hierarchy_rw"].values()),
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
                list(TEST_PARAMETERS["hierarchy_rw"].values()),
            )
            _ = self.models[model_name].add_groups(deepcopy(self.model_groups[model_name]))

            for edge in self.group_edges:
                from_group = self.model_groups[model_name][edge[0]]
                to_group = self.model_groups[model_name][edge[1]]
                self.models[model_name].add_group_graph_edge(from_group, to_group)

        return None

    def run_models(self, missing_saves: list[str] | None = None, worker_pool: WorkerPool | None = None) -> None:
        """
        Runs each model instance in the tester class.

        :param missing_saves: A potentially partial list of model names for models which do not have existing data.
        :type missing_saves: list[str], optional
        """
        print("==== Beginning model iterations ====\n\n")
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
        hierarchy_names: list[str]
        hierarchy_rw: dict[str, tuple[float, float]]
        relationship_rw: tuple[float, float]
        graph_generation_alg: str

    # The relevant parameters that are being applied to this experiment
    TEST_PARAMETERS: TestParameters = {
        "iterations": 100,
        "model_names": [
            "ZERO",
            "POINT-TWO",
            "POINT-FOUR",
            "POINT-SIX",
            "POINT-EIGHT",
            "ONE",
        ],
        "hierarchy_names": ["A", "B", "C", "D"],
        "hierarchy_rw": {
            "A": (0.0, 0.3),
            "B": (0.0, 0.1),
            "C": (0.0, 0.45),
            "D": (0.0, 0.15),
        },
        "relationship_rw": (0.0, 0.1),
        "graph_generation_alg": "small-world",
    }

    class AgentParameters(TypedDict):
        n_agents: int
        opinions: tuple[float, float]
        relationships: tuple[float, float]
        hierarchy_weighting: tuple[float, float]
        personal_benefit: dict[bool, float]
        id_base: str

    # The parameters that will be used to create the Agent population that is shared across models
    AGENT_PARAMETERS: AgentParameters = {
        "n_agents": 100,
        "opinions": (-0.8, 0.8),
        "relationships": (-0.8, 0.8),
        "hierarchy_weighting": (-0.6, 0.6),
        "personal_benefit": {True: 0.25, False: 0.75},
        "id_base": "GEXSS",
    }

    class GroupParameters(TypedDict):
        n_groups: int

    # The parameters that will be used to create the Group population that is shared across models
    GROUP_PARAMETERS: GroupParameters = {
        "n_groups": 20,
    }

    ROOT_DIR: str = "./experiments/GroupBase/SocialSusceptibility"

    # The save directories for each model instance
    SAVEDIRS: dict[str, str] = {
        "ZERO": f"{ROOT_DIR}/SocialSusceptibility_ZERO",
        "POINT-TWO": f"{ROOT_DIR}/SocialSusceptibility_POINT-TWO",
        "POINT-FOUR": f"{ROOT_DIR}/SocialSusceptibility_POINT-FOUR",
        "POINT-SIX": f"{ROOT_DIR}/SocialSusceptibility_POINT-SIX",
        "POINT-EIGHT": f"{ROOT_DIR}/SocialSusceptibility_POINT-EIGHT",
        "ONE": f"{ROOT_DIR}/SocialSusceptibility_ONE",
    }

    # The save paths for each model's logger output (must point to a .csv)
    SAVEFILES: dict[str, str] = {
        "ZERO": f"{ROOT_DIR}/ZERO_model_variables.csv",
        "POINT-TWO": f"{ROOT_DIR}/POINT-TWO_model_variables.csv",
        "POINT-FOUR": f"{ROOT_DIR}/POINT-FOUR_model_variables.csv",
        "POINT-SIX": f"{ROOT_DIR}/POINT-SIX_model_variables.csv",
        "POINT-EIGHT": f"{ROOT_DIR}/POINT-EIGHT_model_variables.csv",
        "ONE": f"{ROOT_DIR}/ONE_model_variables.csv",
    }

    tester: SocialSusceptibilityTester

    # Check for existing model saves
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
        tester = SocialSusceptibilityTester()

        # At least one model exists
        if len(existing_savedirs) > 0:
            tester.load_models(existing_saves=existing_savedirs)
            tester.setup_models(missing_saves=missing_savedirs)
            tester.run_models(missing_saves=missing_savedirs, worker_pool=WORKER_POOL)
        else:
            tester.setup_models()
            tester.run_models(worker_pool=WORKER_POOL)
    else:
        tester = SocialSusceptibilityTester(existing=True)
        tester.load_models()

    # Ensure that the multiprocessing pool is terminated once all processing has finished
    if WORKER_POOL is not None:
        WORKER_POOL.terminate()
