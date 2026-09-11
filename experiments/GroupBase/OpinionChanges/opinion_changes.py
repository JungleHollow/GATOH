from __future__ import annotations

import csv
import gc
import os
import pickle
import random as rd
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TypedDict

from multiprocessing import Pool
# For type checking
from multiprocessing.pool import Pool as WorkerPool

import numpy as np

import gatoh.agents as agt
import gatoh.graphs as gr
import gatoh.groups as grp
import gatoh.model as md
from gatoh.utils import random_coinflip


class SaveStructDict(TypedDict):
    model_id: str
    max_iterations: int
    change_iteration: int
    changed_agents: dict[str, list[str]]
    changed_groups: list[str]
    current_iteration: int


@dataclass
class SaveStruct:
    """
    Dataclass that is used to store the supporting information from a ModelStruct alongside an ABModel's already defined
    save files.
    """

    # All attributes correspond directly to those in ModelStruct (except for the missing 'model', which is replaced with model ID)
    model_id: str
    max_iterations: int
    change_iteration: int
    changed_agents: dict[str, list[str]] = field(default_factory=dict)
    changed_groups: list[str] = field(default_factory=list)
    current_iteration: int = 0

    def __init__(self, struct_to_save: ModelStruct, save_dir: str) -> None:
        """
        Extract all the accompanying information from the given ModelStruct, and then pickle the information to the
        ABModel savedir.

        :param struct_to_save: The structure containing all of the supporting experiment information for a model.
        :type struct_to_save: ModelStruct
        :param save_dir: The model's save directory.
        :type save_dir: str
        """
        self.model_id = struct_to_save.model.model_id
        self.max_iterations = struct_to_save.max_iterations
        self.change_iteration = struct_to_save.change_iteration
        self.changed_agents = deepcopy(struct_to_save.changed_agents)
        self.changed_groups = deepcopy(struct_to_save.changed_groups)
        self.current_iteration = struct_to_save.change_iteration

        # Immediately call self.pickle
        self.pickle_struct(save_dir)

    def pickle_struct(self, save_dir: str) -> None:
        """
        Serialise the accompanying information from the ModelStruct and store it in a pickle file within the
        main ABModel's save directory.

        :param save_dir: The model's save directory.
        :type save_dir: str
        """
        with open(f"{save_dir}/{self.model_id}.pkl", "wb") as pickle_file:
            pickle.dump(self.__dict__, pickle_file)
        return None


@dataclass
class ModelStruct:
    """
    Dataclass that defines a structure to contain all relevant information for handling instance runtimes in this experiment.
    """

    # The ABModel object that has been created for this instance
    model: md.ABModel
    # The maximum number of iterations that the instance will run for
    max_iterations: int
    # The iteration at which the opinion changes will be introduced
    change_iteration: int
    # An <ID : hierarchy name> mapping outlining the Agents whose opinions will be changed, and the hierarchies that this occurs in
    changed_agents: dict[str, list[str]] = field(default_factory=dict)
    # The Groups whose opinions will be changed due to the agent opinion changes
    changed_groups: list[str] = field(default_factory=list)
    # The current iteration that the instance is at
    current_iteration: int = 0

    def __init__(
        self,
        model: md.ABModel,
        max_iterations: int,
        change_iteration: int,
        changed_agents: dict[str, list[str]],
        changed_groups: list[str],
    ) -> None:
        """
        Store the instance model and the relevant iteration information to be able to introduce the opinion changes during runtime.

        :param model: The model object that has been created for this instance.
        :type model: ABModel
        :param max_iterations: The total number of iterations the model will run for.
        :type max_iterations: int
        :param change_iteration: The iteration during which the opinion changes are introduced.
        :type change_iteration: int
        :param changed_agents: An <ID : hierarchy name> mapping outlining the Agents whose opinions will be changed, and the hierarchies that this occurs in.
        :type changed_agents: dict[str, list[str]]
        :param changed_groups: The agents' corresponding groups within which opinion changes will occur.
        :type changed_groups: list[str]
        """
        self.model = model
        self.current_iteration = 0
        self.max_iterations = max_iterations
        self.change_iteration = change_iteration
        self.changed_agents = changed_agents
        self.changed_groups = deepcopy(changed_groups)


class OpinionChangesTester:
    """
    A test class that sets up models which are identical in every way, but sudden and strong opinion changes will
    be introduced in random agents at random iterations during their runtime.

    For this experiment, opinion changes will only be introduced once during a model's runtime, using a randomly
    sampled subset of agents, and a mechanism that will generate significantly different opinion values to assign each
    agent, regardless of the initial polarity or magnitude of the opinion.

    Each instance will use identical random walk parameters, graph structures, agent hierarchy membership, agent group membership,
    and initial conditions; with only the iterations and agents at which opinion chagnes are introduced differing between models.
    """

    def __init__(self, existing: bool = False) -> None:
        """
        :param existing: A flag indicating if the experiment is loading past results.
        :type existing: bool, optional
        """
        self.n_agents: int = AGENT_PARAMETERS["n_agents"]
        self.n_groups: int = GROUP_PARAMETERS["n_groups"]
        self.model_intervals: list[int] = list(
            np.linspace(
                0,
                TEST_PARAMETERS["iterations"],
                num=int(
                    TEST_PARAMETERS["iterations"]
                    // TEST_PARAMETERS["opinion_change_interval"]
                ),
                dtype=int,
            )
        )
        self.model_repeats: int = TEST_PARAMETERS["repeats"]

        self.existing: bool = existing
        self.model_saves: dict[str, str] = {}

        # Dynamic model space
        self.models: list[ModelStruct] = []

        # Define the lists that will contain the populations of Agents, Groups, and Graphs
        self.model_agents: list[agt.Agent] = []
        self.model_graphs: list[gr.Graph] = []
        self.model_groups: list[grp.Group] = []

        self.group_edges: list[tuple[int, int]] = []

        if not self.existing:
            self.create_agents()
            self.create_graphs(self.model_agents)
            self.create_groups(self.model_graphs)
        else:
            self.model_saves = SAVEDIRS
            self.load_agents()
            self.load_graphs()
            self.load_groups()

    def get_struct(self, model_name: str) -> ModelStruct:
        """
        A getter function that iterates over self.models, returning the appropriate ModelStruct object.

        :param model_name: The unique model ID for the model struct's model.
        :type model_name: str
        :raises RuntimeError: If no valid struct is found.
        :return: The model struct containing the model with the specified ID.
        :rtype: ModelStruct
        """
        return_struct: ModelStruct | None = None

        for model_struct in self.models:
            if model_struct.model.model_id == model_name:
                return_struct = model_struct
                break

        if return_struct is None:
            raise RuntimeError(f"Model {model_name} was not found in the tester's instances...")
        else:
            return return_struct

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
            agent_susceptibility: float = rd.uniform(
                AGENT_PARAMETERS["social_susceptibility"][0],
                AGENT_PARAMETERS["social_susceptibility"][1],
            )
            agent_behaviour: tuple[str, float] = (agent_personality, agent_susceptibility)
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

        self.model_agents = deepcopy(created_agents)

        # Manual garbage collection
        del created_agents
        _ = gc.collect()

        # Serialise the created Agent objects so that they remain unchanged across future runs
        self.pickle_agents()

        print("==== Finished Agent creation ====")
        return None

    def pickle_agents(self) -> None:
        """
        Serialises the tester's initial shared Agent population to a subdirectory within the experiment directory.
        """
        agents_path: str = f"{ROOT_DIR}/agents"

        if not os.path.exists(agents_path):
            os.mkdir(agents_path)

        for agent in self.model_agents:
            agent_pickle_path: str = f"{agents_path}/agent_{agent.id}.pkl"
            with open(agent_pickle_path, "wb") as pickle_file:
                pickle.dump(agent, pickle_file)

        return None

    def load_agents(self) -> None:
        """
        Deserialises the tester's initial shared Agent population and loads them into memory.
        """
        agents_path: str = f"{ROOT_DIR}/agents"

        for i in range(self.n_agents):
            agent_id: str = f"{AGENT_PARAMETERS['id_base']}{i + 1:04}"
            agent_pickle_path: str = f"{agents_path}/agent_{agent_id}.pkl"
            agent_obj: agt.Agent
            with open(agent_pickle_path, "rb") as pickle_file:
                agent_obj = pickle.load(pickle_file)

            self.model_agents.append(deepcopy(agent_obj))

            # Manual garbage collection
            del agent_id, agent_pickle_path, agent_obj
            _ = gc.collect()

        return None

    def create_graphs(self, agents: list[agt.Agent]) -> None:
        """
        Generates and sets the shared collection of social hierarchy Graph objects that will be used across the instances.

        :param agents: The population of agents to use for graph creation.
        :type agents: list[Agent]
        """
        print("==== Starting Graph creation ====")

        # Workaround to allow for np random choice
        agent_indices: list[int] = [i for i in range(len(agents))]

        for hierarchy in TEST_PARAMETERS["hierarchy_names"]:
            graph: gr.Graph = gr.Graph(
                hierarchy, TEST_PARAMETERS["relationship_rw"], suppress_warnings=True,
            )

            if hierarchy == "A":
                # Ensure that every agent in the population belongs to at least one hierarchy
                _ = graph.generate_graph(
                    deepcopy(agents),
                    method=TEST_PARAMETERS["graph_generation_alg"],
                    relationship_range=AGENT_PARAMETERS["relationships"],
                )
            else:
                hierarchy_n_agents: int = rd.randint(self.n_agents // 10, self.n_agents)
                selected_agents: list[int] = list(np.random.choice(agent_indices, size=hierarchy_n_agents, replace=False))

                agent_sample: list[agt.Agent] = []
                for index in selected_agents:
                    agent_sample.append(deepcopy(agents[index]))

                _ = graph.generate_graph(
                    deepcopy(agent_sample),
                    method=TEST_PARAMETERS["graph_generation_alg"],
                    relationship_range=AGENT_PARAMETERS["relationships"],
                )

                # Manual garbage collection
                del hierarchy_n_agents, selected_agents, agent_sample
                _ = gc.collect()

            self.model_graphs.append(deepcopy(graph))

            # Manual garbage collectioon
            del graph
            _ = gc.collect()

        # Serialise the created Graph objects so that they remain unchanged across future runs
        self.pickle_graphs()

        print("==== Graph creation finished ====")
        return None

    def pickle_graphs(self) -> None:
        """
        Serialises the tester's initial shared Graph population to a subdirectory within the experiment directory.
        """
        graphs_path: str = f"{ROOT_DIR}/graphs"

        if not os.path.exists(graphs_path):
            os.mkdir(graphs_path)

        for graph in self.model_graphs:
            graph_dir: str = f"{graphs_path}/{graph.name}"
            if not os.path.exists(graph_dir):
                os.mkdir(graph_dir)

            # Write the graphml file for the graph
            graph.save_graph(f"{graph_dir}/graph_{graph.name}.graphml")

            nodes_dir: str = f"{graph_dir}/nodes"
            if not os.path.exists(nodes_dir):
                os.mkdir(nodes_dir)

            for idx, node in enumerate(graph.graph.nodes()):
                node_pickle_path: str = f"{nodes_dir}/node_{idx}.pkl"
                with open(node_pickle_path, "wb") as pickle_file:
                    pickle.dump(node, pickle_file)

            edges_dir: str = f"{graph_dir}/edges"
            if not os.path.exists(edges_dir):
                os.mkdir(edges_dir)

            for idx, edge in enumerate(graph.graph.edges()):
                edge_pickle_path: str = f"{edges_dir}/edge_{idx}.pkl"
                with open(edge_pickle_path, "wb") as pickle_file:
                    pickle.dump(edge, pickle_file)

        return None

    def load_graphs(self) -> None:
        """
        Deserialises the tester's initial shared Graph population and loads it into memory.
        """
        graphs_path: str = f"{ROOT_DIR}/graphs"

        for hierarchy in TEST_PARAMETERS["hierarchy_names"]:
            hierarchy_dir: str = f"{graphs_path}/{hierarchy}"

            new_graph: gr.Graph = gr.Graph("", (0.0, 0.0))
            new_graph.load_graph(
                f"{hierarchy_dir}/graph_{hierarchy}.graphml",
                hierarchy,
                rw_params=TEST_PARAMETERS["hierarchy_rw"][hierarchy],
            )

            nodes_dir: str = f"{hierarchy_dir}/nodes"
            node_paths: list[str] = list(os.walk(nodes_dir))[0][2]

            for node_path in node_paths:
                node_index: int = int(
                    (os.path.basename(node_path).split("_")[-1]).split(".")[0]
                )
                with open(f"{nodes_dir}/{node_path}", "rb") as pickle_file:
                    node_object: gr.GraphNode = pickle.load(pickle_file)
                    new_graph.graph[node_index] = node_object

            edges_dir: str = f"{hierarchy_dir}/edges"
            edge_paths: list[str] = list(os.walk(edges_dir))[0][2]

            for edge_path in edge_paths:
                edge_index: int = int(
                    (os.path.basename(edge_path).split("_")[-1]).split(".")[0]
                )
                with open(f"{edges_dir}/{edge_path}", "rb") as pickle_file:
                    edge_object: gr.GraphEdge = pickle.load(pickle_file)
                    new_graph.graph.update_edge_by_index(edge_index, edge_object)

            self.model_graphs.append(deepcopy(new_graph))

            # Manual garbage collection
            del new_graph, hierarchy_dir, nodes_dir, node_paths, edges_dir, edge_paths
            _ = gc.collect()

        return None

    def create_groups(self, graphs: list[gr.Graph]) -> None:
        """
        Generates and sets the shared collection of Group objects that will be used across the instances.

        :param graphs: The population of social hierarchy graphs to cluster and form groups from.
        :type graphs: list[Graph]
        """
        print("==== Starting Group creation ====")
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
                    f"{GROUP_PARAMETERS['id_base']}{group_count + 1:04}",
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

        self.group_edges = deepcopy(group_relationships)
        self.model_groups = deepcopy(created_groups)

        # Manual garbage collection
        del group_relationships, created_groups
        _ = gc.collect()

        print("==== Finished Group creation ====")
        return None

    def pickle_groups(self) -> None:
        """
        Serialises the tester's initial shared Group population to a subdirectory within the experiment directory.
        """
        groups_path: str = f"{ROOT_DIR}/groups"

        if not os.path.exists(groups_path):
            os.mkdir(groups_path)

        for group in self.model_groups:
            group_pickle_path: str = f"{groups_path}/group_{group.id}.pkl"
            with open(group_pickle_path, "wb") as pickle_file:
                pickle.dump(group, pickle_file)

        # Also pickle the group edges
        with open(f"{groups_path}/group_edges.pkl", "wb") as pickle_file:
            pickle.dump(self.group_edges, pickle_file)

        return None

    def load_groups(self) -> None:
        """
        Deserialises the tester's initial shared Group population and loads them into memory.
        """
        groups_path: str = f"{ROOT_DIR}/groups"

        for i in range(self.n_groups):
            group_id: str = f"{GROUP_PARAMETERS['id_base']}{i + 1:04}"
            group_pickle_path: str = f"{groups_path}/group_{group_id}.pkl"
            group_obj: grp.Group
            with open(group_pickle_path, "rb") as pickle_file:
                group_obj = pickle.load(pickle_file)

            self.model_groups.append(deepcopy(group_obj))

            # Manual garbage collection
            del group_id, group_pickle_path, group_obj
            _ = gc.collect()

        # Also load the group edges
        with open(f"{groups_path}/group_edges.pkl", "rb") as pickle_file:
            self.group_edges = pickle.load(pickle_file)

        return None

    def load_models(self, existing_saves: list[str] | None = None) -> None:
        """
        Loads the model objects that have been previously saved in their respective directories.

        :param existing_saves: A potentially partial list of model names representing models that have existing saves.
        :type existing_saves: list[str], optional
        """
        save_struct_path: str
        save_struct_dict: SaveStructDict
        new_model: md.ABModel
        model_struct: ModelStruct
        if existing_saves is not None:
            for existing_save in existing_saves:
                # Create an empty dummy model
                new_model = md.ABModel(
                    TEST_PARAMETERS["hierarchy_names"], list(TEST_PARAMETERS["hierarchy_rw"].values()),
                )
                new_model.load_model(SAVEDIRS[existing_save])

                save_struct_path = f"{SAVEDIRS[existing_save]}/{new_model.model_id}.pkl"
                with open(save_struct_path, "rb") as pickle_file:
                    save_struct_dict = pickle.load(pickle_file)

                model_struct = ModelStruct(
                    deepcopy(new_model),
                    save_struct_dict["max_iterations"],
                    save_struct_dict["change_iteration"],
                    save_struct_dict["changed_agents"],
                    save_struct_dict["changed_groups"],
                )

                self.models.append(deepcopy(model_struct))

                # Manual garbage collection
                del new_model, save_struct_dict, model_struct, save_struct_path
                _ = gc.collect()
            return None

        for model_name, model_savedir in SAVEDIRS.items():
            new_model = md.ABModel(
                TEST_PARAMETERS["hierarchy_names"], list(TEST_PARAMETERS["hierarchy_rw"].values()),
            )
            new_model.load_model(model_savedir)

            save_struct_path = f"{model_savedir}/{model_name}.pkl"
            with open(save_struct_path, "rb") as pickle_file:
                save_struct_dict = pickle.load(pickle_file)

            model_struct = ModelStruct(
                deepcopy(new_model),
                save_struct_dict["max_iterations"],
                save_struct_dict["change_iteration"],
                save_struct_dict["changed_agents"],
                save_struct_dict["changed_groups"],
            )

            self.models.append(deepcopy(model_struct))

            # Manual garbage collection
            del new_model, save_struct_dict, save_struct_path, model_struct
            _ = gc.collect()

        return None

    def create_savedir_validation(self) -> None:
        """
        Writes a csv file with columns ["model_name", "model_savedir"] containing the relevant information for the models
        of all the instances that have been initialised during model setup.

        This is done in order to allow for checking of missing instance save directories if the tester is being initialised
        from an existing run.
        """
        with open(LOGGED_SAVEDIRS, "w", newline="") as csv_file:
            field_names: list[str] = ["model_name", "model_savedir"]

            csv_writer: csv.DictWriter[str] = csv.DictWriter(
                csv_file, fieldnames=field_names,
            )
            csv_writer.writeheader()

            for model_struct in self.models:
                csv_row: dict[str, str] = {
                    "model_name": model_struct.model.model_id,
                    "model_savedir": model_struct.model.save_dir,
                }
                csv_writer.writerow(csv_row)

        return None

    def initialise_model_structs(self, missing_saves: list[str] | None = None) -> None:
        """
        Uses the relevant information provided to create appropriate ABModel objects and wrap them in a ModelStruct.

        :param missing_saves: A potentially partial list of model names representing models that do not have existing saves.
        :type missing_saves: list[str], optional
        """
        # Workaround to allow for np random choice
        group_indices: list[int] = [i for i in range(len(self.model_groups))]

        for interval in self.model_intervals:
            for repeat in range(self.model_repeats):
                # Generate the unique ID for this instance
                model_name: str = f"ITER-{interval:03}_NUM-{repeat + 1:02}"

                # Only models in missing saves need to be initialised
                if missing_saves is not None:
                    if model_name not in missing_saves:
                        continue

                # Create the ABModel for this instance
                new_model: md.ABModel = md.ABModel(
                    TEST_PARAMETERS["hierarchy_names"],
                    list(TEST_PARAMETERS["hierarchy_rw"].values()),
                    suppress_warnings=True,
                    save_dir=f"{SAVEDIR_ROOT}/OpinionChanges_{model_name}",
                    data_file=f"{SAVEDIR_ROOT}/OpinionChanges_{model_name}/{model_name}_model_variables.csv",
                    model_id=model_name,
                    simulate_groups=True,
                )

                # Add the Agents, Graphs, and Groups to the new model
                _ = new_model.add_agents(deepcopy(self.model_agents))
                _ = new_model.add_graphs(
                    deepcopy(self.model_graphs),
                    TEST_PARAMETERS["hierarchy_names"],
                    list(TEST_PARAMETERS["hierarchy_rw"].values()),
                )
                _ = new_model.add_groups(deepcopy(self.model_groups))

                for edge in self.group_edges:
                    from_group: grp.Group = self.model_groups[edge[0]]
                    to_group: grp.Group = self.model_groups[edge[1]]
                    new_model.add_group_graph_edge(from_group, to_group)

                # Sample groups for which the opinion changes will be introduced
                groups_to_change: int = rd.randint(1, len(self.model_groups))
                groups_to_change_idxs: list[int] = list(
                    np.random.choice(group_indices, size=groups_to_change, replace=False)
                )

                group_ids: list[str] = []
                for grp_idx in groups_to_change_idxs:
                    group_ids.append(self.model_groups[grp_idx].id)

                # To keep track of the different agent members that are having their opinions changed
                agent_ids: dict[str, list[str]] = {}

                # Extract the names of all the hierarchies that each agent in a sampled group belongs to
                for group_id in group_ids:
                    group_obj: grp.Group | None = new_model.groups.get_group_by_id(group_id)
                    if group_obj is not None:
                        group_members: list[str] = group_obj.members
                        for group_member in group_members:
                            agent_ids.setdefault(group_member, []).append(group_obj.hierarchy)

                # Create the ModelStruct object
                model_struct: ModelStruct = ModelStruct(
                    deepcopy(new_model),
                    TEST_PARAMETERS["iterations"],
                    interval,
                    deepcopy(agent_ids),
                    deepcopy(group_ids),
                )

                self.models.append(deepcopy(model_struct))

                # Manual garbage collection
                del new_model, groups_to_change, groups_to_change_idxs, group_ids, agent_ids
                _ = gc.collect()
        self.create_savedir_validation()
        return None

    def save_models(self, missing_saves: list[str] | None = None) -> None:
        """
        Saves the model objects along with the information contained in their corresponding ModelStruct to allow for future loading.

        :param missing_saves: A potentially partial list of model names representing models that do not have existing saves.
        :type missing_saves: list[str], optional
        """
        save_struct: SaveStruct
        data_saved: bool
        if missing_saves is None:
            for model_struct in self.models:
                # Will save the model to a newly created savedir
                model_struct.model.save_model()

                # Call the logger's save_data function which handles data persistence appropriately after the model is saved
                data_saved = model_struct.model.logger.save_data(model_struct.model.data_file)

                if data_saved:
                    print(f"\n\nGATOH logger data was successfully written to the file at path: {model_struct.model.data_file}\n\n")

                # Extract the ModelStruct info (without the ABModel) and immediately pickle it to the model's newly created savedir
                save_struct = SaveStruct(model_struct, model_struct.model.save_dir)

                # Manual garbage collection
                del data_saved, save_struct
                _ = gc.collect()
        else:
            for missing_save in missing_saves:
                struct_to_save: ModelStruct = self.get_struct(missing_save)

                struct_to_save.model.save_model()

                data_saved = struct_to_save.model.logger.save_data(struct_to_save.model.data_file)

                if data_saved:
                    print(f"\n\nGATOH logger data was successfully written to the file at path: {struct_to_save.model.data_file}\n\n")

                save_struct = SaveStruct(struct_to_save, struct_to_save.model.save_dir)

                # Manual garbage collection
                del data_saved, save_struct
                _ = gc.collect()
        return None

    def run_models(self, missing_saves: list[str] | None = None, worker_pool: WorkerPool | None = None) -> None:
        """
        Runs each model instance in the tester class, calling the custom iteration function, and introducing the
        sudden opinion changes at the correct iteration.

        :param missing_saves: A potentially partial list of model names representing models that do not have existing saves.
        :type missing_saves: list[str], optional
        :param worker_pool: A pool of workers that can distribute the processing of the iteration amongst themselves.
        :type worker_pool: :class:`~multiprocessing.pool.Pool`, optional
        """
        print("==== Beginning model iterations ====\n\n")
        if missing_saves is not None:
            for missing_save in missing_saves:
                missing_struct: ModelStruct = self.get_struct(missing_save)
                self.custom_iterate(missing_struct, worker_pool=worker_pool)
            self.save_models(missing_saves=missing_saves)
            return None

        for model_struct in self.models:
            self.custom_iterate(model_struct, worker_pool=worker_pool)
        self.save_models()
        return None

    def group_opinion_change(self, initial_opinion: float) -> float:
        """
        A helper function that looks at the direction and magnitude of an initial Group opinion and then
        significantly changes it following a set process.

        :param initial_opinion: The group's initial opinion.
        :type initial_opinion: float
        :return: An opinion value which is significantly different from the initial one.
        :rtype: float
        """
        changed_opinion: float = 0.0

        if initial_opinion < 0.0:
            # A strong negative opinion becomes moderate, and a weak one becomes strongly positive
            changed_opinion = initial_opinion + 1.0
        elif 0.0 < initial_opinion:
            # A strong positive opinion becomes moderate, and a weak one becomes strongly negative
            changed_opinion = -1.0 + initial_opinion
        else:  # opinion == 0.0...
            # A moderate opinion has an equal chance of becoming strongly positive or strongly negative
            negative_coinflip: bool = random_coinflip("bool")
            if negative_coinflip:
                changed_opinion = rd.uniform(-1.0, -0.75)
            else:
                changed_opinion = rd.uniform(0.75, 1.0)

        # The values should always remain in the valid range, but a check is included just in case
        if changed_opinion < -1.0:
            changed_opinion = -1.0
        elif 1.0 < changed_opinion:
            changed_opinion = 1.0

        return changed_opinion

    def custom_iterate(self, model_struct: ModelStruct, worker_pool: WorkerPool | None = None) -> None:
        """
        A custom model iteration function that is able to introduce the opinion changes at the correct iteration
        across instances.

        :param model_struct: A struct containing all relevant information needed to handle the model runtime.
        :type model_struct: ModelStruct
        :param worker_pool: A pool of workers that can distribute the processing of the iteration amongst themselves.
        :type worker_pool: :class:`~multiprocessing.pool.Pool`, optional
        """
        print(f"==== Iterating model {model_struct.model.model_id} ====")

        while model_struct.current_iteration < model_struct.max_iterations:
            if model_struct.current_iteration == 0:
                model_struct.model.logger.new_iteration(init=True)
            else:
                model_struct.model.logger.new_iteration()

            is_change_iteration: bool = model_struct.current_iteration == model_struct.change_iteration

            agent_changes: dict[str, list[float]] = {}

            for group in model_struct.model.groups:
                # Always store the group's previous opinion at the start of an iteration no matter what
                group.store_previous_opinion()

                if is_change_iteration and group.id in model_struct.changed_groups:
                    changed_opinion: float = self.group_opinion_change(group.aggregate_opinion)

                    # Change the group's aggregate opinion
                    per_agent_delta: float = model_struct.model.group_graph.group_graph.group_opinion_change(group, changed_opinion)

                    # Apply note the change for every member agent in this group's hierarchy
                    for member in group.members:
                        agent_changes.setdefault(member, []).append(per_agent_delta)

            if is_change_iteration:
                # Changes have been noted in agent_changes...
                for agent, changes in agent_changes.items():
                    agent_object: agt.Agent = model_struct.model.agents.get_agent_by_id(agent)
                    total_changes: float = sum(changes)
                    agent_object.change_opinion(total_changes)

            # The actual model iteration process (not when the change iteration happens)

            # Track the group opinion changes separately to prevent recursive updates
            new_group_opinions: dict[str, tuple[float, list[bool]]] = {}

            # First, calculate the opinion changes and store them
            if worker_pool is not None and not is_change_iteration:
                opinion_results = worker_pool.imap(
                    model_struct.model.group_iteration_opinion_calculation,
                    model_struct.model.groups,
                    chunksize=10,
                )

                for opinion_result in opinion_results:
                    new_group_opinions[opinion_result[0]] = opinion_result[1]

                # Manual garbage collection
                del opinion_results
                _ = gc.collect()
            elif worker_pool is None and not is_change_iteration:
                for group in model_struct.model.groups:
                    opinion_result = model_struct.model.group_iteration_opinion_calculation(group)
                    new_group_opinions[opinion_result[0]] = opinion_result[1]

                    # Manual garbage collection
                    del opinion_result
                    _ = gc.collect()

            model_struct.model.group_iteration_opinion_changes(new_group_opinions)
            model_struct.model.step()
            model_struct.model.update(worker_pool=worker_pool)
            model_struct.model.logger_iteration(worker_pool=worker_pool)
            iteration_print_string: str = model_struct.model.logger.iteration_print()
            print(iteration_print_string)

            if model_struct.model.visualise:
                model_struct.model.visualiser.visualiser_iteration(
                    model_struct.model.base_graph,
                    model_struct.model.current_iteration,
                    model_name=model_struct.model.model_id,
                )
            if model_struct.model.checkpointing:
                model_struct.model.save_model()

            model_struct.model.current_iteration += 1
            model_struct.current_iteration += 1
        return None


if __name__ == "__main__":
    MULTIPROCESSED: bool = True
    WORKER_POOL: WorkerPool | None = Pool() if MULTIPROCESSED else None

    class TestParameters(TypedDict):
        iterations: int
        opinion_change_interval: int
        repeats: int
        hierarchy_names: list[str]
        hierarchy_rw: dict[str, tuple[float, float]]
        relationship_rw: tuple[float, float]
        graph_generation_alg: str
        use_subsetting: bool

    # The relevant parameters that are defined for the identical model instances
    TEST_PARAMETERS: TestParameters = {
        "iterations": 100,
        "opinion_change_interval": 20,
        "repeats": 5,
        "hierarchy_names": ["A", "B", "C", "D", "E", "F"],
        "hierarchy_rw": {
            "A": (0.0, 0.3),
            "B": (0.0, 0.1),
            "C": (0.0, 0.45),
            "D": (0.0, 0.15),
            "E": (0.0, 0.05),
            "F": (0.0, 0.25),
        },
        "relationship_rw": (0.0, 0.1),
        "graph_generation_alg": "small-world",
        "use_subsetting": True,
    }

    class AgentParameters(TypedDict):
        n_agents: int
        opinions: tuple[float, float]
        relationships: tuple[float, float]
        hierarchy_weighting: tuple[float, float]
        personal_benefit: dict[bool, float]
        social_susceptibility: tuple[float, float]
        id_base: str

    # The parameters that will be used to create the Agent population that is shared across models
    AGENT_PARAMETERS: AgentParameters = {
        "n_agents": 100,
        "opinions": (-0.9, 0.9),
        "relationships": (-0.9, 0.9),
        "hierarchy_weighting": (-0.75, 0.75),
        "personal_benefit": {True: 0.3, False: 0.7},
        "social_susceptibility": (0.0, 1.0),
        "id_base": "GEXOC"  # (Grouped EXperiment Opinion Changes)
    }

    class GroupParameters(TypedDict):
        n_groups: int
        id_base: str

    # The parameters that will be used to create the Group population that is shared across models
    GROUP_PARAMETERS: GroupParameters = {
        "n_groups": 20,
        "id_base": "GEXOC"  # (Grouped EXperiment Opinion Changes)
    }

    # The root directory of the entire experiment
    ROOT_DIR: str = "./experiments/GroupBase/OpinionChanges"

    # The root of the directory in which each instance's save directory will be located
    # (using a /models subdirectory just for this experiment due to significant increase in number of instances)
    SAVEDIR_ROOT: str = f"{ROOT_DIR}/models"

    # A path to which a validation file will be written -- outlining the model name and save directory that were generated
    # for each instance using the tester initialisation (to allow for checking of missing saves in the future)
    LOGGED_SAVEDIRS: str = f"{ROOT_DIR}/OpinionChanges_logged_savedirs.csv"

    # A <model name : path> mapping of all the model instances that were initially created by the tester
    SAVEDIRS: dict[str, str] = {}

    tester: OpinionChangesTester

    # Check for existing saved models and store the relevant information
    save_dirs: list[str] = list(os.walk(SAVEDIR_ROOT))[0][1]

    directory_missing: bool = False
    existing_savedirs: list[str] = []
    missing_savedirs: list[str] = []

    if not os.path.exists(LOGGED_SAVEDIRS):
        # The tester has not yet been run, or the validation file was removed
        directory_missing = True
    else:
        with open(LOGGED_SAVEDIRS, "r", newline="") as csv_file:
            csv_reader: csv.DictReader[str] = csv.DictReader(csv_file)
            for row in csv_reader:
                SAVEDIRS[row["model_name"]] = row["model_savedir"]

        for model, save_dir in SAVEDIRS.items():
            dir_name: str = deepcopy(save_dir).split("/")[-1]
            if dir_name in save_dirs:
                existing_savedirs.append(model)
            else:
                directory_missing = True
                missing_savedirs.append(model)

    if directory_missing:
        tester = OpinionChangesTester()

        if len(existing_savedirs) > 0:
            # At least one model exists
            tester.load_models(existing_saves=existing_savedirs)
            tester.initialise_model_structs(missing_saves=missing_savedirs)
            tester.run_models(missing_saves=missing_savedirs, worker_pool=WORKER_POOL)
        else:
            tester.initialise_model_structs()
            tester.run_models(worker_pool=WORKER_POOL)
    else:
        tester = OpinionChangesTester(existing=True)
        tester.load_models()

    # Ensure that the multiprocessing pool is terminated once all processing is finished
    if WORKER_POOL is not None:
        WORKER_POOL.terminate()
