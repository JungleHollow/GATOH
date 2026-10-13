from __future__ import annotations

import concurrent.futures
import gc
import os
import pickle
from re import L
import warnings
from copy import deepcopy
from dataclasses import dataclass
from itertools import repeat
from multiprocessing import Pool, set_start_method
# For typechecking
from multiprocessing.pool import Pool as WorkerPool
from typing import TypedDict, NotRequired

import polars as pl
import numpy as np

from gatoh.agents import Agent
from gatoh.graphs import Graph, GraphEdge, GraphNode, GroupGraph, GroupEdge, GroupNode
from gatoh.groups import Group
from gatoh.model import ABModel
from gatoh.utils import random_coinflip


class ModelParamsInput(TypedDict):
    """
    A helper class that is used to type check the input dictionary for ModelParameters.
    """
    model_id: str
    hierarchies: list[str]
    hierarchy_rw: dict[str, tuple[float, float]]
    iterations: int
    agent_opinion_rw: NotRequired[tuple[float, float]]
    silencing_threshold: NotRequired[float]
    negation_threshold: NotRequired[float]
    radicalisation_threshold: NotRequired[float]
    visualisation_dir: NotRequired[str]
    suppress_warnings: NotRequired[bool]
    save_dir: NotRequired[str]
    data_file: NotRequired[str]


@dataclass
class ModelParameters:
    """
    Dataclass that is used to store the model initialisation parameters for each model that is being
    run in the experiment.
    """

    model_id: str
    max_iterations: int
    hierarchy_names: list[str]
    hierarchy_rw_distributions: dict[str, tuple[float, float]]
    agent_opinion_rw: tuple[float, float]
    silencing_threshold: float
    negation_threshold: float
    radicalisation_threshold: float
    visualisation_dir: str
    suppress_warnings: bool
    save_dir: str
    data_file: str

    def __init__(self, parameters_dict: ModelParamsInput) -> None:
        self.model_id = parameters_dict["model_id"]
        self.hierarchy_names = parameters_dict["hierarchies"]
        self.hierarchy_rw_distributions = parameters_dict["hierarchy_rw"]
        self.max_iterations = parameters_dict["iterations"]

        agent_opinion_rw: tuple[float, float] | None = parameters_dict.get("agent_opinion_rw")
        if agent_opinion_rw is not None:
            self.agent_opinion_rw = agent_opinion_rw
        else:
            self.agent_opinion_rw = TEST_PARAMETERS["DEFAULT"]["agent_opinion_rw"]
        del agent_opinion_rw

        silencing_threshold: float | None = parameters_dict.get("silencing_threshold")
        if silencing_threshold is not None:
            self.silencing_threshold = silencing_threshold
        else:
            self.silencing_threshold = TEST_PARAMETERS["DEFAULT"]["silencing_threshold"]
        del silencing_threshold

        negation_threshold: float | None = parameters_dict.get("negation_threshold")
        if negation_threshold is not None:
            self.negation_threshold = negation_threshold
        else:
            self.negation_threshold = TEST_PARAMETERS["DEFAULT"]["negation_threshold"]
        del negation_threshold

        radicalisation_threshold: float | None = parameters_dict.get("radicalisation_threshold")
        if radicalisation_threshold is not None:
            self.radicalisation_threshold = radicalisation_threshold
        else:
            self.radicalisation_threshold = TEST_PARAMETERS["DEFAULT"]["radicalisation_threshold"]
        del radicalisation_threshold

        visualisation_dir: str | None = parameters_dict.get("visualisation_dir")
        if visualisation_dir is not None:
            self.visualisation_dir = visualisation_dir
        else:
            self.visualisation_dir = VISDIRS[self.model_id]
        del visualisation_dir

        suppress_warnings: bool | None = parameters_dict.get("suppress_warnings")
        if suppress_warnings is not None:
            self.suppress_warnings = suppress_warnings
        else:
            self.suppress_warnings = TEST_PARAMETERS["DEFAULT"]["suppress_warnings"]
        del suppress_warnings

        save_dir: str | None = parameters_dict.get("save_dir")
        if save_dir is not None:
            self.save_dir = save_dir
        else:
            self.save_dir = SAVEDIRS[self.model_id]
        del save_dir

        data_file: str | None = parameters_dict.get("data_file")
        if data_file is not None:
            self.data_file = data_file
        else:
            self.data_file = DATAFILES[self.model_id]
        del data_file

        return None


class CaseStudyRunner:
    """
    A class that will be used to create and handle the appropriate GATOH models from the given agent
    and social hierarchy graph subdirectories.

    The expected agent and graph inputs for this class are structured according to the outputs from the
    ResponseParser script.

    :param agent_paths: A <model_name : path> mapping pointing to the subdirectories at which each model's Agent objects are saved.
    :type agent_paths: dict[str, str]
    :param graph_paths: A <model_name : path> mapping pointing to the subdirectories at which each model's Graph objects are saved.
    :type graph_paths: dict[str, str]
    :param group_paths: A <model_name : path> mapping pointing to the subdirectories at which each model's Group objects are saved.
    :type group_paths: dict[str, str]
    :param initial_hierarchies: A list of the social hierarchies that will be present in the initial data passed to the runner.
    :type initial_hierarchies: list[str]
    :param agent_parameters: A <parameter : value> mapping specifying any additional, relevant parameters for agents in this experiment.
    :type agent_parameters: dict[str, float]
    :param group_parameters: A <parameter : value> mapping specifying any additional, relevant parameters for groups in this experiment.
    :type group_parameters: dict[str, int]
    :param test_parameters: A <model : parameters> mapping specifying explicit initialisation and runtime parameters for each model.
    :type test_parameters: dict[str, dict[str, Any]]
    :param opinion_paths: A <model_name : path> mapping pointing to csv files containing dependant variable data (for model validation after running).
    :type opinion_paths: dict[str, str], optional
    :param worker_pool: A pool of workers that can distribute the processing of the object loading and model runtime functions amongst themselves.
    :type worker_pool: :class:`~multiprocessing.pool.Pool`, optional
    :param existing: A flag indicating if the runner is loading an existing experiment.
    :type existing: bool, optional
    """

    def __init__(
        self,
        agent_paths: dict[str, str],
        graph_paths: dict[str, str],
        group_paths: dict[str, str],
        initial_hierarchies: list[str],
        agent_parameters: dict[str, float],
        group_parameters: dict[str, int],
        test_parameters: TestParameters,
        opinion_paths: dict[str, str] | None = None,
        worker_pool: WorkerPool | None = None,
        existing: bool = False,
    ) -> None:
        self.existing: bool = existing

        self.agent_paths: dict[str, str] = agent_paths
        self.graph_paths: dict[str, str] = graph_paths
        self.group_paths: dict[str, str] = group_paths

        self.initial_hierarchies: list[str] = initial_hierarchies

        self.agent_parameters: dict[str, float] = agent_parameters
        self.group_parameters: dict[str, int] = group_parameters

        self.opinion_paths: dict[str, str] | None = opinion_paths
        self.opinion_dfs: dict[str, pl.DataFrame] = {}

        self.agent_objects: dict[str, list[Agent]] = {}
        self.graph_objects: dict[str, list[Graph]] = {}
        self.group_objects: dict[str, list[Group]] = {}

        self.group_edges: dict[str, list[tuple[int, int]]] = {}

        if self.opinion_paths is not None:
            for key, value in self.opinion_paths.items():
                with open(value, "r") as csv_file:
                    self.opinion_dfs[key] = pl.read_csv(csv_file)

        self.model_params: dict[str, ModelParameters] = {}

        if not self.existing:
            for key, value in test_parameters.items():
                if key == "DEFAULT":
                    continue
                else:
                    param_struct: ModelParameters = ModelParameters(value)
                    self.model_params[key] = param_struct

                    # Also initialise the appropriate object lists
                    self.agent_objects[key] = []
                    self.graph_objects[key] = []
                    self.group_objects[key] = []
                    self.group_edges[key] = []

        self.models: dict[str, ABModel] = {}

        self.load_objects(worker_pool=worker_pool)
        return None

    def load_objects(self, worker_pool: WorkerPool | None = None) -> None:
        """
        Loads the Agent, Graph, and Group objects for each model in the experiment.

        :param worker_pool: A pool of workers that can distribute the processing of the object loading amongst themselves.
        :type worker_pool: :class:`~multiprocessing.pool.Pool`, optional
        """
        # First, load the Agent objects
        for model_name, agent_path in self.agent_paths.items():
            agent_pickle_paths: list[str] = list(os.walk(agent_path))[0][2]
            for pickle_path in agent_pickle_paths:
                agent_obj: Agent
                with open(f"{AGENT_PATHS[model_name]}/{pickle_path}", "rb") as pickle_file:
                    agent_obj = pickle.load(pickle_file)

                self.agent_objects.setdefault(model_name, []).append(agent_obj)

        # Next, load the Graph objects
        for model_name, graph_path in self.graph_paths.items():
            graph_names: list[str] = list(os.walk(graph_path))[0][1]

            # Multiprocessing here
            if worker_pool is not None:
                loaded_graphs = worker_pool.starmap(
                    self.load_graphs,
                    [(graph_name, graph_path, self.model_params[model_name].hierarchy_rw_distributions[graph_name], model_name)
                        for graph_name in graph_names
                    ]
                )
                for loaded_graph in loaded_graphs:
                    self.graph_objects.setdefault(model_name, []).append(loaded_graph)

                # Manual garbage collection
                del loaded_graphs
                _ = gc.collect()
            else:
                for graph_name in graph_names:
                    loaded_graph = self.load_graphs(
                        graph_name,
                        graph_path,
                        self.model_params[model_name].hierarchy_rw_distributions[graph_name],
                        model_name,
                    )

                    self.graph_objects.setdefault(model_name, []).append(loaded_graph)

        # Finally, load the Groups
        for model_name, group_path in self.group_paths.items():
            # -1 as one of the pickle files corresponds to the group relationships
            group_num: int = len(list(os.walk(group_path))[0][2]) - 1

            for i in range(group_num):
                group_pickle_path: str = f"{group_path}/group_G{model_name}{i + 1:03}.pkl"
                group_obj: Group
                with open(group_pickle_path, "rb") as pickle_file:
                    group_obj = pickle.load(pickle_file)

                self.group_objects[model_name].append(deepcopy(group_obj))

                # Manual garbage collection
                del group_pickle_path, group_obj
                _ = gc.collect()

            # Also load the group edges
            with open(f"{group_path}/group_edges.pkl", "rb") as pickle_file:
                self.group_edges[model_name] = pickle.load(pickle_file)

        return None

    def load_graphs(self, graph_name: str, subdirectory_path: str, rw_params: tuple[float, float], community: str) -> Graph:
        """
        A helper function that allows for parallel processing of Graph loading for :meth:`~CaseStudyRunner.load_objects`.

        :param graph_name: The name of the directory to which all of a graph's files have been saved to.
        :type graph_name: str
        :param subdirectory_path: The path to the subdirectory in which the graph's save directory is located.
        :type subdirectory_path: str
        :param rw_params: The (mean, variance) random-walk parameters for the graph that is being loaded.
        :type rw_params: tuple[float, float]
        :param community: The name of the community for which the graph is being loaded.
        :type community: str
        :return: A loded graph with all included nodes and edges.
        :rtype: Graph
        """
        graphml_path: str = f"{subdirectory_path}/{graph_name}/graph_{graph_name}.graphml"

        new_graph: Graph = Graph("", (0.0, 0.0))
        new_graph.load_graph(graphml_path, graph_name, rw_params)

        node_dir: str = f"{subdirectory_path}/{graph_name}/nodes"
        edge_dir: str = f"{subdirectory_path}/{graph_name}/edges"

        node_files: list[str] = list(os.walk(node_dir))[0][2]
        edge_files: list[str] = list(os.walk(edge_dir))[0][2]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            loaded_nodes = {executor.submit(self.load_node, node_dir, node_file): node_file for node_file in node_files}
            for future in concurrent.futures.as_completed(loaded_nodes):
                node_file = loaded_nodes[future]
                try:
                    node_info = future.result()
                except Exception as exc:
                    print(f"Failed to extract the pickled node at file {node_file} for graph {graph_name} in community {community} with exception: {exc}")
                else:
                    new_graph.graph[node_info[1]] = node_info[0]

            loaded_edges = {executor.submit(self.load_edge, edge_dir, edge_file) : edge_file for edge_file in edge_files}
            for future in concurrent.futures.as_completed(loaded_edges):
                edge_file = loaded_edges[future]
                try:
                    edge_info = future.result()
                except Exception as exc:
                    print(f"Failed to extract the pickled edge at file {edge_file} for graph {graph_name} in community {community} with exception: {exc}")
                else:
                    new_graph.graph.update_edge_by_index(edge_info[1], edge_info[0])
        return new_graph

    def load_node(self, node_dir: str, node_file: str) -> tuple[GraphNode, int]:
        """
        A helper function that allows for multithreading in :meth:`~CaseStudyRunner.load_graphs`.

        :param node_dir: The root directory where all graph nodes have been saved.
        :type node_dir: str
        :param node_file: The name of the pickle file that is being loaded.
        :type node_file: str
        :return: The unpickled graph node object and its corresponding index in the graph.
        :rtype: tuple[GraphNode, int]
        """
        node_index: int = int((os.path.basename(node_file).split("_")[-1]).split(".")[0])
        with open(f"{node_dir}/{node_file}", "rb") as node_pickle:
            node_object: GraphNode = pickle.load(node_pickle)
        return (node_object, node_index)

    def load_edge(self, edge_dir: str, edge_file: str) -> tuple[GraphEdge, int]:
        """
        A helper funtion that allows for multithreading in :meth:`~CaseStudyRunner.load_graphs`.

        :param edge_dir: The root directory where all graph edges have been saved.
        :type edge_dir: str
        :param edge_file: The name of the pickle file that is being loaded.
        :type edge_file: str
        :return: The unpickled graph edge object and its corresponding index in the graph.
        :rtype: tuple[GraphEdge, int]
        """
        edge_index: int = int((os.path.basename(edge_file).split("_")[-1]).split(".")[0])
        with open(f"{edge_dir}/{edge_file}", "rb") as edge_pickle:
            edge_object: GraphEdge = pickle.load(edge_pickle)
        return (edge_object, edge_index)

    def load_models(self, existing_saves: list[str] | None = None) -> None:
        """
        Loads the model objects that have been previously saved in their respective directories.

        :param existing_saves: The model names of the partial existing models that can be loaded.
        :type existing_saves: list[str], optional
        """
        new_model: ABModel
        if existing_saves is not None:
            for existing_save in existing_saves:
                # Create an empty dummy model
                new_model = ABModel(
                    TEST_PARAMETERS["DEFAULT"]["hierarchies"],
                    list(TEST_PARAMETERS["DEFAULT"]["hierarchy_rw"].values()),
                )
                new_model.load_model(SAVEDIRS[existing_save])

                self.models[existing_save] = new_model
            return None

        for model_name, model_savedir in SAVEDIRS.items():
            new_model = ABModel(
                TEST_PARAMETERS["DEFAULT"]["hierarchies"],
                list(TEST_PARAMETERS["DEFAULT"]["hierarchy_rw"].values()),
            )
            new_model.load_model(model_savedir)

            self.models[model_name] = new_model
        return None

    def create_models(self, missing_saves: list[str] | None = None) -> None:
        """
        Use the loaded Agent, Graph, and Group objects plus the defined ModelParameters to create the appropriate
        model instances for use in this experiment,

        :param missing_saves: The partial model names of non-existing models that should be initialised.
        :type missing_saves: list[str], optional
        """
        for model_name, model_parameters in self.model_params.items():
            # Only models in missing saves need to be initialised if specified
            if missing_saves is not None:
                if model_name not in missing_saves:
                    continue

            # Create the ABModel for this instance
            new_model: ABModel = ABModel(
                model_parameters.hierarchy_names,
                list(model_parameters.hierarchy_rw_distributions.values()),
                agent_opinion_rw=model_parameters.agent_opinion_rw,
                iterations=model_parameters.max_iterations,
                silencing_threshold=model_parameters.silencing_threshold,
                negation_threshold=model_parameters.negation_threshold,
                radicalisation_threshold=model_parameters.radicalisation_threshold,
                visualisation_dir=model_parameters.visualisation_dir,
                suppress_warnings=model_parameters.suppress_warnings,
                save_dir=model_parameters.save_dir,
                data_file=model_parameters.data_file,
                model_id=model_parameters.model_id,
                simulate_groups=True,
            )

            # Add the Agents, Graphs, and Groups to the new model
            _ = new_model.add_agents(self.agent_objects[model_name])
            _ = new_model.add_graphs(
                self.graph_objects[model_name],
                model_parameters.hierarchy_names,
                list(model_parameters.hierarchy_rw_distributions.values()),
            )
            _ = new_model.add_groups(self.group_objects[model_name])

            for edge in self.group_edges[model_name]:
                from_group: Group = self.group_objects[model_name][edge[0]]
                to_group: Group = self.group_objects[model_name][edge[1]]
                new_model.add_group_graph_edge(from_group, to_group)

            # Store the model object
            self.models[model_name] = new_model
        return None
