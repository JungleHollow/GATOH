from __future__ import annotations

import os
import pickle
import warnings
from contextlib import closing
from copy import deepcopy
from dataclasses import dataclass
from itertools import repeat
from multiprocessing import Pool
from typing import Any

import polars as pl
from rustworkx import NodeIndices

from gatoh.agents.agents import Agent
from gatoh.graphs.graphs import Graph, GraphEdge, GraphNode
from gatoh.model.model import ABModel

# Declare all relevant global variables here
DEBUG: bool = True
MULTIPROCESSING: bool = True

SAVEDIR_ROOT: str = "./gatoh/experiments/CaseStudy/Results"

VISUALISATION_ROOT: str = f"{SAVEDIR_ROOT}/Visualisations"

SAVEDIRS: dict[str, str] = {
    "NONMN": f"{SAVEDIR_ROOT}/NONMN",
    "MINNG": f"{SAVEDIR_ROOT}/MINNG",
}

SAVEFILES: dict[str, str] = {
    "NONMN": f"{SAVEDIRS['NONMN']}/NONMN_model_variables.csv",
    "MINNG": f"{SAVEDIRS['MINNG']}/MINNG_model_variables.csv",
}

VISDIRS: dict[str, str] = {
    "NONMN": f"{VISUALISATION_ROOT}/NONMN",
    "MINNG": f"{VISUALISATION_ROOT}/MINNG",
}

AGENT_PATHS: dict[str, str] = {
    "NONMN": "./gatoh/experiments/CaseStudy/Agents/NONMN_Agents",
    "MINNG": "./gatoh/experiments/CaseStudy/Agents/MINNG_Agents",
}

GRAPH_PATHS: dict[str, str] = {
    "NONMN": "./gatoh/experiments/CaseStudy/Graphs/NONMN_Graphs",
    "MINNG": "./gatoh/experiments/CaseStudy/Graphs/MINNG_Graphs",
}

BASE_HIERARCHIES: list[str] = [
    # Removing Age and Gender as graphs for now, as these are much too densely connected for reasonable runtimes
    # "Age",
    # "Gender",
    "Friends",
    "Family",
    "Cultural",
    "Religious",
    "Geographical",
    "Social",
]

HIERARCHY_RW: dict[str, tuple[float, float]] = {
    # "Age": (0.0, 0.04),
    # "Gender": (0.0, 0.02),
    "Friends": (0.0, 0.05),
    "Family": (0.0, 0.01),
    "Religious": (0.0, 0.1),
    "Cultural": (0.0, 0.15),
    "Geographical": (0.0, 0.0),
    "Social": (0.0, 0.1),
}

# The relevant parameters that are defined for the model instances
TEST_PARAMETERS: dict[str, dict[str, Any]] = {
    "NONMN": {
        "model_id": "NONMN",
        "iterations": 100,
        "hierarchies": deepcopy(BASE_HIERARCHIES),
        "hierarchy_rw": deepcopy(HIERARCHY_RW),
    },
    "MINNG": {
        "model_id": "MINNG",
        "iterations": 100,
        "hierarchies": deepcopy(BASE_HIERARCHIES),
        "hierarchy_rw": deepcopy(HIERARCHY_RW),
    },
    "DEFAULT": {
        "iterations": 100,
        "hierarchies": deepcopy(BASE_HIERARCHIES),
        "hierarchy_rw": deepcopy(HIERARCHY_RW),
        "agent_opinion_rw": (0.0, 0.05),
        "silencing_threshold": 0.95,
        "negation_threshold": 0.999,
        "radicalisation_threshold": 0.99,
        "suppress_warnings": True,
    },
}

# Used to assign changeable weightings to different agent attributes throughout the models
AGENT_PARAMETERS: dict[str, float] = {
    "age_weighting": 1.1,
    "gender_weighting": 1.25,
}

# Specify the dependant variable CSV paths here:
OPINION_PATHS: dict[str, str] = {}


@dataclass
class ModelParameters:
    """
    Dataclass that is used to store the model initialisation parameters for each model that is being run in the
    experiment.
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

    def __init__(self, parameters_dict: dict[str, Any]) -> None:
        self.model_id = parameters_dict["model_id"]
        self.hierarchy_names = deepcopy(parameters_dict["hierarchies"])
        self.hierarchy_rw_distributions = deepcopy(parameters_dict["hierarchy_rw"])
        self.max_iterations = parameters_dict["iterations"]

        if "agent_opinion_rw" in parameters_dict.keys():
            self.agent_opinion_rw = deepcopy(parameters_dict["agent_opinion_rw"])
        else:
            self.agent_opinion_rw = TEST_PARAMETERS["DEFAULT"]["agent_opinion_rw"]

        if "silencing_threshold" in parameters_dict.keys():
            self.silencing_threshold = parameters_dict["silencing_threshold"]
        else:
            self.silencing_threshold = TEST_PARAMETERS["DEFAULT"]["silencing_threshold"]

        if "negation_threshold" in parameters_dict.keys():
            self.negation_threshold = parameters_dict["negation_threshold"]
        else:
            self.negation_threshold = TEST_PARAMETERS["DEFAULT"]["negation_threshold"]

        if "radicalisation_threshold" in parameters_dict.keys():
            self.radicalisation_threshold = parameters_dict["radicalisation_threshold"]
        else:
            self.radicalisation_threshold = TEST_PARAMETERS["DEFAULT"][
                "radicalisation_threshold"
            ]

        if "visualisation_dir" in parameters_dict.keys():
            self.visualisation_dir = parameters_dict["visualisation_dir"]
        else:
            self.visualisation_dir = VISDIRS[self.model_id]

        if "suppress_warnings" in parameters_dict.keys():
            self.suppress_warnings = parameters_dict["suppress_warnings"]
        else:
            self.suppress_warnings = TEST_PARAMETERS["DEFAULT"]["suppress_warnings"]

        if "save_dir" in parameters_dict.keys():
            self.save_dir = parameters_dict["save_dir"]
        else:
            self.save_dir = SAVEDIRS[self.model_id]

        if "data_file" in parameters_dict.keys():
            self.data_file = parameters_dict["data_file"]
        else:
            self.data_file = SAVEFILES[self.model_id]


class DataReader:
    """
    A class that will be used to create and handle an appropriate GATOH model from the given agent and
    social hierarchy graph subdirectories.

    The expected agent and graph inputs for this class are structured according to the outputs from the
    ResponseParser script.
    """

    def __init__(
        self,
        agent_paths: dict[str, str],
        graph_paths: dict[str, str],
        initial_hierarchies: list[str],
        test_parameters: dict[str, dict[str, Any]],
        opinion_paths: dict[str, str] | None = None,
        existing: bool = False,
    ) -> None:
        """
        :param agent_paths: A <model name : path> mapping pointing to the subdirectories at which each model's Agent objects are saved.
        :param graph_paths: A <model name: path> mapping pointing to the subdirectories at which each model's Graph objects are saved.
        :param initial_hierarchies: A list of the social hierarchies that will be present in the initial data passed to the reader.
        :param test_parameters: A <model: parameters> mapping specifying explicit initialisation and runtime parameters for each model.
        :param opinion_paths: An optional <model name: path> mapping pointing to csv files containing dependant variable data (for model validation after running).
        :param existing: A flag indicating if the DataReader is loading an existing experiment.
        """
        self.existing: bool = existing

        self.agent_paths: dict[str, str] = agent_paths
        self.graph_paths: dict[str, str] = graph_paths

        self.initial_hierarchies: list[str] = initial_hierarchies

        self.opinion_paths: dict[str, str] | None = opinion_paths
        self.opinion_dfs: dict[str, pl.DataFrame] = {}

        self.agent_objects: dict[str, list[Agent]] = {}
        self.graph_objects: dict[str, list[Graph]] = {}

        if self.opinion_paths:
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
                    self.model_params[key] = deepcopy(param_struct)

                    # Also initialise the appropriate object lists
                    self.agent_objects[key] = []
                    self.graph_objects[key] = []

        self.models: dict[str, ABModel] = {}

        self.load_objects()

    def load_objects(self) -> None:
        """
        Loads the Agent and Graph objects for each model in the experiment.
        """
        # First, load the Agent objects
        for model_name, agent_path in self.agent_paths.items():
            agent_pickle_paths: list[str] = list(os.walk(agent_path))[0][2]
            for pickle_path in agent_pickle_paths:
                agent_obj: Agent
                with open(
                    f"{AGENT_PATHS[model_name]}/{pickle_path}", "rb"
                ) as pickle_file:
                    agent_obj = pickle.load(pickle_file)

                model_agents = self.agent_objects.setdefault(model_name, [])
                model_agents.append(deepcopy(agent_obj))

                # Manual garbage collection
                del agent_obj

        # Next, load the unpopulated Graph objects
        for model_name, graph_path in self.graph_paths.items():
            graph_names: list[str] = list(os.walk(graph_path))[0][1]
            for graph_name in graph_names:
                graph_subdir: str = f"{graph_path}/{graph_name}"

                new_graph: Graph = Graph("", (0.0, 0.0))
                new_graph.load_graph(
                    f"{graph_subdir}/graph_{graph_name}.graphml",
                    graph_name,
                    rw_params=self.model_params[model_name].hierarchy_rw_distributions[
                        graph_name
                    ],
                )

                nodes_dir: str = f"{graph_subdir}/nodes"
                node_names: list[str] = list(os.walk(nodes_dir))[0][2]
                for node_name in node_names:
                    node_index: int = int(node_name.split("_")[-1].split(".")[0])
                    with open(f"{nodes_dir}/{node_name}", "rb") as pickle_file:
                        node_object: GraphNode = pickle.load(pickle_file)
                        new_graph.graph[node_index] = node_object

                edges_dir: str = f"{graph_subdir}/edges"
                edge_names: list[str] = list(os.walk(edges_dir))[0][2]
                for edge_name in edge_names:
                    edge_index: int = int(edge_name.split("_")[-1].split(".")[0])
                    with open(f"{edges_dir}/{edge_name}", "rb") as pickle_file:
                        edge_object: GraphEdge = pickle.load(pickle_file)
                        new_graph.graph.update_edge_by_index(edge_index, edge_object)

                graph_objects = self.graph_objects.setdefault(model_name, [])
                graph_objects.append(deepcopy(new_graph))

                # Manual garbage collection
                del new_graph, nodes_dir, node_names, edges_dir, edge_names
        return None

    def load_models(self, existing_saves: list[str] | None = None) -> None:
        """
        Loads the model objects that have been previously saved in their respective directories.

        :param existing_saves: An optional partial list of the model names representing the models that can be loaded.
        """
        if existing_saves:
            for existing_save in existing_saves:
                # Create an empty dummy model
                new_model: ABModel = ABModel(
                    TEST_PARAMETERS["DEFAULT"]["hierarchies"],
                    TEST_PARAMETERS["DEFAULT"]["hierarchy_rw"],
                )
                new_model.load_model(SAVEDIRS[existing_save])

                self.models[existing_save] = deepcopy(new_model)

                # Manual garbage collection
                del new_model
            return None

        for model_name, model_savedir in SAVEDIRS.items():
            new_model: ABModel = ABModel(
                TEST_PARAMETERS["DEFAULT"]["hierarchies"],
                TEST_PARAMETERS["DEFAULT"]["hierarchy_rw"],
            )
            new_model.load_model(model_savedir)

            self.models[model_name] = deepcopy(new_model)

            # Manual garbage collection
            del new_model
        return None

    def create_models(
        self, missing_saves: list[str] | None = None, multiprocessing: Any | None = None
    ) -> None:
        """
        Use the loaded Agent and Graph objects plus the defined ModelParameters to create the appropriate model
        instances to use in this experiment.

        :param missing_saves: An optional partial list of model names representing models that should be initialised.
        :param multiprocessing: An optional pool of workers that can share the processing of model iteration functions.
        """
        for model_name, model_parameters in self.model_params.items():
            # Only models in missing saves need to be initialised
            if missing_saves:
                if model_name not in missing_saves:
                    continue

            # Create the ABModel for this instance
            new_model: ABModel = ABModel(
                deepcopy(model_parameters.hierarchy_names),
                deepcopy(list(model_parameters.hierarchy_rw_distributions.values())),
                agent_opinion_rw=model_parameters.agent_opinion_rw,
                iterations=model_parameters.max_iterations,
                silencing_threshold=model_parameters.silencing_threshold,
                negation_threshold=model_parameters.negation_threshold,
                radicalisation_threshold=model_parameters.radicalisation_threshold,
                visualisation_dir=model_parameters.visualisation_dir,
                suppress_warnings=model_parameters.suppress_warnings,
                multiprocessing=multiprocessing,
                save_dir=model_parameters.save_dir,
                data_file=model_parameters.data_file,
                model_id=model_parameters.model_id,
            )

            # Add the Agents and Graphs to the new model
            _ = new_model.add_agents(deepcopy(self.agent_objects[model_name]))
            _ = new_model.add_graphs(
                deepcopy(self.graph_objects[model_name]),
                deepcopy(model_parameters.hierarchy_names),
                deepcopy(list(model_parameters.hierarchy_rw_distributions.values())),
            )

            # Store the model object
            self.models[model_name] = deepcopy(new_model)

            # Manual garbage collection
            del new_model
        return None

    def save_models(self, missing_saves: list[str] | None = None) -> None:
        """
        Saves the model objects to allow for future loading.

        :param missing_saves: An optional partial list of model names representing the models that should be saved.
        """
        if missing_saves is None:
            for model in self.models.values():
                # Save the model to a newly created savedir
                model.save_model()

                # Call the logger's save_data function which handles data persistence appropriately after the model is saved
                data_saved = model.logger.save_data(model.data_file)

                if data_saved:
                    print(
                        f"\n\nGATOH logger data was successfully written to the file at path: {model.data_file}\n\n"
                    )
        else:
            for missing_save in missing_saves:
                model_to_save: ABModel = self.models[missing_save]

                model_to_save.save_model()

                data_saved = model_to_save.logger.save_data(model_to_save.data_file)

                if data_saved:
                    print(
                        f"\n\nGATOH logger data was successfully written to the file at path: {model_to_save.data_file}\n\n"
                    )
        return None

    def custom_iterate(self, model_to_iterate: ABModel) -> ABModel:
        """
        Custom iteration loop used for this experiment -- accounts for "Age" and "Gender" as Agent attributes.

        :param model_to_iterate: The model that is being run.
        :type model_to_iterate: ABModel
        :return: The model that has been run.
        :rtype: ABModel
        """
        while model_to_iterate.current_iteration < model_to_iterate.max_iterations:
            # Initialise the logger state for the current iteration
            if model_to_iterate.current_iteration == 0:
                model_to_iterate.logger.new_iteration(init=True)
            else:
                model_to_iterate.logger.new_iteration()

            # Initialise a dictionary to keep track of agent opinion changes
            # (this is done to prevent recursive updating of opinions during the evolution of opinions)
            new_agent_opinions: dict[str, tuple[float, list[float], list[bool]]] = {}

            # First each agent looks at its neighbours to see how their opinion will evolve this iteration
            if model_to_iterate.pool is not None:
                opinion_results = model_to_iterate.pool.starmap(
                    self.custom_iter_opinion_calc,
                    zip(model_to_iterate.agents, repeat(model_to_iterate.model_id)),
                )
                for opinion_result in opinion_results:
                    new_agent_opinions[opinion_result[0]] = opinion_result[1]

                # Manual garbage collection
                del opinion_results
            else:
                for agent in model_to_iterate.agents:
                    opinion_result = self.custom_iter_opinion_calc(
                        agent, model_to_iterate.model_id
                    )
                    new_agent_opinions[opinion_result[0]] = opinion_result[1]

                    # Manual garbage collection
                    del opinion_result

            model_to_iterate.iteration_opinion_changes(new_agent_opinions)
            model_to_iterate.step()
            model_to_iterate.update()

            model_to_iterate.logger_iteration()  # Handle the logger's iteration() calculations and call its method

            # Get this iteration's print string (will be formatted appropriately based on the print interval)
            iteration_print_string: str = model_to_iterate.logger.iteration_print()
            print(iteration_print_string)

            if model_to_iterate.visualise:
                model_to_iterate.visualiser.visualiser_iteration()
            if model_to_iterate.checkpointing:
                model_to_iterate.save_model()

            model_to_iterate.current_iteration += 1
        # Call the logger's save_data function which handles data persistence appropriately
        data_saved: bool = model_to_iterate.logger.save_data(model_to_iterate.data_file)
        if data_saved:
            print(
                f"\n\nGATOH logger data was successfully written to the file at path: {model_to_iterate.data_file}\n\n"
            )
        if model_to_iterate.visualise:
            # Make sure that the pyplot figure is closed after iterations to prevent excess memory usage
            import matplotlib.pyplot as plt  # Just using pyplot for this single code block

            plt.close(model_to_iterate.fig)
            del model_to_iterate.fig, model_to_iterate.ax
        return model_to_iterate

    def custom_iter_opinion_calc(
        self,
        agent: Agent,
        model_name: str,
    ) -> tuple[str, tuple[float, list[float], list[bool]]]:
        """
        A helper function that calculates the per-agent, per-hierarchy changes to opinions for the iteration,
        returning all necessary information for :meth:`~gatoh.model.model.ABModel.iteration_opinion_changes`
        to apply the opinion changes.

        This function was primarily created to allow for multiprocessing in the main :meth:`~self.custom_iterate`
        function.

        :param agent: The agent for which the opinion changes are being calculated.
        :type agent: Agent
        :param model_name: The model which is iterating.
        :type model_name: str
        :return: An <Agent ID : Changes info> mapping that provides all necessary information to apply the opinion changes for a specific agent.
        :rtype: tuple[str, tuple[float, list[float], list[bool]]]
        """
        model_to_iterate: ABModel = self.models[model_name]

        agent.previous_opinion = agent.opinion
        for hierarchy in model_to_iterate.graphs:
            if not hierarchy.agent_in_graph(agent):
                continue
            # Update the previous opinion across all hierarchies
            hierarchy.agent_previous_opinion(agent)

        collective_changes: list[float] = []
        for hierarchy in model_to_iterate.graphs:
            # Custom neighbour_influences that accounts for "Age" and "Gender" attributes when determining influences
            neighbour_influences: float | None = self.custom_neighbour_influences(
                agent, hierarchy
            )

            if neighbour_influences is not None:
                collective_changes.append(neighbour_influences)
        total_change: float = sum(collective_changes)

        # Check for the existence of personal benefit across all of the agent's neighbours
        all_neighbour_indices: list[int] = list(
            model_to_iterate.base_graph.graph.neighbors(agent.index)
        )
        all_neighbour_benefits: list[bool] = []
        for neighbour_index in all_neighbour_indices:
            neighbour_object: GraphNode = model_to_iterate.base_graph.graph[
                neighbour_index
            ]
            all_neighbour_benefits.append(neighbour_object.agent.personal_benefit)

        # Define the type of the return
        opinion_result: tuple[str, tuple[float, list[float], list[bool]]]

        # Constrain to [-1, 1]
        # 100.0 and -100.0 are used as key delta values indicating that the opinion needs to be constrained
        if agent.opinion + total_change < -1.0:
            opinion_result = (
                agent.id,
                (
                    -100.0,
                    deepcopy(collective_changes),
                    deepcopy(all_neighbour_benefits),
                ),
            )
        elif agent.opinion + total_change > 1.0:
            opinion_result = (
                agent.id,
                (100.0, deepcopy(collective_changes), deepcopy(all_neighbour_benefits)),
            )
        else:
            opinion_result = (
                agent.id,
                (
                    total_change,
                    deepcopy(collective_changes),
                    deepcopy(all_neighbour_benefits),
                ),
            )
        return opinion_result

    def custom_neighbour_influences(
        self, agent: Agent, hierarchy_graph: Graph
    ) -> float | None:
        """
        A custom version of neighbour_influences() that also accounts for "Age" and "Gender" as universally
        modifying attributes that affect neighbour influences across all hierarchies.

        :param agent: The agent for which the neighbour influences are being determined.
        :type agent: Agent
        :param hierarchy_graph: The hierarchy in which the neighbour influences are being determined.
        :type hierarchy_graph: Graph
        :return: The total value of the neighbour influences on the agent in this specific hierarchy, or None if the agent has no neighbours.
        :rtype: float | None
        """
        agent_hierarchy_weighting: float = agent.social_weightings[hierarchy_graph.name]
        agent_index: int | None = hierarchy_graph.get_agent_index(agent)
        if agent_index is None:
            if not hierarchy_graph.suppress_warnings:
                warnings.warn(
                    f"Input Agent {agent.id} does not exist in this hierarchy ({hierarchy_graph.name})",
                    category=UserWarning,
                )
            return None
        neighbour_indices: NodeIndices = hierarchy_graph.graph.neighbors(agent_index)

        weighted_deltas: list[float] = []
        delta_weightings: list[float] = []
        for neighbour_index in neighbour_indices:
            neighbour_node: GraphNode | None = hierarchy_graph.get_node(neighbour_index)
            if neighbour_node is None:
                # This should never be reached and is only included for type checking purposes
                continue

            relationship_strength: float = hierarchy_graph.get_relationship(
                agent, neighbour_node.agent
            )

            average_opinion: float = (
                agent.opinion + neighbour_node.agent.opinion
            ) / 2.0  # Simple average of own and neighbour opinions
            distance_from_avg: float = (
                average_opinion - agent.opinion
            )  # The delta that must be applied to own opinion to reach the average
            weighted_delta: float = (
                distance_from_avg * agent_hierarchy_weighting * relationship_strength
            )  # The final opinion change

            # Account for neighbour radicalisation
            # (neutral personality means that the existence or lack of neighbour radicalisation will have no effect)
            relative_weighting: float = 1.0
            if (
                neighbour_node.agent.radicalised
                and agent.personality != "neutral"
                and not agent.radicalised
            ):
                if agent.personality in ["rational", "social"]:
                    # "rational" or "social" agents that are not radicalised will have a generally lesser view of radicalised opinions
                    relative_weighting = 0.5
                elif agent.personality == "impulsive":
                    # "impulsive" agents will always view radicalised opinions more favourably
                    relative_weighting = 2.0
                else:  # Agent personality is "erratic"
                    # "erratic" agents act randomly...
                    from gatoh.utils.utils import random_coinflip

                    erratic_coinflip: bool = random_coinflip("bool")
                    if erratic_coinflip:
                        relative_weighting = 2.0
                    else:
                        relative_weighting = 0.5
            elif neighbour_node.agent.radicalised and agent.radicalised:
                if distance_from_avg <= 0.25:
                    # Both agents are radicalised towards the same opinion
                    relative_weighting = 4.0
                else:
                    # Both agent are radicalised in opposing opinions
                    relative_weighting = 0.25

            # Account for "Age" and "Gender" as further modifiers to the relative weighting
            if neighbour_node.agent.get_attribute("age") == agent.get_attribute("age"):
                relative_weighting *= AGENT_PARAMETERS["age_weighting"]
            else:
                relative_weighting *= 1.0 / AGENT_PARAMETERS["age_weighting"]

            if neighbour_node.agent.get_attribute("gender") == agent.get_attribute(
                "gender"
            ):
                relative_weighting *= AGENT_PARAMETERS["gender_weighting"]
            else:
                relative_weighting *= 1.0 / AGENT_PARAMETERS["gender_weighting"]

            weighted_deltas.append(weighted_delta)
            delta_weightings.append(relative_weighting)
        # Calculate the final change
        final_change: float = 0.0
        total_weightings: float = sum(delta_weightings)
        for idx, weighted_delta in enumerate(weighted_deltas):
            if total_weightings == 0.0:
                final_change += 0.0
            else:
                final_change += weighted_delta * (
                    delta_weightings[idx] / total_weightings
                )
        return final_change

    def run_models(self, missing_saves: list[str] | None = None) -> None:
        """
        Runs each model instance in the experiment.

        :param missing_saves: An optional partial list of the model names representing models that should be run.
        """
        print("==== Beginning model iterations ====\n\n")
        if missing_saves:
            for missing_save in missing_saves:
                model_to_run: ABModel = self.models[missing_save]
                _ = self.custom_iterate(model_to_run)
            # Only save the models which were missing
            self.save_models(missing_saves=missing_saves)
            return None

        for model in self.models.values():
            _ = self.custom_iterate(model)
        self.save_models()
        return None


if __name__ == "__main__":
    if MULTIPROCESSING:
        worker_pool = Pool()
    else:
        worker_pool = None

    if not os.path.exists(SAVEDIR_ROOT):
        os.mkdir(SAVEDIR_ROOT)

    if not os.path.exists(VISUALISATION_ROOT):
        os.mkdir(VISUALISATION_ROOT)

    # Check for existing saved models and store the relevant information
    save_dirs: list[str] = list(os.walk(SAVEDIR_ROOT))[0][1]

    directory_missing: bool = False
    existing_savedirs: list[str] = []
    missing_savedirs: list[str] = []

    for model_name, save_dir in SAVEDIRS.items():
        dir_name: str = os.path.basename(save_dir)
        if dir_name in save_dirs:
            existing_savedirs.append(model_name)
        else:
            directory_missing = True
            missing_savedirs.append(model_name)

    data_reader: DataReader
    if directory_missing:
        # Create the tester normally, setup the models, and begin iterations
        data_reader = DataReader(
            AGENT_PATHS,
            GRAPH_PATHS,
            BASE_HIERARCHIES,
            TEST_PARAMETERS,
            opinion_paths=None,
        )

        if len(existing_savedirs) > 0:  # At least one model exists
            data_reader.load_models(existing_saves=existing_savedirs)
            data_reader.create_models(
                missing_saves=missing_savedirs, multiprocessing=worker_pool
            )
            data_reader.run_models(missing_saves=missing_savedirs)
        else:
            data_reader.create_models(multiprocessing=worker_pool)
            data_reader.run_models()
    else:
        # Create the tester in "existing" mode, and examine the results
        data_reader = DataReader(
            AGENT_PATHS,
            GRAPH_PATHS,
            BASE_HIERARCHIES,
            TEST_PARAMETERS,
            opinion_paths=OPINION_PATHS,
            existing=True,
        )
        data_reader.load_models()

    # TODO: Add the graph visualisation functions here once those features are implemented...
