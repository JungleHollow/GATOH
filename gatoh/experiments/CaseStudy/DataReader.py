from __future__ import annotations

import os
import pickle
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import polars as pl

from gatoh.agents.agents import Agent
from gatoh.graphs.graphs import Graph, GraphEdge, GraphNode
from gatoh.model.model import ABModel


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

    def create_models(self, missing_saves: list[str] | None = None) -> None:
        """
        Use the loaded Agent and Graph objects plus the defined ModelParameters to create the appropriate model
        instances to use in this experiment.

        :param missing_saves: An optional partial list of model names representing models that should be initialised.
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
                suppress_warnings=model_parameters.suppress_warnings,
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

    def run_models(self, missing_saves: list[str] | None = None) -> None:
        """
        Runs each model instance in the experiment.

        :param missing_saves: An optional partial list of the model names representing models that should be run.
        """
        print("==== Beginning model iterations ====\n\n")
        if missing_saves:
            for missing_save in missing_saves:
                model_to_run: ABModel = self.models[missing_save]
                model_to_run.iterate()
            # Only save the models which were missing
            self.save_models(missing_saves=missing_saves)
            return None

        for model in self.models.values():
            model.iterate()
        self.save_models()
        return None


if __name__ == "__main__":
    SAVEDIR_ROOT: str = "./gatoh/experiments/CaseStudy/Results"

    SAVEDIRS: dict[str, str] = {
        "NONMN": f"{SAVEDIR_ROOT}/NONMN",
        "MINNG": f"{SAVEDIR_ROOT}/MINNG",
    }

    SAVEFILES: dict[str, str] = {
        "NONMN": f"{SAVEDIRS['NONMN']}/NONMN_model_variables.csv",
        "MINNG": f"{SAVEDIRS['MINNG']}/MINNG_model_variables.csv",
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
        "Age",
        "Gender",
        "Friends",
        "Family",
        "Cultural",
        "Religious",
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

    # Specify the dependant variable CSV paths here:
    OPINION_PATHS: dict[str, str] = {}

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
            data_reader.create_models(missing_saves=missing_savedirs)
            data_reader.run_models(missing_saves=missing_savedirs)
        else:
            data_reader.create_models()
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
