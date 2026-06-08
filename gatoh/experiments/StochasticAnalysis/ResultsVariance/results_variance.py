from __future__ import annotations

import csv
import os
import pickle
import random as rd
from copy import deepcopy
from typing import Any

import numpy as np

import gatoh.agents.agents as agt
import gatoh.graphs.graphs as gr
import gatoh.model.model as md


class AnalysisResults:
    """
    A container class whose main purpose is to collect the observed results from the different models
    that are run in this experiment.

    This class will also provide all functions related to the aggregation, exploration, and visualisation
    of the results.
    """

    def __init__(self) -> None:
        self.aggregate_opinions: dict[str, list[float]] = {}
        self.radicalised_agents: dict[str, list[int]] = {}
        self.polarisations: dict[str, dict[str, list[float]]] = {}

    def init_model(
        self,
        model_id: str,
        aggregate_opinion: list[float],
        radicalised_agent: list[int],
        polarisation: dict[str, list[float]],
    ) -> None:
        """
        An initialisation function that creates the appropriate entries in the data dicts for the specified model.

        :param model_id: The ID that has been assigned to the model object.
        :param aggregate_opinion: The runtime aggregate opinions that the model logged over its iterations.
        :param radicalised_agent: The total radicalised agents that the model logged over its iterations.
        :param polarisation: The network polarisations that the model logged over its iterations.
        """
        if model_id not in self.aggregate_opinions.keys():
            self.aggregate_opinions[model_id] = deepcopy(aggregate_opinion)
        if model_id not in self.radicalised_agents.keys():
            self.radicalised_agents[model_id] = deepcopy(radicalised_agent)
        if model_id not in self.polarisations.keys():
            self.polarisations[model_id] = deepcopy(polarisation)

    def save_results(self) -> None:
        """
        Saves the results stored within this class into separate .csv files for each model parameter type.
        """
        aggregates_path: str = f"{ROOT_DIR}/aggregate_opinions.csv"
        radicalised_path: str = f"{ROOT_DIR}/radicalised_agents.csv"
        polarisation_path: str = f"{ROOT_DIR}/polarisations.csv"

        aggregates_fieldnames: list[str] = ["model_id"]
        radicalised_fieldnames: list[str] = ["model_id"]
        polarisation_fieldnames: list[str] = ["model_id"]

        for i in range(TEST_PARAMETERS["iterations"]):
            aggregates_fieldnames.append(f"iteration_{i + 1}")
            radicalised_fieldnames.append(f"iteration_{i + 1}")
            for hierarchy in TEST_PARAMETERS["hierarchy_names"]:
                polarisation_fieldnames.append(f"{hierarchy}_iteration_{i + 1}")

        csv_writer: csv.DictWriter
        row_dict: dict[str, str | int | float]

        # First store the aggregate opinions
        with open(aggregates_path, "w", newline="") as csv_file:
            csv_writer = csv.DictWriter(csv_file, aggregates_fieldnames)
            csv_writer.writeheader()

            for model_id, parameter_list in self.aggregate_opinions.items():
                row_dict = {"model_id": model_id}
                for idx, parameter in enumerate(parameter_list):
                    row_dict[f"iteration_{idx + 1}"] = parameter

                csv_writer.writerow(row_dict)

        # Next store the radicalised agents
        with open(radicalised_path, "w", newline="") as csv_file:
            csv_writer = csv.DictWriter(csv_file, radicalised_fieldnames)
            csv_writer.writeheader()

            for model_id, parameter_list in self.radicalised_agents.items():
                row_dict = {"model_id": model_id}
                for idx, parameter in enumerate(parameter_list):
                    row_dict[f"iteration_{idx + 1}"] = parameter

                csv_writer.writerow(row_dict)

        # Finally, store the polarisations
        with open(polarisation_path, "w", newline="") as csv_file:
            csv_writer = csv.DictWriter(csv_file, polarisation_fieldnames)
            csv_writer.writeheader()

            for model_id, parameter_dict in self.polarisations.items():
                row_dict = {"model_id": model_id}
                for hierarchy, parameter_list in parameter_dict.items():
                    for idx, parameter in enumerate(parameter_list):
                        row_dict[f"{hierarchy}_iteration_{idx + 1}"] = parameter

                csv_writer.writerow(row_dict)
        return None

    def calculate_average_opinions(self) -> list[float]:
        """
        Calculates the average of the aggregate opinion across the models.

        :return: A list containing the average value of the aggregate opinion at each iteration.
        """
        average_opinions: list[float] = []

        for i in range(TEST_PARAMETERS["iterations"]):
            iteration_sum: float = 0.0

            for model_values in self.aggregate_opinions.values():
                iteration_sum += model_values[i]

            iteration_average: float = iteration_sum / len(
                list(self.aggregate_opinions.keys())
            )

            average_opinions.append(iteration_average)

        return average_opinions

    def calculate_average_radicalised(self) -> list[float]:
        """
        Calculates the average total number of radicalised agents across the models.

        :return: A list containing the average value of total radicalised agents at each iteration.
        """
        average_radicalised: list[float] = []

        for i in range(TEST_PARAMETERS["iterations"]):
            iteration_sum: float = 0.0

            for model_values in self.radicalised_agents.values():
                iteration_sum += model_values[i]

            iteration_average: float = iteration_sum / len(
                list(self.radicalised_agents.keys())
            )

            average_radicalised.append(iteration_average)

        return average_radicalised

    def calculate_average_polarisation(self) -> dict[str, list[float]]:
        """
        Calculates the average polarisation across models for each hierarchy.

        :return: A <hierarchy, list> mapping containing the average value of polarisation at each iteration per hierarchy.
        """
        average_polarisation: dict[str, list[float]] = {}

        for hierarchy in TEST_PARAMETERS["hierarchy_names"]:
            average_polarisation[hierarchy] = []

        for i in range(TEST_PARAMETERS["iterations"]):
            iteration_sums: dict[str, float] = {
                hierarchy: 0.0 for hierarchy in TEST_PARAMETERS["hierarchy_names"]
            }

            for hierarchy_dict in self.polarisations.values():
                for hierarchy, hierarchy_values in hierarchy_dict.items():
                    iteration_sums[hierarchy] += hierarchy_values[i]

            for hierarchy, iteration_sum in iteration_sums.items():
                average_polarisation[hierarchy].append(
                    iteration_sum / len(list(self.polarisations.keys()))
                )

        return average_polarisation


class VarianceTester:
    """
    The main tester class which will handle the set up, iteration, and persistence of the different models
    that are used in this experiment.

    For this experiment, all models will be identical; the purpose being to inspect the level of variance
    across the output model parameters and emergent behaviour from the stochastic runtime processes.

    A secondary objective being a characterisation of the number of model repetitions that need to be run
    and aggregated before any further reductions to the variance are insignificant.
    """

    def __init__(
        self, results_container: AnalysisResults, existing: bool = False
    ) -> None:
        """
        :param results_container: The AnalysisResults object to which the tester's model results will be stored to.
        :param existing: A flag indicating if the experiment has already been run and saved models are present to inspect.
        """
        self.results: AnalysisResults = results_container
        self.num_agents: int = AGENT_PARAMETERS["n_agents"]

        self.existing: bool = existing
        self.model_saves: dict[str, str] = {}

        # Dynamic model space
        self.models: dict[str, md.ABModel] = {}

        # Define the lists that will contain the populations of Agents and Graphs
        self.model_agents: list[agt.Agent] = []
        self.model_graphs: list[gr.Graph] = []

        if not self.existing:
            self.create_models()
            self.create_agents()
            self.create_graphs(self.model_agents)
        else:
            self.model_saves = SAVEDIRS
            # load_models is called from __main__ as any missing savefiles are checked for there
            self.load_agents()
            self.load_graphs()

    def create_models(self) -> None:
        """
        Creates the empty model objects that will later be set up and run for the experiment.
        """
        for i in range(TEST_PARAMETERS["repetitions"]):
            model_id: str = f"{TEST_PARAMETERS['model_id_base']}-{i + 1:03}"

            model_savedir: str = f"{SAVEDIR_ROOT}/{model_id}"
            if not os.path.exists(model_savedir):
                os.mkdir(model_savedir)

            model_datafile: str = f"{model_savedir}/{model_id}_variables.csv"

            new_model: md.ABModel = md.ABModel(
                deepcopy(TEST_PARAMETERS["hierarchy_names"]),
                deepcopy(list(TEST_PARAMETERS["hierarchy_rw"].values())),
                save_dir=model_savedir,
                data_file=model_datafile,
                model_id=model_id,
            )
            self.models[model_id] = deepcopy(new_model)

            # Manual garbage collection
            del new_model
        return None

    def create_agents(self) -> None:
        """
        Generates and sets the shared population of Agent objects that will be used across the instances.
        """
        print("==== Starting Agent creation ====")
        created_agents: list[agt.Agent] = []

        benefit_flags: list[bool] = list(AGENT_PARAMETERS["personal_benefit"].keys())
        benefit_p: list[float] = list(AGENT_PARAMETERS["personal_benefit"].values())

        for i in range(self.num_agents):
            agent_id: str = f"{AGENT_PARAMETERS['id_base']}{i + 1:04}"
            agent_opinion: float = rd.uniform(
                AGENT_PARAMETERS["opinions"][0], AGENT_PARAMETERS["opinions"][1]
            )
            agent_personality: str = agt.draw_personality()
            agent_susceptibility: float = rd.uniform(
                AGENT_PARAMETERS["social_susceptibility"][0],
                AGENT_PARAMETERS["social_susceptibility"][1],
            )
            agent_behaviour: tuple[str, float] = (
                agent_personality,
                agent_susceptibility,
            )
            agent_benefit: bool = bool(
                np.random.choice(benefit_flags, size=1, p=benefit_p)[0]
            )

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

            created_agents.append(deepcopy(created_agent))

            # Manual garbage collection
            del created_agent

        self.model_agents = deepcopy(created_agents)

        # Manual garbage collection
        del created_agents

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

        for i in range(self.num_agents):
            agent_id: str = f"{AGENT_PARAMETERS['id_base']}{i + 1:04}"
            agent_pickle_path: str = f"{agents_path}/agent_{agent_id}.pkl"
            agent_obj: agt.Agent
            with open(agent_pickle_path, "rb") as pickle_file:
                agent_obj = pickle.load(pickle_file)

            self.model_agents.append(deepcopy(agent_obj))

            # Manual garbage collection
            del agent_id, agent_pickle_path, agent_obj

        return None

    def create_graphs(self, agents: list[agt.Agent]) -> None:
        """
        Generates and sets the shared collection of social hierarchy Graph objects that will be used across the instances.

        :param agents: The population of Agents to use for Graph creation.
        """
        print("==== Starting Graph creation ====")

        # Workaround to allow for np random choice
        agent_indices: list[int] = [i for i in range(len(agents))]

        for hierarchy in TEST_PARAMETERS["hierarchy_names"]:
            graph: gr.Graph = gr.Graph(
                hierarchy, TEST_PARAMETERS["relationship_rw"], suppress_warnings=True
            )

            # Ensures that every Agent in the population belongs to at least one hierarchy
            if hierarchy == "B":
                _ = graph.generate_graph(
                    deepcopy(agents),
                    method=TEST_PARAMETERS["graph_generation_alg"],
                    relationship_range=AGENT_PARAMETERS["relationships"],
                )
            else:
                hierarchy_n_agents: int = rd.randint(
                    AGENT_PARAMETERS["subset_size_range"][0],
                    AGENT_PARAMETERS["subset_size_range"][1],
                )
                selected_agents: list[int] = list(
                    np.random.choice(
                        agent_indices, size=hierarchy_n_agents, replace=False
                    )
                )

                agent_sample: list[agt.Agent] = []
                for index in selected_agents:
                    agent_sample.append(deepcopy(agents[index]))

                _ = graph.generate_graph(
                    deepcopy(agent_sample),
                    method=TEST_PARAMETERS["graph_generation_alg"],
                    relationship_range=AGENT_PARAMETERS["relationships"],
                )

                # Manual garbage collection
                del agent_sample

            self.model_graphs.append(deepcopy(graph))

            # Manual garbage collection
            del graph

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
        return None

    def load_models(self, existing_saves: list[str] | None = None) -> None:
        """
        Loads the model objects that have been previously saved in their respective directories.

        :param existing_saves: An optional partial list of the model names representing the models that can be loaded.
        """
        if existing_saves:
            for existing_save in existing_saves:
                # Create an empty dummy model object
                new_model: md.ABModel = md.ABModel(
                    TEST_PARAMETERS["hierarchy_names"], TEST_PARAMETERS["hierarchy_rw"]
                )
                new_model.load_model(SAVEDIRS[existing_save])

                self.models[existing_save] = deepcopy(new_model)

                self.store_model_results(new_model)

                # Manual garbage collection
                del new_model
            return None

        for model_name, model_savedir in SAVEDIRS.items():
            new_model: md.ABModel = md.ABModel(
                TEST_PARAMETERS["hierarchy_names"],
                TEST_PARAMETERS["hierarchy_rw"],
            )
            new_model.load_model(model_savedir)

            self.models[model_name] = deepcopy(new_model)

            self.store_model_results(new_model)

            # Manual garbage collection
            del new_model
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

            csv_writer: csv.DictWriter = csv.DictWriter(
                csv_file, fieldnames=field_names
            )
            csv_writer.writeheader()

            for model in self.models.values():
                csv_row: dict[str, str] = {
                    "model_name": model.model_id,
                    "model_savedir": model.save_dir,
                }
                csv_writer.writerow(csv_row)

        return None

    def setup_models(self, missing_saves: list[str] | None = None) -> None:
        """
        Creates the Model objects and adds copies of the shared Agent and Graph populations to them.

        :param missing_saves: An optional partial list of the model names representing the models that should be set up.
        """
        print("==== Setting up the model instances ====")
        if missing_saves:
            for missing_save in missing_saves:
                _ = self.models[missing_save].add_agents(deepcopy(self.model_agents))
                _ = self.models[missing_save].add_graphs(
                    deepcopy(self.model_graphs),
                    deepcopy(TEST_PARAMETERS["hierarchy_names"]),
                    deepcopy(TEST_PARAMETERS["hierarchy_rw"]),
                )
            return None

        for model in self.models.values():
            _ = model.add_agents(deepcopy(self.model_agents))
            _ = model.add_graphs(
                deepcopy(self.model_graphs),
                deepcopy(TEST_PARAMETERS["hierarchy_names"]),
                deepcopy(TEST_PARAMETERS["hierarchy_rw"]),
            )
        return None

    def store_model_results(self, model: md.ABModel) -> None:
        """
        A helper class that takes a model which has finished iterating and calls AnalysisResults.init_model() appropriately.

        :param model: The model that is being initialised in the AnalysisResults.
        """
        self.results.init_model(
            model.model_id,
            model.logger.variables.aggregate_opinions,
            model.logger.variables.radicalised_agents,
            model.logger.variables.layer_polarisations,
        )
        return None

    def save_results(self) -> None:
        """
        Stores the parameters stored within the AnalysisResults object into separate .csv files for each of the parameters.
        """
        self.results.save_results()
        return None

    def run_models(self, missing_saves: list[str] | None = None) -> None:
        """
        Runs each model instance in the tester class.

        :param missing_saves: An optional partial list of the model names representing models that should be run.
        """
        if missing_saves:
            for missing_save in missing_saves:
                self.models[missing_save].iterate()
                self.models[missing_save].save_model()
                self.store_model_results(self.models[missing_save])
            return None

        for model in self.models.values():
            model.iterate()
            model.save_model()
            self.store_model_results(model)

        return None


if __name__ == "__main__":
    # The relevant parameters that are defined for the identical model instances
    TEST_PARAMETERS: dict[str, Any] = {
        "iterations": 100,
        "repetitions": 100,
        "hierarchy_names": ["A", "B", "C"],
        "hierarchy_rw": {
            "A": (0.0, 0.3),
            "B": (0.0, 0.1),
            "C": (0.0, 0.45),
        },
        "relationship_rw": (0.0, 0.1),
        "graph_generation_alg": "small-world",
        "model_id_base": "SARV-Model",
    }

    # The parameters that will be used to create the Agent population that is shared across models
    AGENT_PARAMETERS: dict[str, Any] = {
        "n_agents": 100,
        "subset_size_range": (25, 50),
        "opinions": (-1.0, 1.0),
        "relationships": (-1.0, 1.0),
        "hierarchy_weighting": (-1.0, 1.0),
        "personal_benefit": {True: 0.25, False: 0.75},
        "social_susceptibility": (0.0, 1.0),
        "id_base": "SARV",  # (Stochastic Analysis Results Variance)
    }

    # The root directory of the experiment itself
    ROOT_DIR: str = "./gatoh/experiments/StochasticAnalysis/ResultsVariance"

    # The root directory in which each instance's save directory will be located
    # (using a /models subdirectory for this experiment due to large number of instances)
    SAVEDIR_ROOT: str = f"{ROOT_DIR}/models"

    # A path to which a validation file will be written -- outlining the model name and save directory that were generated
    # for each instance during the tester initialisation (to allow for reduced, partial runtimes in the future if some files are missing)
    LOGGED_SAVEDIRS: str = f"{ROOT_DIR}/ResultsVariance_logged_savedirs.csv"

    # A <model_name, path> mapping of all the model instances that were initially created by the tester
    SAVEDIRS: dict[str, str] = {}

    results: AnalysisResults
    tester: VarianceTester

    # Check for existing saved models and store the relevant information
    save_dirs: list[str] = list(os.walk(SAVEDIR_ROOT))[0][1]

    directory_missing: bool = False
    existing_savedirs: list[str] = []
    missing_savedirs: list[str] = []

    # The tester has not yet been run of the validation file was removed
    if not os.path.exists(LOGGED_SAVEDIRS):
        directory_missing = True
    else:
        with open(LOGGED_SAVEDIRS, "r", newline="") as csv_file:
            csv_reader: csv.DictReader = csv.DictReader(csv_file)
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
        results = AnalysisResults()
        tester = VarianceTester(results)

        # At least one model exists
        if len(existing_savedirs) > 0:
            tester.load_models(existing_saves=existing_savedirs)
            tester.setup_models(missing_saves=missing_savedirs)
            tester.run_models(missing_saves=missing_savedirs)
            tester.save_results()
        else:
            tester.setup_models()
            tester.run_models()
            tester.save_results()
    else:
        results = AnalysisResults()
        tester = VarianceTester(results, existing=True)
        tester.load_models()
