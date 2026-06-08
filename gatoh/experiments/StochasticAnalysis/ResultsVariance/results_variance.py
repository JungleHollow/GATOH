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
        """ """
        pass


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
        self.models: list[md.ABModel] = []

        # Define the lists that will contain the populations of Agents and Graphs
        self.model_agents: list[agt.Agent] = []
        self.model_graphs: list[gr.Graph] = []

        if not self.existing:
            self.create_agents()
            self.create_graphs(self.model_agents)
        else:
            self.model_saves = SAVEDIRS
            self.load_agents()
            self.load_graphs()

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
        return None

    def load_graphs(self) -> None:
        """
        Deserialises the tester's initial shared Graph population and loads it into memory.
        """
        return None

    def load_models(self, existing_saves: list[str] | None = None) -> None:
        """
        Loads the model objects that have been previously saved in their respective directories.

        :param existing_saves: An optional partial list of the model names representing the models that can be loaded.
        """
        return None

    def create_savedir_validation(self) -> None:
        """
        Writes a csv file with columns ["model_name", "model_savedir"] containing the relevant information for the models
        of all the instances that have been initialised during model setup.

        This is done in order to allow for checking of missing instance save directories if the tester is being initialised
        from an existing run.
        """
        return None

    def setup_models(self, missing_saves: list[str] | None = None) -> None:
        """
        Creates the Model objects and adds copies of the shared Agent and Graph populations to them.

        :param missing_saves: An optional partial list of the model names representing the models that should be set up.
        """
        return None

    def run_models(self, missing_saves: list[str] | None = None) -> None:
        """
        Runs each model instance in the tester class.

        :param missing_saves: An optional partial list of the model names representing models that should be run.
        """
        return None


if __name__ == "__main__":
    # The relevant parameters that are defined for the identical model instances
    TEST_PARAMETERS: dict[str, Any] = {
        "iterations": 100,
        "hierarchy_names": ["A", "B", "C"],
        "hierarchy_rw": {
            "A": (0.0, 0.3),
            "B": (0.0, 0.1),
            "C": (0.0, 0.45),
        },
        "relationship_rw": (0.0, 0.1),
        "graph_generation_alg": "small-world",
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
        else:
            tester.setup_models()
            tester.run_models()
    else:
        results = AnalysisResults()
        tester = VarianceTester(results, existing=True)
        tester.load_models()
