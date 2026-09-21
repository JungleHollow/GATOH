from __future__ import annotations

import csv
import gc
import os
import pickle
import random as rd
from copy import deepcopy
from typing import Self, TypedDict

from multiprocessing import Pool
# For typechecking
from multiprocessing.pool import Pool as WorkerPool

import matplotlib.pyplot as plt
import numpy as np

import gatoh.agents as agt
import gatoh.graphs as gr
import gatoh.groups as grp
import gatoh.model as md


class OutputDict(TypedDict):
    """
    A helper class used for typechecking output_dict in AnalysisResults.calculate_results_statistics.
    """
    opinion_statistics: tuple[list[float], list[float]]
    rad_agts_statistics: tuple[list[float], list[float]]
    rad_grps_statistics: tuple[list[float], list[float]]
    polarisation_statistics: tuple[dict[str, list[float]], dict[str, list[float]]]


class AnalysisResults:
    """
    A container class whose main purpose is to collect the observed results from the different models
    that are run in this experiment.

    This class will also provide all functions related to the aggregation, exploration, and visualisation
    of the results.

    :param model_id: The ID that has been assigned to the model object.
    :type model_id: str
    :param aggregate_opinion: The runtime aggregate opinions that the model logged over its iterations.
    :type aggregate_opinion: list[float]
    :param radicalised_agent: The total radicalised agents that the model logged over its iterations.
    :type radicalised_agent: list[int]
    :param radicalised_group: The total radicalised groups that the model logged over its iterations.
    :type radicalised_group: list[int]
    :param polarisation: The network polarisations that the model logged over its iterations.
    :type polarisations: dict[str, list[float]]
    """

    def __init__(self) -> None:
        self.aggregate_opinions: dict[str, list[float]] = {}
        self.radicalised_agents: dict[str, list[int]] = {}
        self.radicalised_groups: dict[str, list[int]] = {}
        self.polarisations: dict[str, dict[str, list[float]]] = {}

    def init_model(
        self,
        model_id: str,
        aggregate_opinion: list[float],
        radicalised_agent: list[int],
        radicalised_group: list[int],
        polarisation: dict[str, list[float]],
        ) -> None:
            self.aggregate_opinions.setdefault(model_id, deepcopy(aggregate_opinion))
            self.radicalised_agents.setdefault(model_id, deepcopy(radicalised_agent))
            self.radicalised_groups.setdefault(model_id, deepcopy(radicalised_group))
            self.polarisations.setdefault(model_id, deepcopy(polarisation))
            return None

    def save_results(self) -> None:
        """
        Saves the results stored within this class into separate .csv files for each model parameter type.
        """
        aggregates_path: str = f"{ROOT_DIR}/aggregate_opinions.csv"
        rad_agts_path: str = f"{ROOT_DIR}/radicalised_agents.csv"
        rad_grps_path: str = f"{ROOT_DIR}/radicalised_groups.csv"
        polarisation_path: str = f"{ROOT_DIR}/polarisations.csv"

        aggregates_fieldnames: list[str] = ["model_id"]
        rad_agts_fieldnames: list[str] = ["model_id"]
        rad_grps_fieldnames: list[str] = ["model_id"]
        polarisation_fieldnames: list[str] = ["model_id"]

        for i in range(TEST_PARAMETERS["iterations"]):
            aggregates_fieldnames.append(f"iteration_{i + 1}")
            rad_agts_fieldnames.append(f"iteration_{i + 1}")
            rad_grps_fieldnames.append(f"iteration_{i + 1}")
            for hierarchy in TEST_PARAMETERS["hierarchy_names"]:
                polarisation_fieldnames.append(f"{hierarchy}_iteration_{i + 1}")

        csv_writer: csv.DictWriter[str]
        row_dict: dict[str, str | int | float]

        # First, aggregate opinions
        with open(aggregates_path, "w", newline="") as csv_file:
            csv_writer = csv.DictWriter(csv_file, aggregates_fieldnames)
            csv_writer.writeheader()

            for model_id, parameter_list in self.aggregate_opinions.items():
                row_dict = {"model_id": model_id}
                for idx, parameter in enumerate(parameter_list):
                    row_dict[f"iteration_{idx + 1}"] = parameter

                csv_writer.writerow(row_dict)

        # Next, radicalised agents
        with open(rad_agts_path, "w", newline="") as csv_file:
            csv_writer = csv.DictWriter(csv_file, rad_agts_fieldnames)
            csv_writer.writeheader()

            for model_id, parameter_list in self.radicalised_agents.items():
                row_dict = {"model_id": model_id}
                for idx, parameter in enumerate(parameter_list):
                    row_dict[f"iteration_{idx + 1}"] = parameter

                csv_writer.writerow(row_dict)

        # Next, radicalised groups
        with open(rad_grps_path, "w", newline="") as csv_file:
            csv_writer = csv.DictWriter(csv_file, rad_grps_fieldnames)
            csv_writer.writeheader()

            for model_id, parameter_list in self.radicalised_groups.items():
                row_dict = {"model_id": model_id}
                for idx, parameter in enumerate(parameter_list):
                    row_dict[f"iteration_{idx + 1}"] = parameter

                csv_writer.writerow(row_dict)

        # Finally, the polarisations
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

    def load_results(self) -> Self:
        """
        Loads results that have been saved following the above format.
        """
        aggregates_path: str = f"{ROOT_DIR}/aggregate_opinions.csv"
        rad_agts_path: str = f"{ROOT_DIR}/radicalised_agents.csv"
        rad_grps_path: str = f"{ROOT_DIR}/radicalised_groups.csv"
        polarisation_path: str = f"{ROOT_DIR}/polarisations.csv"

        csv_reader: csv.DictReader[str]
        model_name: str = ""

        # First load the aggregate opinions
        with open(aggregates_path, "r", newline="") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                for idx, value in enumerate(row.values()):
                    if idx == 0:
                        model_name = value
                        self.aggregate_opinions[model_name] = []
                    else:
                        self.aggregate_opinions[model_name].append(float(value))

        # Next load the radicalised agents
        with open(rad_agts_path, "r", newline="") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                for idx, value in enumerate(row.values()):
                    if idx == 0:
                        model_name = value
                        self.radicalised_agents[model_name] = []
                    else:
                        self.radicalised_agents[model_name].append(int(value))

        # Then the radicalised groups
        with open(rad_grps_path, "r", newline="") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                for idx, value in enumerate(row.values()):
                    if idx == 0:
                        model_name = value
                        self.radicalised_groups[model_name] = []
                    else:
                        self.radicalised_groups[model_name].append(int(value))

        # Finally load the polarisations
        with open(polarisation_path, "r", newline="") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                hierarchy_name: str
                for key, value in row.items():
                    if key == "model_id":
                        model_name = value
                        self.polarisations[model_name] = {}
                    else:
                        hierarchy_name = key.split("_")[0]
                        self.polarisations[model_name].setdefault(hierarchy_name, []).append(float(value))
        return self

    def calculate_opinions_statistics(self) -> tuple[list[float], list[float]]:
        """
        Calculates the basic statistics of the aggregate opinion across the models.

        :return: The average and standard deviation of the aggregate opinions across the models.
        :rtype: tuple[list[float], list[float]]
        """
        average_opinions: list[float] = []
        opinions_sd: list[float] = []

        for i in range(TEST_PARAMETERS["iterations"]):
            iteration_values: list[float] = []

            for model_values in self.aggregate_opinions.values():
                iteration_values.append(model_values[i])

            iteration_average: float = np.average(iteration_values)
            iteration_sd: float = float(np.std(iteration_values))

            average_opinions.append(iteration_average)
            opinions_sd.append(iteration_sd)

        return average_opinions, opinions_sd

    def calculate_rad_agt_statistics(self) -> tuple[list[float], list[float]]:
        """
        Calculates the basic statistics of the total number of radicalised agents across the models.

        :return: The average and standard deviation of the total number of radicalised agents across the models.
        :rtype: tuple[list[float], list[float]]
        """
        average_rad_agts: list[float] = []
        rad_agts_sd: list[float] = []

        for i in range(TEST_PARAMETERS["iterations"]):
            iteration_values: list[int] = []

            for model_values in self.radicalised_agents.values():
                iteration_values.append(model_values[i])

            iteration_average: float = np.average(iteration_values)
            iteration_sd: float = float(np.std(iteration_values))

            average_rad_agts.append(iteration_average)
            rad_agts_sd.append(iteration_sd)

        return average_rad_agts, rad_agts_sd

    def calculate_rad_grp_statistics(self) -> tuple[list[float], list[float]]:
        """
        Calculates the basic statistics of the total number of radicalised groups across the models.

        :return: The average and standard deviation of the total number of radicalised groups across the models.
        :rtype: tuple[list[float], list[float]]
        """
        average_rad_grps: list[float] = []
        rad_grps_sd: list[float] = []

        for i in range(TEST_PARAMETERS["iterations"]):
            iteration_values: list[int] = []

            for model_values in self.radicalised_groups.values():
                iteration_values.append(model_values[i])

            iteration_average: float = np.average(iteration_values)
            iteration_sd: float = float(np.std(iteration_values))

            average_rad_grps.append(iteration_average)
            rad_grps_sd.append(iteration_sd)

        return average_rad_grps, rad_grps_sd

    def calculate_polarisation_statistics(self) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        """
        Calculates the basic statistics for the polarisation across models for each hierarchy.

        :return: Two <hierarchy : list> mappings containing the average and standard deviation of polarisation at each iteration per hierarchy.
        :rtype: tuple[dict[str, list[float]], dict[str, list[float]]]
        """
        average_polarisation: dict[str, list[float]] = {}
        polarisation_sd: dict[str, list[float]] = {}

        for hierarchy in TEST_PARAMETERS["hierarchy_names"]:
            average_polarisation[hierarchy] = []
            polarisation_sd[hierarchy] = []

        for i in range(TEST_PARAMETERS["iterations"]):
            iteration_values: dict[str, list[float]] = {hierarchy: [] for hierarchy in TEST_PARAMETERS["hierarchy_names"]}

            for hierarchy_dict in self.polarisations.values():
                for hierarchy, hierarchy_values in hierarchy_dict.items():
                    iteration_values[hierarchy].append(hierarchy_values[i])

            for hierarchy, values_list in iteration_values.items():
                hierarchy_average: float = np.average(values_list)
                hierarchy_sd: float = float(np.std(values_list))

                average_polarisation[hierarchy].append(hierarchy_average)
                polarisation_sd[hierarchy].append(hierarchy_sd)

        return average_polarisation, polarisation_sd


class VarianceTester:
    """
    The main tester class which will handle the set up, iteration, and persistence of the different models
    that are used in this experiment.

    For this experiment, all models will be identical; the purpose being to inspect the level of variance
    across the output model parameters and emergent behaviour from the stochastic runtime processes.

    A secondary objective being a characterisation of the number of model repetitions that need to be run
    and aggregated before any further reductions to the variance are insignificant.

    This experiment specifically is an extension of the original ResultsVariance, using GATOH's Groups module
    and clustered simulations rather than individual agents; done in order to ensure that the results variance
    within the group simulations are similar to those seen with individual agents.

    :param results_container: The container to which the tester's model results will be stored to.
    :type results_container: AnalysisResults
    :param existing: A flag indicating if the experiment has already been run and saved models are present to inspect.
    :type existing: bool, optional
    """


if __name__ == "__main__":
    MULTIPROCESSING: bool = True
    WORKER_POOL: WorkerPool | None = Pool() if MULTIPROCESSING else None

    class TestParameters(TypedDict):
        """
        A helper class used for typechecking of TEST_PARAMETERS.
        """
        iterations: int
        repetitions: int
        hierarchy_names: list[str]
        hierarchy_rw: dict[str, tuple[float, float]]
        relationship_rw: tuple[float, float]
        graph_generation_alg: str
        model_id_base: str

    # The relevant parameters that are defined for the identical model instances
    TEST_PARAMETERS: TestParameters = {
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
        "model_id_base": "GSARV-Model",
    }

    class AgentParameters(TypedDict):
        """
        A helper class used for typechecking of AGENT_PARAMETERS.
        """
        n_agents: int
        subset_size_range: tuple[int, int]
        opinions: tuple[float, float]
        relationships: tuple[float, float]
        hierarchy_weighting: tuple[float, float]
        personal_benefit: dict[bool, float]
        social_susceptibility: tuple[float, float]
        id_base: str

    # The parameters that will be used to create the Agent population that is shared across models
    AGENT_PARAMETERS: AgentParameters = {
        "n_agents": 100,
        "subset_size_range": (25, 50),
        "opinions": (-1.0, 1.0),
        "relationships": (-1.0, 1.0),
        "hierarchy_weighting": (-1.0, 1.0),
        "personal_benefit": {True: 0.25, False: 0.75},
        "social_susceptibility": (0.0, 1.0),
        "id_base": "GSARVA",  # (Group Stochastic Analysis Results Variance Agent)
    }

    class GroupParameters(TypedDict):
        """
        A helper class used for typechecking of GROUP_PARAMETERS.
        """
        n_groups: int
        id_base: str

    # The parameters that will be used to create the Group population that is shared across models
    GROUP_PARAMETERS: GroupParameters = {
        "n_groups": 5,
        "id_base": "GSARVG",  # (Group Stochastic Analysis Results Variance Group)
    }

    # The root directory of the experiment itself
    ROOT_DIR: str = "./experiments/StochasticAnalysis/GroupResultsVariance"

    # The root directory in which each instance's save directory will be located
    # (using a /models subdirectory for this experiment due to large number of instances)
    SAVEDIR_ROOT: str = f"{ROOT_DIR}/models"

    # A path to which a validation file will be written -- outlining the model name and save directory that were generated
    # for each instance during the tester initialisation (to allow for reduced, partial runtimes in the future if some files are missing)
    LOGGED_SAVEDIRS: str = f"{ROOT_DIR}/GroupResultsVariance_logged_savedirs.csv"

    # A <model_name : path> mapping of all the model instances that were initially created by the tester
    SAVEDIRS: dict[str, str] = {}

    results: AnalysisResults
    tester: VarianceTester

    # Ensure the subdirectory exists
    if not os.path.exists(SAVEDIR_ROOT):
        os.mkdir(SAVEDIR_ROOT)

    # Check for existing saved models and store the relevant information
    save_dirs: list[str] = list(os.walk(SAVEDIR_ROOT))[0][1]

    directory_missing: bool = False
    existing_savedirs: list[str] = []
    missing_savedirs: list[str] = []

    # The tester has not yet been run or the validation file was removed
    if not os.path.exists(LOGGED_SAVEDIRS):
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
        print("One or more model subdirectories are missing; initialising and running the missing models...")
        results = AnalysisResults()
        tester = VarianceTester(results)

        # At least one model exists
        if len(existing_savedirs) > 0:
            tester.load_models(existing_saves=existing_savedirs)
            tester.setup_models(missing_saves=missing_savedirs)
            tester.run_models(missing_saves=missing_savedirs, worker_pool=WORKER_POOL)
            tester.save_results()
        else:
            tester.setup_models()
            tester.run_models(worker_pool=WORKER_POOL)
            tester.save_results()
    else:
        print("Loading a saved, previously run instance of the Group ResultsVariance experiment...")
        results = AnalysisResults()
        tester = VarianceTester(results, existing=True)
        tester.load_models()
        tester.load_results()

        # Calculate the means and standard deviations for all relevant model parameters
        analysis_statistics: OutputDict = tester.calculate_results_statistics()

        # Create all relevant output plots for this experiment
        create_analysis_plots(tester.results, analysis_statistics)

    # Ensure the worker pool is terminated if it exists once all processing is finished
    if WORKER_POOL is not None:
        WORKER_POOL.terminate()
