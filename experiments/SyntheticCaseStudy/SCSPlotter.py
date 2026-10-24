from __future__ import annotations

import csv
import os
from copy import deepcopy
from typing import TypedDict, Self

import matplotlib.pyplot as plt
import numpy as np


class OutputDict(TypedDict):
    """
    A helper class used for typechecking output_dict in AnalysisResults.calculate_results_statistics.
    """
    opinion_statistics: tuple[list[float], list[float]]
    rad_agts_statistics: tuple[list[float], list[float]]
    rad_grps_statistics: tuple[list[float], list[float]]
    polarisation_statistics: tuple[dict[str, list[float]], dict[str, list[float]]]


class GroupedOutputDict(TypedDict):
    """
    A helper class used for typechecking output_dict in AnalysisResults.calculate_grouped_results_statistics.
    """
    opinion_statistics: dict[str, tuple[list[float], list[float]]]
    rad_agts_statistics: dict[str, tuple[list[float], list[float]]]
    rad_grps_statistics: dict[str, tuple[list[float], list[float]]]
    polarisation_statistics: dict[str, tuple[dict[str, list[float]], dict[str, list[float]]]]


class AnalysisResults:
    """
    A container class whose main purpose is to collect the observed results from the different models
    that are run in the synthetic case study experiments.

    This class will also provide all functions related to aggregation, exploration, and visualisation
    of the results.

    :param root_dir: The root directory of the experiment -- where all output .csv and visualisations should be saved.
    :type root_dir: str
    :param max_iterations: The maximum number of iterations that all models ran for.
    :type max_iterations: int
    :param hierarchies: The names of all social hierarchies that were modelled by the experiment models.
    :type hierarchies: list[str]
    :param repetitions: The number of model repetitions that have been run for each community.
    :type repetitions: int
    :param communities: The names of all types of communities being modelled in the experiment.
    :type communities: list[str]
    :param model_id: The ID that has been assigned to the model object.
    :type model_id: str
    :param aggregate_opinion: The runtime aggregate opinions that the model logged over its iterations.
    :type aggregate_opinion: list[float]
    :param radicalised_agent: The total radicalised agents that the model logged over its iterations.
    :type radicalised_agent: list[int]
    :param radicalised_group: The total radicalised groups that the model logged over its iterations.
    :type radicalised_group: list[int]
    :param polarisation: The network polarisations that the model logged over its iterations.
    :type polarisation: dict[str, list[float]]
    """
    def __init__(self, root_dir: str, max_iterations: int, hierarchies: list[str], repetitions: int, communities: list[str]) -> None:
        self.aggregate_opinions: dict[str, list[float]] = {}
        self.radicalised_agents: dict[str, list[int]] = {}
        self.radicalised_groups: dict[str, list[int]] = {}
        self.polarisations: dict[str, dict[str, list[float]]] = {}
        self.analysis_statistics: OutputDict
        self.grouped_analysis_statistics: GroupedOutputDict
        self.root_dir: str = root_dir
        self.max_iterations: int = max_iterations
        self.hierarchies: list[str] = hierarchies
        self.repetitions: int = repetitions
        self.communities: list[str] = communities

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
        aggregates_path: str = f"{self.root_dir}/aggregate_opinions.csv"
        rad_agts_path: str = f"{self.root_dir}/radicalised_agents.csv"
        rad_grps_path: str = f"{self.root_dir}/radicalised_groups.csv"
        polarisation_path: str = f"{self.root_dir}/polarisations.csv"

        aggregates_fieldnames: list[str] = ["model_id"]
        rad_agts_fieldnames: list[str] = ["model_id"]
        rad_grps_fieldnames: list[str] = ["model_id"]
        polarisation_fieldnames: list[str] = ["model_id"]

        for i in range(self.max_iterations):
            aggregates_fieldnames.append(f"iteration_{i + 1}")
            rad_agts_fieldnames.append(f"iteration_{i + 1}")
            rad_grps_fieldnames.append(f"iteration_{i + 1}")
            for hierarchy in self.hierarchies:
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
        aggregates_path: str = f"{self.root_dir}/aggregate_opinions.csv"
        rad_agts_path: str = f"{self.root_dir}/radicalised_agents.csv"
        rad_grps_path: str = f"{self.root_dir}/radicalised_groups.csv"
        polarisation_path: str = f"{self.root_dir}/polarisations.csv"

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

    def calculate_opinion_statistics(self) -> tuple[list[float], list[float]]:
        """
        Calculates the basic statistics of the aggregate opinion across the models.

        :return: The average and standard deviation of the aggregate opinions across the models.
        :rtype: tuple[list[float], list[float]]
        """
        average_opinions: list[float] = []
        opinions_sd: list[float] = []

        for i in range(self.max_iterations):
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

        for i in range(self.max_iterations):
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

        for i in range(self.max_iterations):
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

        for hierarchy in self.hierarchies:
            average_polarisation[hierarchy] = []
            polarisation_sd[hierarchy] = []

        for i in range(self.max_iterations):
            iteration_values: dict[str, list[float]] = {hierarchy: [] for hierarchy in self.hierarchies}

            for hierarchy_dict in self.polarisations.values():
                for hierarchy, hierarchy_values in hierarchy_dict.items():
                    iteration_values[hierarchy].append(hierarchy_values[i])

            for hierarchy, values_list in iteration_values.items():
                hierarchy_average: float = np.average(values_list)
                hierarchy_sd: float = float(np.std(values_list))

                average_polarisation[hierarchy].append(hierarchy_average)
                polarisation_sd[hierarchy].append(hierarchy_sd)

        return average_polarisation, polarisation_sd

    def calculate_grouped_opinion_statistics(self) -> dict[str, tuple[list[float], list[float]]]:
        """
        Calculates the basic statistics of the aggregate opinion for the MINNG and NONMN communities separately.

        :return: The average and standard deviation of the aggregate opinions across the communities.
        :rtype: dict[str, tuple[list[float], list[float]]]
        """
        return_dict: dict[str, tuple[list[float], list[float]]] = {}

        for community in self.communities:
            average_opinions: list[float] = []
            opinions_sd: list[float] = []

            for i in range(self.max_iterations):
                iteration_values: list[float] = []

                for j in range(self.repetitions):
                    model_name: str
                    if j == 0:
                        model_name = community
                    else:
                        model_name = f"{community}{j + 1}"

                    iteration_values.append(self.aggregate_opinions[model_name][i])

                iteration_average: float = np.average(iteration_values)
                iteration_sd: float = float(np.std(iteration_values))

                average_opinions.append(iteration_average)
                opinions_sd.append(iteration_sd)

            return_dict[community] = (average_opinions, opinions_sd)

        return return_dict

    def calculate_grouped_rad_agts_statistics(self) -> dict[str, tuple[list[float], list[float]]]:
        """
        Calculates the basic statistics of the radicalised agents for the MINNG and NONMN communities separately.

        :return: The average and standard deviation of the number of radicalised agents across the communities.
        :rtype: dict[str, tuple[list[float], list[float]]]
        """
        return_dict: dict[str, tuple[list[float], list[float]]] = {}

        for community in self.communities:
            average_rad_agts: list[float] = []
            rad_agts_sd: list[float] = []

            for i in range(self.max_iterations):
                iteration_values: list[int] = []

                for j in range(self.repetitions):
                    model_name: str
                    if j == 0:
                        model_name = community
                    else:
                        model_name = f"{community}{j + 1}"

                    iteration_values.append(self.radicalised_agents[model_name][i])

                iteration_average: float = np.average(iteration_values)
                iteration_sd: float = float(np.std(iteration_values))

                average_rad_agts.append(iteration_average)
                rad_agts_sd.append(iteration_sd)

            return_dict[community] = (average_rad_agts, rad_agts_sd)

        return return_dict

    def calculate_grouped_rad_grps_statistics(self) -> dict[str, tuple[list[float], list[float]]]:
        """
        Calculates the basic statistics of the radicalised groups for the MINNG and NONMN communities separately.

        :return: The average and standard deviations of the number of radicalised groups across the communities.
        :rtype: dict[str, tuple[list[float], list[float]]]
        """
        return_dict: dict[str, tuple[list[float], list[float]]] = {}

        for community in self.communities:
            average_rad_grps: list[float] = []
            rad_grps_sd: list[float] = []

            for i in range(self.max_iterations):
                iteration_values: list[int] = []

                for j in range(self.repetitions):
                    model_name: str
                    if j == 0:
                        model_name = community
                    else:
                        model_name = f"{community}{j + 1}"

                    iteration_values.append(self.radicalised_groups[model_name][i])

                iteration_average: float = np.average(iteration_values)
                iteration_sd: float = float(np.std(iteration_values))

                average_rad_grps.append(iteration_average)
                rad_grps_sd.append(iteration_sd)

            return_dict[community] = (average_rad_grps, rad_grps_sd)

        return return_dict

    def calculate_grouped_polarisation_statistics(self) -> dict[str, tuple[dict[str, list[float]], dict[str, list[float]]]]:
        """
        Calculates the basic statistics for the polarisations across the communities for each hierarchy.

        :return: A <community : <hierarchy : list>> nested mapping containing the average and standard deviation of the polarisation at each iteration per hierarchy per community.
        :rtype: dict[str, tuple[dict[str, list[float]], dict[str, list[float]]]]
        """
        return_dict: dict[str, tuple[dict[str, list[float]], dict[str, list[float]]]] = {}

        for community in self.communities:
            average_polarisation: dict[str, list[float]] = {}
            polarisation_sd: dict[str, list[float]] = {}

            for hierarchy in self.hierarchies:
                average_polarisation[hierarchy] = []
                polarisation_sd[hierarchy] = []

            for i in range(self.max_iterations):
                iteration_values: dict[str, list[float]] = {hierarchy: [] for hierarchy in self.hierarchies}

                for j in range(self.repetitions):
                    model_name: str
                    if j == 0:
                        model_name = community
                    else:
                        model_name = f"{community}{j + 1}"

                    for hierarchy, hierarchy_values in self.polarisations[model_name].items():
                        iteration_values[hierarchy].append(hierarchy_values[i])

                for hierarchy, values_list in iteration_values.items():
                    hierarchy_average: float = np.average(values_list)
                    hierarchy_sd: float = float(np.std(values_list))

                    average_polarisation[hierarchy].append(hierarchy_average)
                    polarisation_sd[hierarchy].append(hierarchy_sd)

            return_dict[community] = (average_polarisation, polarisation_sd)

        return return_dict

    def calculate_all_stats(self) -> None:
        """
        A helper function that calls all other relevant statistics calculation functions,
        formats them into an OutputDict, and adds it as an attribute to the AnalysisResults.
        """
        aggregate_opinions: tuple[list[float], list[float]] = self.calculate_opinion_statistics()
        radical_agents: tuple[list[float], list[float]] = self.calculate_rad_agt_statistics()
        radical_groups: tuple[list[float], list[float]] = self.calculate_rad_grp_statistics()
        polarisations: tuple[dict[str, list[float]], dict[str, list[float]]] = self.calculate_polarisation_statistics()
        output_dict: OutputDict = {
            "opinion_statistics": aggregate_opinions,
            "rad_agts_statistics": radical_agents,
            "rad_grps_statistics": radical_groups,
            "polarisation_statistics": polarisations,
        }
        grouped_aggregate_opinions: dict[str, tuple[list[float], list[float]]] = self.calculate_grouped_opinion_statistics()
        grouped_radical_agents: dict[str, tuple[list[float], list[float]]] = self.calculate_grouped_rad_agts_statistics()
        grouped_radical_groups: dict[str, tuple[list[float], list[float]]] = self.calculate_grouped_rad_grps_statistics()
        grouped_polarisations: dict[str, tuple[dict[str, list[float]], dict[str, list[float]]]] = self.calculate_grouped_polarisation_statistics()
        grouped_output_dict: GroupedOutputDict = {
            "opinion_statistics": grouped_aggregate_opinions,
            "rad_agts_statistics": grouped_radical_agents,
            "rad_grps_statistics": grouped_radical_groups,
            "polarisation_statistics": grouped_polarisations,
        }
        self.analysis_statistics = output_dict
        self.grouped_analysis_statistics = grouped_output_dict
        return None

    def plot_var_over_models(self) -> None:
        """
        Plots the variance of the aggregate opinions across the models.
        """
        current_opinion_values: list[float] = []

        y_values: list[float] = []  # Variance of n models' values
        x_values: list[int] = []  # Number of models

        for idx, values in enumerate(self.aggregate_opinions.values()):
            current_opinion_values += deepcopy(values)

            current_vals_variance: float = float(np.var(current_opinion_values))

            y_values.append(current_vals_variance)
            x_values.append(idx + 1)

        fig, ax = plt.subplots()

        _ = ax.plot(x_values, y_values, "--k")
        _ = ax.set_xlabel("Number of Models")
        _ = ax.set_ylabel("Aggregate Opinion Variance")
        _ = ax.set_title("Variance of Aggregate Opinions by Number of Models Used")

        save_path: str = f"{self.root_dir}/GroupVarianceOverModels.png"

        plt.savefig(save_path, dpi=300.0)

        return None

    def plot_model_runtimes(self) -> None:
        """
        Plot the runtime opinion values for all models in the experiment, along with an overall average trend.
        """
        iterations: list[int] = [i + 1 for i in range(self.max_iterations)]
        fig, ax = plt.subplots()

        for values in self.aggregate_opinions.values():
            _ = ax.plot(iterations, values, "-k", linewidth=0.7, alpha=0.25)

        _ = ax.plot(
            iterations,
            self.analysis_statistics["opinion_statistics"][0],
            "-r",
            linewidth=0.8,
            alpha=1.0,
            label="Average",
        )

        _ = ax.legend()
        _ = ax.set_xlabel("Iterations")
        _ = ax.set_ylabel("Aggregate Opinion")
        _ = ax.set_title("Model Aggregate Opinions over Iterations")

        save_path: str = f"{self.root_dir}/GroupModelRuntimes.png"

        plt.savefig(save_path, dpi=300.0)

        return None

    def plot_parameter_whiskers(self) -> None:
        """
        Plot box and whiskers for each of the model parameters.
        """
        unpacked_results: list[list[float | int]] = []
        tick_labels: list[str] = []

        for hierarchy, statistics in self.analysis_statistics["polarisation_statistics"][0].items():
            unpacked_results.append(deepcopy(statistics))
            tick_labels.append(f"Hierarchy Polarisation ({hierarchy})")

        fig, ax = plt.subplots()

        box_plot = ax.boxplot(
            unpacked_results, notch=False, orientation="vertical", whis=1.5,
        )
        _ = plt.setp(box_plot["boxes"], color="black")
        _ = plt.setp(box_plot["whiskers"], color="black")
        _ = plt.setp(box_plot["fliers"], color="red", marker="+")

        ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)

        _ = ax.set(
            axisbelow=True,
            title="Group ResultsVariance -- Polarisation Statistics",
            xlabel="Parameter",
            ylabel="Value",
        )

        _ = ax.set_xticklabels(tick_labels, rotation=0, fontsize=8)

        save_path: str = f"{self.root_dir}/GroupPolarisationStatistics.png"

        plt.savefig(save_path, dpi=300.0)

        unpacked_opinion: list[float] = deepcopy(self.analysis_statistics["opinion_statistics"][0])
        opinion_label: list[str] = ["Aggregate Opinions"]

        fix, ax = plt.subplots()

        box_plot = ax.boxplot(
            unpacked_opinion, notch=False, orientation="vertical", whis=1.5,
        )
        _ = plt.setp(box_plot["boxes"], color="black")
        _ = plt.setp(box_plot["whiskers"], color="black")
        _ = plt.setp(box_plot["fliers"], color="red", marker="+")

        ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)

        _ = ax.set(
            axisbelow=True,
            title="Group ResultsVariance -- Aggregate Opinion Statistics",
            xlabel="Parameter",
            ylabel="Value",
        )

        _ = ax.set_xticklabels(opinion_label, rotation=0, fontsize=8)

        save_path = f"{self.root_dir}/GroupOpinionStatistics.png"

        plt.savefig(save_path, dpi=300.0)

        unpacked_rad_agt: list[float] = deepcopy(self.analysis_statistics["rad_agts_statistics"][0])
        rad_agt_label: list[str] = ["Radicalised Agents"]

        fig, ax = plt.subplots()

        box_plot = ax.boxplot(
            unpacked_rad_agt, notch=False, orientation="vertical", whis=1.5,
        )
        _ = plt.setp(box_plot["boxes"], color="black")
        _ = plt.setp(box_plot["whiskers"], color="black")
        _ = plt.setp(box_plot["fliers"], color="red", marker="+")

        ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)

        _ = ax.set(
            axisbelow=True,
            title="Group ResultsVariance -- Radicalised Agent Statistics",
            xlabel="Parameter",
            ylabel="Value",
        )

        _ = ax.set_xticklabels(rad_agt_label, rotation=0, fontsize=8)

        save_path = f"{self.root_dir}/GroupRadicalAgentStatistics.png"

        plt.savefig(save_path, dpi=300.0)

        unpacked_rad_grp: list[float] = deepcopy(self.analysis_statistics["rad_grps_statistics"][0])
        rad_grp_label: list[str] = ["Radicalised Groups"]

        fig, ax = plt.subplots()

        box_plot = ax.boxplot(
            unpacked_rad_grp, notch=False, orientation="vertical", whis=1.5,
        )
        _ = plt.setp(box_plot["boxes"], color="black")
        _ = plt.setp(box_plot["whiskers"], color="black")
        _ = plt.setp(box_plot["fliers"], color="red", marker="+")

        ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)

        _ = ax.set(
            axisbelow=True,
            title="Group ResultsVariance -- Radicalised Groups Statistics",
            xlabel="Parameter",
            ylabel="Value",
        )

        _ = ax.set_xticklabels(rad_grp_label, rotation=0, fontsize=8)

        save_path = f"{self.root_dir}/GroupRadicalGroupStatistics.png"

        plt.savefig(save_path, dpi=300.0)

        return None


    def plot_var_over_models_grouped(self) -> None:
        """
        Plots the variance of the opinions across the models; grouped by the number of agents in each instance's population.
        """
        # TODO: Implement this
        return None


    def plot_model_runtimes_grouped(self) -> None:
        """
        Plot the runtime opinion values for all models in each population size group in the experiment, along with
        an overall average trend.
        """
        # TODO: Implement this
        return None


    def plot_parameter_whiskers_grouped(self) -> None:
        """
        Plot the box and whiskers for all model parameters at each individual population size.
        """
        # TODO: Implement this
        return None

    def create_analysis_plots(self) -> None:
        """
        A wrapper that simply ensures all non-grouped plots are created.
        """
        self.plot_var_over_models()
        self.plot_model_runtimes()
        self.plot_parameter_whiskers()
        return None

    def create_grouped_analysis_plots(self) -> None:
        """
        A wrapper that simply ensures all grouped plots are created.
        """
        self.plot_var_over_models_grouped()
        self.plot_model_runtimes_grouped()
        self.plot_parameter_whiskers_grouped()
        return None

    def create_all_plots(self) -> None:
        """
        A wrapper that simply ensures all experiment plots are created.
        """
        self.create_analysis_plots()
        self.create_grouped_analysis_plots()
        return None
