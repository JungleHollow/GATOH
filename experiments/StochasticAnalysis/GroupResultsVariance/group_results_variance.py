from __future__ import annotations

import csv
import gc
import os
import pickle
import random as rd
from copy import deepcopy
from typing import Self, TypedDict

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
