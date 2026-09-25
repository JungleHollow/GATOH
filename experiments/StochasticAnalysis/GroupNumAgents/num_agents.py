from __future__ import annotations

import csv
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
    :param polarisation: The network polarisation that the model logged over its iterations.
    :type polarisation: dict[str, list[float]]
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

        # TODO: FINISH AFTER NumIterations...
