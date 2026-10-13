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
