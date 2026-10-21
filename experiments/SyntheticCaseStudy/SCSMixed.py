from __future__ import annotations

import concurrent.futures
import gc
import os
import pickle
import warnings
import random as rd
from copy import deepcopy
from dataclasses import dataclass
from itertools import repeat
from multiprocessing import Pool, set_start_method
# For typechecking
from multiprocessing.pool import Pool as WorkerPool
from typing import TypedDict, NotRequired, override

import numpy as np
import polars as pl
from rustworkx import NodeIndices

from gatoh.agents import Agent
from gatoh.graphs import Graph, GraphEdge, GraphNode
from gatoh.groups import Group
from gatoh.model import ABModel
from gatoh.utils import random_coinflip

from SyntheticCaseStudy import ModelParamsInput, ModelParameters
# For class inheritance
from SyntheticCaseStudy import CaseStudyRunner


class MixedEventsRunner(CaseStudyRunner):
    """
    A class that will be used to create and handle the appropriate GATOH models from the given
    agent, group, and social hierarchy graph subdirectories.

    The expected agent, group, and graph inputs for this class are structured according to the outputs
    from the ResponseParser script.

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
    :param probability_tables: A <community : events> mapping that outlines the relative probabilities of specific events occuring at each iteration.
    :type probability_tables: dict[str, list[str]]
    :param event_effects: A <event : effect on community> mapping that outlines the effect on opinion and what proportion of the population is affected for each event.
    :type event_effects: dict[str, tuple[float, float]]
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
        probability_tables: dict[str, list[str]],
        event_effects: dict[str, tuple[float, float]],
        visdirs: dict[str, str],
        savedirs: dict[str, str],
        savefiles: dict[str, str],
        opinion_paths: dict[str, str] | None = None,
        worker_pool: WorkerPool | None = None,
        existing: bool = False,
    ) -> None:
        super().__init__(
            agent_paths,
            graph_paths,
            group_paths,
            initial_hierarchies,
            agent_parameters,
            group_parameters,
            test_parameters,
            visdirs,
            savedirs,
            savefiles,
            opinion_paths,
            worker_pool,
            existing,
        )
        self.probability_tables: dict[str, list[str]] = probability_tables
        self.event_effects: dict[str, tuple[float, float]] = event_effects

    def apply_probability_table(self, model_to_alter: ABModel) -> ABModel:
        """
        Takes the input probability tables and determines if the input model is mining-affected or not,
        to then select the correct event probability table to use for this iteration.

        The events in these tables may have positive, negative, or no effects on the opinions of the
        communities' inhabitants.

        The probability of different events ocurring will be different for the mining-affected communities
        and the non-affected communities.

        :param model_to_alter: The model whose population is being affected by a stochastic event.
        :type model_to_alter: ABModel
        :return: The model which had the stochastic event applied appropriately.
        :rtype: ABModel
        """
        return model_to_alter

    @override
    def custom_group_iterate(self, model_to_iterate: ABModel, worker_pool: WorkerPool | None = None) -> ABModel:
        """
        Custom iteration loop used for this experiment -- operates on the group objects in the models.

        :param model_to_iterate: The model that is being run.
        :type model_to_iterate: ABModel
        :param worker_pool: A pool of workers that can distribute the iteration processing amongst themselves.
        :type worker_pool: :class:`~multiprocessing.pool.Pool`, optional
        :return: The model that has been run.
        :rtype: ABModel
        """
        while model_to_iterate.current_iteration < model_to_iterate.max_iterations:
            # Initialise the logger state for the current iteration
            if model_to_iterate.current_iteration == 0:
                model_to_iterate.logger.new_iteration(init=True)
            else:
                model_to_iterate.logger.new_iteration()

            for group in model_to_iterate.groups:
                # Always store the groups' previous opinions at the start of an iteration no matter what
                group.store_previous_opinion()

            # At the start of each iteration, roll for random effects
            model_to_iterate = self.apply_probability_table(model_to_iterate)

            # Track the group opinion changes separately to prevent recursive updates
            new_group_opinions: dict[str, tuple[float, list[bool]]] = {}

            # First, calculate the opinion changes and store them
            if worker_pool is not None:
                opinion_results = worker_pool.imap(
                    model_to_iterate.group_iteration_opinion_calculation,
                    model_to_iterate.groups,
                    chunksize=10,
                )

                for opinion_result in opinion_results:
                    new_group_opinions[opinion_result[0]] = opinion_result[1]

                # Manual garbage collection
                del opinion_results
                _ = gc.collect()
            else:
                for group in model_to_iterate.groups:
                    opinion_result = model_to_iterate.group_iteration_opinion_calculation(group)
                    new_group_opinions[opinion_result[0]] = opinion_result[1]

                    # Manual garbage collection
                    del opinion_result
                    _ = gc.collect()

            model_to_iterate.group_iteration_opinion_changes(new_group_opinions)
            model_to_iterate.step()
            model_to_iterate.update(worker_pool=worker_pool)
            model_to_iterate.logger_iteration(worker_pool=worker_pool)

            # Get this iteration's print string (will be formatted appropriately based on the print interval)
            iteration_print_string: str = model_to_iterate.logger.iteration_print()
            print(iteration_print_string)

            if model_to_iterate.visualise:
                model_to_iterate.visualiser.visualiser_iteration(
                    model_to_iterate.base_graph,
                    model_to_iterate.current_iteration,
                    model_name=model_to_iterate.model_id,
                )

            if model_to_iterate.checkpointing:
                model_to_iterate.save_model()

            model_to_iterate.current_iteration += 1

        # Call the logger's save data function which handles persistence appropriately
        data_saved: bool = model_to_iterate.logger.save_data(model_to_iterate.data_file)
        if data_saved:
            print(f"\n\nGATOH logger data was successfully written to the file at path: {model_to_iterate.data_file}\n\n")

        return model_to_iterate


if __name__ == "__main__":
    MULTIPROCESSING: bool = True
    SIMULATE_GROUPS: bool = True

    SAVEDIR_ROOT: str = "./experiments/SyntheticCaseStudy/MixedEvents/Results"
    VISUALISATION_ROOT: str = f"{SAVEDIR_ROOT}/Visualisations"

    SAVEDIRS: dict[str, str] = {
        "NONMN": f"{SAVEDIR_ROOT}/NONMN",
        "NONMN2": f"{SAVEDIR_ROOT}/NONMN2",
        "NONMN3": f"{SAVEDIR_ROOT}/NONMN3",
        "NONMN4": f"{SAVEDIR_ROOT}/NONMN4",
        "NONMN5": f"{SAVEDIR_ROOT}/NONMN5",
        "MINNG": f"{SAVEDIR_ROOT}/MINNG",
        "MINNG2": f"{SAVEDIR_ROOT}/MINNG2",
        "MINNG3": f"{SAVEDIR_ROOT}/MINNG3",
        "MINNG4": f"{SAVEDIR_ROOT}/MINNG4",
        "MINNG5": f"{SAVEDIR_ROOT}/MINNG5",
    }
    SAVEFILES: dict[str, str] = {
        "NONMN": f"{SAVEDIRS['NONMN']}/NONMN_model_variables.csv",
        "NONMN2": f"{SAVEDIRS['NONMN2']}/NONMN2_model_variables.csv",
        "NONMN3": f"{SAVEDIRS['NONMN3']}/NONMN3_model_variables.csv",
        "NONMN4": f"{SAVEDIRS['NONMN4']}/NONMN4_model_variables.csv",
        "NONMN5": f"{SAVEDIRS['NONMN5']}/NONMN5_model_variables.csv",
        "MINNG": f"{SAVEDIRS['MINNG']}/MINNG_model_variables.csv",
        "MINNG2": f"{SAVEDIRS['MINNG2']}/MINNG2_model_variables.csv",
        "MINNG3": f"{SAVEDIRS['MINNG3']}/MINNG3_model_variables.csv",
        "MINNG4": f"{SAVEDIRS['MINNG4']}/MINNG4_model_variables.csv",
        "MINNG5": f"{SAVEDIRS['MINNG5']}/MINNG5_model_variables.csv",
    }
    VISDIRS: dict[str, str] = {
        "NONMN": f"{VISUALISATION_ROOT}/NONMN",
        "NONMN2": f"{VISUALISATION_ROOT}/NONMN2",
        "NONMN3": f"{VISUALISATION_ROOT}/NONMN3",
        "NONMN4": f"{VISUALISATION_ROOT}/NONMN4",
        "NONMN5": f"{VISUALISATION_ROOT}/NONMN5",
        "MINNG": f"{VISUALISATION_ROOT}/MINNG",
        "MINNG2": f"{VISUALISATION_ROOT}/MINNG2",
        "MINNG3": f"{VISUALISATION_ROOT}/MINNG3",
        "MINNG4": f"{VISUALISATION_ROOT}/MINNG4",
        "MINNG5": f"{VISUALISATION_ROOT}/MINNG5",
    }
    AGENT_PATHS: dict[str, str] = {
        "NONMN": "./experiments/SyntheticCaseStudy/MixedEvents/Agents/NONMN_Agents",
        "NONMN2": "./experiments/SyntheticCaseStudy/MixedEvents/Agents/NONMN_Agents",
        "NONMN3": "./experiments/SyntheticCaseStudy/MixedEvents/Agents/NONMN_Agents",
        "NONMN4": "./experiments/SyntheticCaseStudy/MixedEvents/Agents/NONMN_Agents",
        "NONMN5": "./experiments/SyntheticCaseStudy/MixedEvents/Agents/NONMN_Agents",
        "MINNG": "./experiments/SyntheticCaseStudy/MixedEvents/Agents/MINNG_Agents",
        "MINNG2": "./experiments/SyntheticCaseStudy/MixedEvents/Agents/MINNG_Agents",
        "MINNG3": "./experiments/SyntheticCaseStudy/MixedEvents/Agents/MINNG_Agents",
        "MINNG4": "./experiments/SyntheticCaseStudy/MixedEvents/Agents/MINNG_Agents",
        "MINNG5": "./experiments/SyntheticCaseStudy/MixedEvents/Agents/MINNG_Agents",
    }
    GRAPH_PATHS: dict[str, str] = {
        "NONMN": "./experiments/SyntheticCaseStudy/MixedEvents/Graphs/NONMN_Graphs",
        "NONMN2": "./experiments/SyntheticCaseStudy/MixedEvents/Graphs/NONMN_Graphs",
        "NONMN3": "./experiments/SyntheticCaseStudy/MixedEvents/Graphs/NONMN_Graphs",
        "NONMN4": "./experiments/SyntheticCaseStudy/MixedEvents/Graphs/NONMN_Graphs",
        "NONMN5": "./experiments/SyntheticCaseStudy/MixedEvents/Graphs/NONMN_Graphs",
        "MINNG": "./experiments/SyntheticCaseStudy/MixedEvents/Graphs/MINNG_Graphs",
        "MINNG2": "./experiments/SyntheticCaseStudy/MixedEvents/Graphs/MINNG_Graphs",
        "MINNG3": "./experiments/SyntheticCaseStudy/MixedEvents/Graphs/MINNG_Graphs",
        "MINNG4": "./experiments/SyntheticCaseStudy/MixedEvents/Graphs/MINNG_Graphs",
        "MINNG5": "./experiments/SyntheticCaseStudy/MixedEvents/Graphs/MINNG_Graphs",
    }
    GROUP_PATHS: dict[str, str] = {
        "NONMN": "./experiments/SyntheticCaseStudy/MixedEvents/Groups/NONMN_Groups",
        "NONMN2": "./experiments/SyntheticCaseStudy/MixedEvents/Groups/NONMN_Groups",
        "NONMN3": "./experiments/SyntheticCaseStudy/MixedEvents/Groups/NONMN_Groups",
        "NONMN4": "./experiments/SyntheticCaseStudy/MixedEvents/Groups/NONMN_Groups",
        "NONMN5": "./experiments/SyntheticCaseStudy/MixedEvents/Groups/NONMN_Groups",
        "MINNG": "./experiments/SyntheticCaseStudy/MixedEvents/Groups/MINNG_Groups",
        "MINNG2": "./experiments/SyntheticCaseStudy/MixedEvents/Groups/MINNG_Groups",
        "MINNG3": "./experiments/SyntheticCaseStudy/MixedEvents/Groups/MINNG_Groups",
        "MINNG4": "./experiments/SyntheticCaseStudy/MixedEvents/Groups/MINNG_Groups",
        "MINNG5": "./experiments/SyntheticCaseStudy/MixedEvents/Groups/MINNG_Groups",
    }

    BASE_HIERARCHIES: list[str] = [
        # Removing Age and Gender as full graphs for now, as these are much too densely connected for reasonable runtimes
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
        "Cultural": (0.0, 0.15),
        "Religious": (0.0, 0.1),
        "Geographical": (0.0, 0.0),
        "Social": (0.0, 0.1),
    }

    class ModelTestParameters(TypedDict):
        """
        A helper class for typechecking the model-specific TEST_PARAMETERS.
        """
        model_id: str
        iterations: int
        hierarchies: list[str]
        hierarchy_rw: dict[str, tuple[float, float]]

    class DefaultTestParameters(TypedDict):
        """
        A helper class for typechecking the default (shared) values in TEST_PARAMETERS.
        """
        iterations: int
        hierarchies: list[str]
        hierarchy_rw: dict[str, tuple[float, float]]
        agent_opinion_rw: tuple[float, float]
        silencing_threshold: float
        negation_threshold: float
        radicalisation_threshold: float
        suppress_warnings: bool

    class TestParameters(TypedDict):
        """
        A helper class for typechecking the TEST_PARAMETERS container.
        """
        NONMN: ModelTestParameters
        NONMN2: ModelTestParameters
        NONMN3: ModelTestParameters
        NONMN4: ModelTestParameters
        NONMN5: ModelTestParameters
        MINNG: ModelTestParameters
        MINNG2: ModelTestParameters
        MINNG3: ModelTestParameters
        MINNG4: ModelTestParameters
        MINNG5: ModelTestParameters
        DEFAULT: DefaultTestParameters

    # The relevant parameters that are defined for the model instances
    TEST_PARAMETERS: TestParameters = {
        "NONMN": {
            "model_id": "NONMN",
            "iterations": 100,
            "hierarchies": BASE_HIERARCHIES,
            "hierarchy_rw": HIERARCHY_RW,
        },
        "NONMN2": {
            "model_id": "NONMN2",
            "iterations": 100,
            "hierarchies": BASE_HIERARCHIES,
            "hierarchy_rw": HIERARCHY_RW,
        },
        "NONMN3": {
            "model_id": "NONMN3",
            "iterations": 100,
            "hierarchies": BASE_HIERARCHIES,
            "hierarchy_rw": HIERARCHY_RW,
        },
        "NONMN4": {
            "model_id": "NONMN4",
            "iterations": 100,
            "hierarchies": BASE_HIERARCHIES,
            "hierarchy_rw": HIERARCHY_RW,
        },
        "NONMN5": {
            "model_id": "NONMN5",
            "iterations": 100,
            "hierarchies": BASE_HIERARCHIES,
            "hierarchy_rw": HIERARCHY_RW,
        },
        "MINNG": {
            "model_id": "MINNG",
            "iterations": 100,
            "hierarchies": BASE_HIERARCHIES,
            "hierarchy_rw": HIERARCHY_RW,
        },
        "MINNG2": {
            "model_id": "MINNG2",
            "iterations": 100,
            "hierarchies": BASE_HIERARCHIES,
            "hierarchy_rw": HIERARCHY_RW,
        },
        "MINNG3": {
            "model_id": "MINNG3",
            "iterations": 100,
            "hierarchies": BASE_HIERARCHIES,
            "hierarchy_rw": HIERARCHY_RW,
        },
        "MINNG4": {
            "model_id": "MINNG4",
            "iterations": 100,
            "hierarchies": BASE_HIERARCHIES,
            "hierarchy_rw": HIERARCHY_RW,
        },
        "MINNG5": {
            "model_id": "MINNG5",
            "iterations": 100,
            "hierarchies": BASE_HIERARCHIES,
            "hierarchy_rw": HIERARCHY_RW,
        },
        "DEFAULT": {
            "iterations": 100,
            "hierarchies": BASE_HIERARCHIES,
            "hierarchy_rw": HIERARCHY_RW,
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

    # Used to change the number of groups that each hierarchy is clustered into (possibly extended on in the future)
    GROUP_PARAMETERS: dict[str, int] = {
        "n_groups": 10,
    }

    # Used to define the relative probability of events for each community type
    #
    # With the relative probabilities below, both the mining and non-mining communities
    # have a similar probability of "nothing" happening
    PROBABILITY_TABLE: dict[str, list[str]] = {
        "NONMN": [
            "New project announced - negative",
            "Nearby community suffering from mining",
            "Nearby community suffering from mining",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nearby community profiting from mining",
            "Nearby community profiting from mining",
            "New project announced - positive",
        ],
        "MINNG": [
            "Mine collapse",
            "Eviction of indigenous group",
            "Clearing of forested area",
            "Promises being broken by mining company",
            "Minesite layoffs for local inhabitants",
            "Repression of a community protest",
            "Environmental pollution by minesite",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Nothing",
            "Community outreach program",
            "Minor dontation from mining company",
            "Environmental and social promises are being met",
            "Moderate donation from mining company",
            "New jobs for inhabitants",
            "Large donation from mining company",
            "Extensive infrastructure investment by mining company",
        ],
    }

    # Used to specify the effect on opinion, and the proportion of inhabitants that each stochastic event affects
    #
    # Each tuple is representative of (change to individual opinions, proportion of the population) for each event
    EVENT_EFFECTS: dict[str, tuple[float, float]] = {
        "Nothing": (0.0, 0.0),
        "New project announced - negative": (-0.6, 0.4),
        "Nearby community suffering from mining": (-0.15, 0.25),
        "Nearby community profiting from mining": (0.15, 0.25),
        "New project announced - positive": (0.6, 0.4),
        "Mine collapse": (-0.8, 0.75),
        "Eviction of indigenous group": (-0.75, 0.2),
        "Clearing of forested areas": (-0.75, 0.1),
        "Promises being broken by mining company": (-0.1, 0.7),
        "Minesite layoffs for local inhabitants": (-0.9, 0.1),
        "Repression of a community protest": (-0.2, 0.4),
        "Environmental pollution by minesite": (-0.05, 0.2),
        "Community outreach program": (0.05, 0.2),
        "Minor donation from mining company": (0.2, 0.4),
        "Environmental and social promises are being met": (0.1, 0.7),
        "Moderate donation from mining company": (0.3, 0.6),
        "New jobs for inhabitants": (0.8, 0.15),
        "Large donation from mining company": (0.4, 0.8),
        "Extensive infrastructure investment by mining company": (0.8, 0.75),
    }

    # Specify the dependant variable CSV paths here
    OPINION_PATHS: dict[str, str] = {}

    # Use "spawn" as the default to optimise memory usage and cross-platform compatibility
    set_start_method("spawn", force=True)
    WORKER_POOL: WorkerPool | None = Pool() if MULTIPROCESSING else None

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

    runner: MixedEventsRunner
    if directory_missing:
        # Create the tester normally, setup the models, and begin iterations
        runner = MixedEventsRunner(
            AGENT_PATHS,
            GRAPH_PATHS,
            GROUP_PATHS,
            BASE_HIERARCHIES,
            AGENT_PARAMETERS,
            GROUP_PARAMETERS,
            TEST_PARAMETERS,
            PROBABILITY_TABLE,
            EVENT_EFFECTS,
            VISDIRS,
            SAVEDIRS,
            SAVEFILES,
            opinion_paths=None,
            worker_pool=WORKER_POOL,
        )

        if len(existing_savedirs) > 0:
            # At least one model exists
            runner.load_models(existing_saves=existing_savedirs)
            runner.create_models(missing_saves=missing_savedirs)
            runner.run_models(missing_saves=missing_savedirs, worker_pool=WORKER_POOL)
        else:
            runner.create_models()
            runner.run_models(worker_pool=WORKER_POOL)
    else:
        # Create the runner in "existing" mode, and examine the results
        runner = MixedEventsRunner(
            AGENT_PATHS,
            GRAPH_PATHS,
            GROUP_PATHS,
            BASE_HIERARCHIES,
            AGENT_PARAMETERS,
            GROUP_PARAMETERS,
            TEST_PARAMETERS,
            PROBABILITY_TABLE,
            EVENT_EFFECTS,
            VISDIRS,
            SAVEDIRS,
            SAVEFILES,
            opinion_paths=OPINION_PATHS,
            worker_pool=WORKER_POOL,
            existing=True,
        )
        runner.load_models()

    # Ensure that the multiprocessing pool is closed after all processing has finished
    if WORKER_POOL is not None:
        WORKER_POOL.terminate()
