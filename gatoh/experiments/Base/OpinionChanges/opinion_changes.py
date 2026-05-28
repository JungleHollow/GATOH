from __future__ import annotations

import csv
import os
import random as rd
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import gatoh.agents.agents as agt
import gatoh.graphs.graphs as gr
import gatoh.model.model as md


@dataclass
class ModelStruct:
    """
    Dataclass that defines a structure to contain all relevant information for handling instance runtimes in this experiment.
    """

    # The ABModel object that has been created for this instance
    model: md.ABModel
    # The maximum number of iterations that the instance will run for
    max_iterations: int
    # The iteration at which the opinion changes will be introduced
    change_iteration: int
    # An <ID, hierarchy name> mapping outlining the Agents whose opinions will be changed, and the hierarchies that they belong to
    changed_agents: dict[str, list[str]] = field(default_factory=dict)
    # The current iteration that the instance is at
    current_iteration: int = 0

    def __init__(
        self,
        model: md.ABModel,
        max_iterations: int,
        change_iteration: int,
        changed_agents: dict[str, list[str]],
    ) -> None:
        """
        Store the instance model and the relevant iteration information to be able to introduce the opinion changes during runtime.

        :param model: The ABModel object that has been created for this instance.
        :param max_iterations: The total number of iterations that this instance will run for.
        :param change_iteration: The iteration during which the opinion changes will be introduced.
        :param changed_agents: An <ID, hierarchy name> mapping outlining the Agents whose opinions will be changed, and the hierarchies they belong to.
        """
        self.model = model
        self.current_iteration = 0
        self.max_iterations = max_iterations
        self.change_iteration = change_iteration
        self.changed_agents = changed_agents


class OpinionChangesTester:
    """
    A test class that sets up models which are identical in every way, but sudden and strong opinion changes will
    be introduced in random agents at random iterations during their runtime.

    For this experiment, opinion changes will only be introduced once during a model's runtime, using a randomly
    sampled subset of agents, and a mechanism that will generate significantly different opinion values to assign each
    agent, regardless of the initial polarity or magnitude of the opinion.

    Each instance will use identical random walk parameters, graph structures, agent hierarchy membership,
    and initial conditions, with only the iterations and agents at which opinion changes are introduced differing between models.
    """

    def __init__(self, existing: bool = False) -> None:
        """
        :param existing: A flag indicating if the experiment has already been run and saved models are present to inspect.
        """
        self.num_agents: int = AGENT_PARAMETERS["n_agents"]
        self.model_intervals: list[int] = list(
            np.linspace(
                0,
                TEST_PARAMETERS["iterations"],
                num=int(
                    TEST_PARAMETERS["iterations"]
                    // TEST_PARAMETERS["opinion_change_interval"]
                ),
                dtype=np.int64,
            )
        )
        self.model_repeats: int = TEST_PARAMETERS["repeats"]

        self.existing: bool = existing
        self.model_saves: dict[str, str] = {}

        # Dynamic model space
        self.models: list[ModelStruct] = []

        # Define the lists that will contain the populations of Agents and Graphs
        self.model_agents: list[agt.Agent] = []
        self.model_graphs: list[gr.Graph] = []

        if not self.existing:
            self.create_agents()
            self.create_graphs(self.model_agents)
        else:
            self.model_saves = SAVEDIRS

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

        print("==== Finished Agent creation ====")
        return None

    def create_graphs(self, agents: list[agt.Agent]) -> None:
        """
        Generates and sets the shared collection of social hierarchy Graph objects that will be used across the instances.
        """
        print("==== Starting Graph creation ====")
        created_graphs: list[gr.Graph] = []

        # Workaround to allow for np random choice
        agent_indices: list[int] = [i for i in range(len(agents))]

        for hierarchy in TEST_PARAMETERS["hierarchy_names"]:
            graph: gr.Graph = gr.Graph(hierarchy, TEST_PARAMETERS["relationship_rw"])

            if (
                hierarchy == "A"
            ):  # Ensures that every Agent in the population belongs to at least one hierarchy
                graph.generate_graph(
                    deepcopy(agents),
                    method=TEST_PARAMETERS["graph_generation_alg"],
                    relationship_range=AGENT_PARAMETERS["relationships"],
                )
            else:
                hierarchy_n_agents: int = rd.randint(
                    self.num_agents // 10, self.num_agents
                )
                selected_agents: list[int] = list(
                    np.random.choice(
                        agent_indices, size=hierarchy_n_agents, replace=False
                    )
                )

                agent_sample: list[agt.Agent] = []
                for index in selected_agents:
                    agent_sample.append(deepcopy(agents[index]))

                graph.generate_graph(
                    deepcopy(agent_sample),
                    method=TEST_PARAMETERS["graph_generation_alg"],
                    relationship_range=AGENT_PARAMETERS["relationships"],
                )

                # Manual garbage colleciton
                del agent_sample

            created_graphs.append(deepcopy(graph))

            # Manual garbage collection
            del graph
        print("==== Graph creation finished ====")
        return None

    def load_models(self, existing_saves: list[str] | None = None) -> None:
        """
        Loads the model objects that have been previously saved in their respective directories.

        :param existing_saves: An optional partial list of the model names representing the models that can be loaded.
        """
        pass

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

            for model_struct in self.models:
                csv_row: dict[str, str] = {
                    model_struct.model.model_id: model_struct.model.save_dir
                }
                csv_writer.writerow(csv_row)

        return None

    def initialise_model_structs(self) -> None:
        """
        Uses the relevant information provided to create appropriate ABModel objects and wrap them in a ModelStruct.
        """
        # Workaround to allow for np random choice
        agent_indices: list[int] = [i for i in range(len(self.model_agents))]

        for interval in self.model_intervals:
            for repeat in range(self.model_repeats):
                # Generate the unique model ID for this instance
                model_name: str = f"ITER-{interval:03}_NUM-{repeat + 1:02}"

                # Create the ABModel object for this instance
                new_model: md.ABModel = md.ABModel(
                    deepcopy(TEST_PARAMETERS["hierarchy_names"]),
                    deepcopy(list(TEST_PARAMETERS["hierarchy_rw"].values())),
                    save_dir=f"{SAVEDIR_ROOT}/OpinionChanges_{model_name}",
                    data_file=f"{SAVEDIR_ROOT}/OpinionChanges_{model_name}/{model_name}_model_variables.csv",
                    model_id=model_name,
                )

                # Add the Agents and Graphs to the new model
                _ = new_model.add_agents(deepcopy(self.model_agents))
                _ = new_model.add_graphs(
                    deepcopy(self.model_graphs),
                    deepcopy(TEST_PARAMETERS["hierarchy_names"]),
                    deepcopy(TEST_PARAMETERS["hierarchy_rw"]),
                )

                # Sample Agents for which the opinion changes will be introduced in
                agents_to_change: int = rd.randint(1, self.num_agents)
                agent_ids: list[str] = list(
                    np.random.choice(
                        agent_indices, size=agents_to_change, replace=False
                    )
                )

                # Extract the names of all the hierarchies that each sampled Agent belongs to in the model
                changed_agents: dict[str, list[str]] = {}
                for agent_id in agent_ids:
                    agent_obj: Any = new_model.agents.get_agent_by_id(agent_id)
                    agent_hierarchies: list[str] = (
                        new_model.graphs.get_agent_hierarchies(agent_obj)
                    )
                    changed_agents[agent_id] = deepcopy(agent_hierarchies)

                # Create the ModelStruct object
                model_struct: ModelStruct = ModelStruct(
                    deepcopy(new_model),
                    TEST_PARAMETERS["iterations"],
                    interval,
                    changed_agents,
                )

                self.models.append(deepcopy(model_struct))

                # Manual garbage collection
                del new_model, model_struct

        self.create_savedir_validation()
        return None

    def setup_models(self, missing_saves: list[str] | None = None) -> None:
        """
        Adds copies of the shared Agent and Graph populations to all the models.

        :param missing_saves: An optional partial list of the model names representing models that should be setup.
        """
        pass

    def run_models(self, missing_saves: list[str] | None = None) -> None:
        """
        Runs each model instance in the tester class, calling the custom iteration function, and introducing the
        sudden opinion changes at the correct iteration.

        :param missing_saves: An optional partial list of the model names representing models that should be run.
        """
        pass

    def custom_iterate(self, model_struct: ModelStruct) -> None:
        """
        A custom model iteration function that is able to introduce the opinion changes at the correct iterations across
        instances.

        :param model_struct: A ModelStruct object containing all the relevant information needed to handle the model runtime.
        """
        pass


if __name__ == "__main__":
    # The relevant parameters that are defined for the identical model instances
    TEST_PARAMETERS: dict[str, Any] = {
        "iterations": 100,
        "opinion_change_interval": 20,
        "repeats": 5,
        "hierarchy_names": ["A", "B", "C", "D", "E", "F"],
        "hierarchy_rw": {
            "A": (0.0, 0.3),
            "B": (0.0, 0.1),
            "C": (0.0, 0.45),
            "D": (0.0, 0.15),
            "E": (0.0, 0.05),
            "F": (0.0, 0.25),
        },
        "relationship_rw": (0.0, 0.1),
        "graph_generation_alg": "small-world",
        "use_subsetting": True,
    }

    # The parameters that will be used to create the Agent population that is shared across models
    AGENT_PARAMETERS: dict[str, Any] = {
        "n_agents": 40,
        "opinions": (-0.9, 0.9),
        "relationships": (-0.9, 0.9),
        "hierarchy_weighting": (-0.75, 0.75),
        "personal_benefit": {True: 0.3, False: 0.7},
        "social_susceptibility": (0.0, 1.0),
        "id_base": "EXOC",
    }

    # The root of the directory in which each instance's save directory will be located
    SAVEDIR_ROOT: str = "./gatoh/experiments/Base/OpinionChanges"

    # A path to which a validation file will be written -- outlining the model name and generated save directory that were generated
    # for each instance during the tester initialisation (to allow for checking of missing saves in the future)
    LOGGED_SAVEDIRS: str = (
        "./gatoh/experiments/Base/OpinionChanges/OpinionChanges_logged_savedirs.csv"
    )

    # A <model name, path> mapping of all the model instances that were initially created by the tester
    SAVEDIRS: dict[str, str] = {}

    tester: OpinionChangesTester

    # Check for existing saved models and store the relevant information
    save_dirs: list[str] = list(os.walk(SAVEDIR_ROOT))[0][1]

    directory_missing: bool = False
    existing_savedirs: list[str] = []
    missing_savedirs: list[str] = []

    if not os.path.exists(
        LOGGED_SAVEDIRS
    ):  # The tester has not yet been run, or the validation file was removed
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
        tester = OpinionChangesTester()

        if len(existing_savedirs) > 0:  # At least one model exists
            tester.load_models(existing_saves=existing_savedirs)
            tester.setup_models(missing_saves=missing_savedirs)
            tester.run_models(missing_saves=missing_savedirs)
        else:
            tester.setup_models()
            tester.run_models()
    else:
        tester = OpinionChangesTester(existing=True)
        tester.load_models()

    # TODO: Add the graph visualisation functions here when they have been implemented...
