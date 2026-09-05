from __future__ import annotations

import gc
import os
import random as rd
from copy import deepcopy
from typing import TypedDict

import gatoh.agents as agt
import gatoh.graphs as gr
import gatoh.groups as grp
import gatoh.model as md


class GraphAlgTester:
    """
    A test class that sets up and runs models for each of the random graph generation algorithms in the GATOH framework
    for an experimental comparison.

    Particularly, this test class will be clustering the agents into groups, generating random group graphs, and then
    running the models for the clustered group graphs.
    """

    def __init__(self, existing: bool = False) -> None:
        """
        :param existing: A flag indicating if the tester is loading an existing experiment.
        """
        # Store the class parameters within the instance
        self.algorithms: list[str] = TEST_PARAMETERS["generation_algorithms"]
        self.num_agents: int = TEST_PARAMETERS["num_agents"]
        self.num_groups: int = TEST_PARAMETERS["num_groups"]

        self.existing: bool = existing

        # Define the data types without assigning values
        self.model_agents: list[agt.Agent]
        self.model_graphs: dict[str, gr.GroupGraph]
        self.model_groups: list[grp.Group]

        # Dynamic model space
        self.models: dict[str, md.ABModel] = {}

        # Create the model objects no matter what
        for algorithm in self.algorithms:
            algorithm_model: md.ABModel = md.ABModel(
                HIERARCHY_NAMES,
                HIERARCHY_RW_DISTRIBUTIONS,
                save_dir=MODEL_SAVEDIRS[algorithm],
                data_file=MODEL_DATAFILES[algorithm],
                model_id=algorithm.upper(),
                simulate_groups=True,
            )
            self.models[algorithm] = algorithm_model

        # Objects are needed to define and run the models
        if not self.existing:
            self.model_agents = self.create_agents(self.num_agents)

            self.model_graphs = {}
            for algorithm in self.algorithms:
                algorithm_graph: gr.GroupGraph = self.create_group_graph(
                    algorithm,
                    HIERARCHY_NAMES,
                    HIERARCHY_RW_DISTRIBUTIONS,
                    self.model_agents,
                )
                self.model_graphs[algorithm] = algorithm_graph

    def create_agents(self, num_agents: int) -> list[agt.Agent]:
        """
        Generates and returns the population of Agents that will be used across all models.

        For the purposes of this experiment, the populations will be identical across all models,
        and only the way that relationships between Groups are generated will change.

        :param num_agents: The number of agents to generate for the total population.
        :type num_agents: int
        :return: The total Agent population that will be used for the models in this experiment.
        :rtype: list[Agent]
        """
        created_agents: list[agt.Agent] = []

        opinion_range: tuple[float, float] = AGENT_CHARACTERISTICS["opinion"]

        # Define the data types but do not assign any values
        agent_id: str
        agent_opinion: float
        agent: agt.Agent
        agent_behaviour: tuple[str, float]
        personal_benefit: bool

        created_count: int = 0
        while created_count < num_agents:
            created_count += 1
            agent_id = f"AGNT{created_count:04}"

            # Stochastically generate the important Agent attributes
            agent_opinion = rd.uniform(opinion_range[0], opinion_range[1])
            agent_behaviour = (agt.draw_personality(), rd.uniform(0.0, 1.0))
            personal_benefit = rd.choice([True, False])

            agent = agt.Agent(
                agent_id,
                HIERARCHY_WEIGHTINGS,
                agent_opinion,
                agent_behaviour,
                personal_benefit,
            )

            created_agents.append(agent)

            # Manual garbage collection
            del agent
            _ = gc.collect()

        return created_agents

    def create_group_graph(self) -> gr.GroupGraph:
        """
        """
        pass

    def load_models(self, existing_saves: list[str] | None = None) -> None:
        """
        Loads the model objects that have been previously saved at their respective directories.

        :param existing_saves: The potentially partial names of the algorithms that have an existing save.
        :type existing_saves: list[str], optional
        """
        if existing_saves is not None:
            for existing_save in existing_saves:
                self.models[existing_save].load_model(MODEL_SAVEDIRS[existing_save])
            return None

        for algorithm in self.algorithms:
            self.models[algorithm].load_model(MODEL_SAVEDIRS[algorithm])
        return None

    def setup_models(self, missing_saves: list[str] | None = None) -> None:
        """
        Adds the appropriate model objects to each relevant model.

        :param missing_saves: The potentially partial names of the algorithms that do not have an existing save.
        :type missing_saves: list[str], optional
        """

        # TODO: Add in the relevant group graph addition here when that is defined...

        if missing_saves is not None:
            for missing_save in missing_saves:
                _ = self.models[missing_save].add_agents(deepcopy(self.model_agents))
                _ = self.models[missing_save].add_groups(deepcopy(self.model_groups))
            return None

        for algorithm in self.algorithms:
            _ = self.models[algorithm].add_agents(deepcopy(self.model_agents))
            _ = self.models[algorithm].add_groups(deepcopy(self.model_groups))
        return None

    def run_models(self, missing_saves: list[str] | None = None) -> None:
        """
        Run each model in the tester class.

        :param missing_saves: The potentially partial names of the algorithms that do not have an existing save.
        :type missing_saves: list[str], optional
        """
        if missing_saves is not None:
            for missing_save in missing_saves:
                self.models[missing_save].iterate()
                self.models[missing_save].save_model()
            return None

        for algorithm in self.algorithms:
            self.models[algorithm].iterate()
            self.models[algorithm].save_model()

        return None

if __name__ == "__main__":
    class TestParameters(TypedDict):
        generation_algorithms: list[str]
        num_agents: int
        num_groups: int

    # The parameters set for the tester class itself
    TEST_PARAMETERS: TestParameters = {
        "generation_algorithms": [
            "small-world",
            "scale-free",
            "random",
            "blockmodel",
        ],
        "num_agents": 40,
        "num_groups": 4,
    }

    # Default model parameters will be used for all scenarios, no need to set explicitly

    # The social hierarchies that will exist in the models
    HIERARCHY_NAMES: list[str] = [
        "family",
        "friends",
        "neighbours",
        "religion",
        "cultural",
    ]

    # The random walk distribution parameters for each hierarchy
    HIERARCHY_RW_DISTRIBUTIONS: list[tuple[float, float]] = [
        (0.0, 0.01),  # Family
        (0.0, 0.05),  # Friends
        (0.0, 0.08),  # Neighbours
        (0.0, 0.15),  # Religion
        (0.0, 0.2),  # Cultural
    ]

    # The hierarchy weightings that will be given to each hierarchy (shared amongst all agents)
    HIERARCHY_WEIGHTINGS: dict[str, float] = {
        "family": 0.9,
        "friends": 0.7,
        "neighbours": 0.55,
        "religion": 0.5,
        "cultural": 0.25,
    }

    class AgentCharacteristics(TypedDict):
        opinion: tuple[float, float]
        connectivity: int
        relationship: tuple[float, float]

    # Defining the distributions of the characteristics for Agents that will be used in the experiment
    AGENT_CHARACTERISTICS: AgentCharacteristics = {
        "opinion": (-0.8, 0.8),
        "connectivity": 6,
        "relationship": (-0.2, 0.8),
    }

    # The root directory for the experiment
    ROOT_DIR: str = "./experiments/GroupBase/GraphAlgorithms"

    # Define the save directories for each model
    MODEL_SAVEDIRS: dict[str, str] = {}
    for algorithm in TEST_PARAMETERS["generation_algorithms"]:
        MODEL_SAVEDIRS[algorithm] = f"{ROOT_DIR}/GraphAlgorithms_{algorithm}"

    # Define the save paths for each model's logged variables (must point to a .csv file)
    MODEL_DATAFILES: dict[str, str] = {}
    for algorithm in TEST_PARAMETERS["generation_algorithms"]:
        MODEL_DATAFILES[algorithm] = f"{ROOT_DIR}/{algorithm}_model_variables.csv"

    tester: GraphAlgTester

    # Check for existing saved models and store the relevant information
    save_dirs: list[str] = list(os.walk(ROOT_DIR))[0][1]

    directory_missing: bool = False
    existing_savedirs: list[str] = []
    missing_savedirs: list[str] = []

    for algorithm, save_dir in MODEL_SAVEDIRS.items():
        dir_name: str = deepcopy(save_dir).split("/")[-1]
        if dir_name in save_dirs:
            existing_savedirs.append(algorithm)
        else:
            directory_missing = True
            missing_savedirs.append(algorithm)

    # At least one algorithm's save subdirectory does not exist
    if directory_missing:
        # Create the tester normally, setup the models, and begin iterations
        tester = GraphAlgTester()

        if len(existing_savedirs) > 0:
            # At least one model exists
            tester.load_models(existing_saves=existing_savedirs)
            tester.setup_models(missing_saves=missing_savedirs)
            tester.run_models(missing_saves=missing_savedirs)
        else:
            # Assume all models should be newly created and run
            tester.setup_models()
            tester.run_models()
    # Assume that all existing subdirectories include every algorithm's valid save subdirectory...
    else:
        # Create the tester in "existing" mode, and examine the results
        tester = GraphAlgTester(existing=True)
        tester.load_models()
