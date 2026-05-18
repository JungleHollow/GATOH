from __future__ import annotations

import os
import random as rd
from copy import deepcopy
from typing import Any

import numpy as np

import gatoh.agents as agt
import gatoh.graphs as gr
import gatoh.model as md


class RandomWalkTester:
    """
    A test class that sets up models which use a variety of random walk distributions for their
    dynamic relationships and hierarchy weighting mechanisms.

    The main scenarios are:
        - A `base' model with identical random walk parameters for all dynamic relationships and hierarchy weightings
        - A model with identical parameters for dynamic relationships and unique parameters for hierarchy weightings
        - A model with identical parameters for hierarchy weightings and unique parameters for dynamic relationships
        - A model with unique parameters across both dynamic relationships and hierarchy weightings

    For each of these models, there will also be an exploration of the effects that different strengths of parameters
    may have on emergent behaviour; i.e. positive vs. negative distribution means, small vs. large variances,
    parameter strength tied to agent influentiality, etc.

    Each instance will use identical model parameters, graphs, and agent populations; with only the
    random walk parameters being different between them.
    """

    def __init__(self, existing: bool = False) -> None:
        """
        :param existing: A flag indicating if the tester is loading an existing experiment.
        """
        self.model_names: list[str] = TEST_PARAMETERS["model_names"]
        self.num_agents: int = AGENT_PARAMETERS["n_agents"]

        self.existing: bool = existing

        # Dynamic model space
        self.models: dict[str, md.ABModel] = {}

        for model_name in self.model_names:
            new_model: md.ABModel = md.ABModel(
                deepcopy(TEST_PARAMETERS["hierarchy_names"]),
                deepcopy(
                    list(TEST_PARAMETERS["shared_hierarchy_rw"].values())
                ),  # The rw params passed to model are overridden by the explicit ones set for the Agents and Graphs...
                save_dir=SAVEDIRS[model_name],
                data_file=SAVEFILES[model_name],
                model_id=model_name,
            )
            self.models[model_name] = deepcopy(new_model)

            # Manual garbage collection
            del new_model

        # Define the dicts that will contain the populations of Agents and Graphs
        self.model_agents: dict[str, list[agt.Agent]] = {}
        self.model_graphs: dict[str, list[gr.Graph]] = {}

        if not self.existing:
            self.model_agents["BASE"] = self.create_agents()
            self.model_agents["RELS"] = deepcopy(
                self.model_agents["BASE"]
            )  # No changes in the Agents from the base case

            # HIER Model requires all Agents to have unique hierarchy rw params
            self.model_agents["HIER"] = deepcopy(self.model_agents["BASE"])
            for agent in self.model_agents["HIER"]:
                agent.rw_distributions = TEST_PARAMETERS["unique_hierarchy_rw"]

            self.model_agents["BOTH"] = deepcopy(
                self.model_agents["HIER"]
            )  # Agents require unique hierarchy params

            print("== All model Agents successfully created ==")

            # Create the BASE Graphs
            self.model_graphs["BASE"] = self.create_graphs(self.model_agents["BASE"])

            self.model_graphs["RELS"] = deepcopy(self.model_graphs["BASE"])
            for graph in self.model_graphs["RELS"]:
                # First, update all the nodes with the appropriate Agent objects
                for idx, node in enumerate(graph.graph.nodes()):
                    node.agent = deepcopy(self.model_agents["RELS"][idx])
                # For this case, unique rw parameters must be generated for each edge
                for edge in graph.graph.edges():
                    generated_mean: float = rd.uniform(
                        TEST_PARAMETERS["unique_rel_rw_range"][0][0],
                        TEST_PARAMETERS["unique_rel_rw_range"][0][1],
                    )
                    generated_variance: float = rd.uniform(
                        TEST_PARAMETERS["unique_rel_rw_range"][1][0],
                        TEST_PARAMETERS["unique_rel_rw_range"][1][1],
                    )
                    edge.set_rw_params((generated_mean, generated_variance))

            self.model_graphs["HIER"] = deepcopy(self.model_graphs["BASE"])
            for graph in self.model_graphs["HIER"]:
                # Only the nodes must be updated for this case
                for idx, node in enumerate(graph.graph.nodes()):
                    node.agent = deepcopy(self.model_agents["HIER"][idx])

            self.model_graphs["BOTH"] = deepcopy(self.model_graphs["RELS"])
            for graph in self.model_graphs["BOTH"]:
                # Only the nodes must be updated as unique rels were already generated for RELS
                for idx, node in enumerate(graph.graph.nodes()):
                    node.agent = deepcopy(self.model_agents["BOTH"][idx])

            print("== All model Graphs successfully created ==")

    def create_agents(self) -> list[agt.Agent]:
        """
        Generates and returns the population of Agents that will be used across all models.

        :return: A list containing all the created Agent objects.
        """
        print("Starting Agent creation")
        created_agents: list[agt.Agent] = []

        personalities: list[str] = list(AGENT_PARAMETERS["personality"].keys())
        personality_p: list[float] = list(AGENT_PARAMETERS["personality"].values())

        benefit_flags: list[bool] = list(AGENT_PARAMETERS["personal_benefit"].keys())
        benefit_p: list[float] = list(AGENT_PARAMETERS["personal_benefit"].values())

        for i in range(self.num_agents):
            agent_id: str = f"{AGENT_PARAMETERS['id_base']}{i + 1:04}"
            agent_opinion: float = rd.uniform(
                AGENT_PARAMETERS["opinions"][0], AGENT_PARAMETERS["opinions"][1]
            )
            agent_personality: str = str(
                np.random.choice(personalities, size=1, p=personality_p)[0]
            )
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
            # Initialise the base population with shared dynamic hierarchy rw params
            created_agent.rw_distributions = TEST_PARAMETERS["shared_hierarchy_rw"]

            created_agents.append(deepcopy(created_agent))

            # Manual garbage collection
            del created_agent

        print("Finished Agent creation")
        return deepcopy(created_agents)

    def create_graphs(self, agents: list[agt.Agent]) -> list[gr.Graph]:
        """
        Creates the graphs that will be used across all models.

        :param agents: The population of Agents that will be used to generate graphs.
        :return: A list containing all the created Graph objects
        """
        print("Starting Graph creation")
        created_graphs: list[gr.Graph] = []

        for hierarchy in TEST_PARAMETERS["hierarchy_names"]:
            graph: gr.Graph = gr.Graph(
                hierarchy, TEST_PARAMETERS["shared_relationship_rw"]
            )
            graph.generate_graph(
                deepcopy(agents),
                method=TEST_PARAMETERS["graph_generation_alg"],
                relationship_range=AGENT_PARAMETERS["relationships"],
            )

            created_graphs.append(deepcopy(graph))

            # Manual garbage collection
            del graph

        print("Finished Graph creation")
        return deepcopy(created_graphs)

    def load_models(self, existing_saves: list[str] | None = None) -> None:
        """
        Loads the model objects that have been previously saved at their respective directories.

        :param existing_saves: An optional partial list of the model names representing the existing models that can be loaded.
        """
        if existing_saves:
            for existing_save in existing_saves:
                self.models[existing_save].load_model(SAVEDIRS[existing_save])
            return None

        for model_name in self.model_names:
            self.models[model_name].load_model(SAVEDIRS[model_name])
        return None

    def setup_models(self, missing_saves: list[str] | None = None) -> None:
        """
        Adds the appropriate Agent and Graph obkects to all models.

        :param missing_saves: An optional partial list of the model names representing models that should be setup.
        """
        print("Setting up the model instances")
        if missing_saves:
            for missing_save in missing_saves:
                _ = self.models[missing_save].add_agents(
                    deepcopy(self.model_agents[missing_save])
                )
                _ = self.models[missing_save].add_graphs(
                    deepcopy(self.model_graphs[missing_save]),
                    deepcopy(TEST_PARAMETERS["hierarchy_names"]),
                    deepcopy(TEST_PARAMETERS["shared_hierarchy_rw"]),
                )
            return None

        for model_name in self.model_names:
            _ = self.models[model_name].add_agents(
                deepcopy(self.model_agents[model_name])
            )
            _ = self.models[model_name].add_graphs(
                deepcopy(self.model_graphs[model_name]),
                deepcopy(TEST_PARAMETERS["hierarchy_names"]),
                deepcopy(TEST_PARAMETERS["shared_hierarchy_rw"]),
            )
        return None

    def run_models(self, missing_saves: list[str] | None = None) -> None:
        """
        Runs each model in the tester class.

        :param missing_saves: An optional partial list of the model names representing models that should be run.
        """
        print("Beginning model iterations\n\n")
        if missing_saves:
            for missing_save in missing_saves:
                self.models[missing_save].iterate()
                self.models[missing_save].save_model()
            return None

        for model_name in self.model_names:
            self.models[model_name].iterate()
            self.models[model_name].save_model()

        return None


if __name__ == "__main__":
    # The relevant parameters that are being applied in this experiment
    TEST_PARAMETERS: dict[str, Any] = {
        "iterations": 100,
        "model_names": ["BASE", "RELS", "HIER", "BOTH"],
        "shared_hierarchy_rw": {
            "A": (0.0, 1.0),
            "B": (0.0, 1.0),
            "C": (0.0, 1.0),
            "D": (0.0, 1.0),
            "E": (0.0, 1.0),
            "F": (0.0, 1.0),
        },
        "unique_hierarchy_rw": {
            "A": (0.0, 1.5),
            "B": (0.0, 0.3),
            "C": (0.0, 0.1),
            "D": (-0.1, 0.2),
            "E": (0.0, 0.01),
            "F": (0.1, 0.2),
        },
        "shared_relationship_rw": (0.0, 0.1),
        "unique_rel_rw_range": (
            (-0.1, 0.1),
            (0.0, 0.5),
        ),  # Ranges for the mean and variance values of the rw params
        "hierarchy_names": ["A", "B", "C", "D", "E", "F"],
        "graph_generation_alg": "small-world",
    }

    # The parameters that will be used to create the shared population of Agents to be used across models
    AGENT_PARAMETERS: dict[str, Any] = {
        "n_agents": 40,
        "opinions": (-0.8, 0.8),
        "relationships": (-0.8, 0.8),
        "social_susceptibility": (0.2, 0.8),
        "hierarchy_weighting": (
            -0.6,
            0.6,
        ),  # Range of possible initial hierarchy weighting values
        "personality": {
            "social": 0.4,
            "neutral": 0.4,
            "impulsive": 0.2,
        },
        "personal_benefit": {True: 0.25, False: 0.75},
        "id_base": "EXRW",
    }

    # The save directories for each model instance
    SAVEDIRS: dict[str, str] = {
        "BASE": "./experiments/Base/RandomWalk/RandomWalk_BASE",
        "RELS": "./experiments/Base/RandomWalk/RandomWalk_RELS",
        "HIER": "./experiments/Base/RandomWalk/RandomWalk_HIER",
        "BOTH": "./experiments/Base/RandomWalk/RandomWalk_BOTH",
    }

    # The save paths for each model's logger outputs (must point to .csv files)
    SAVEFILES: dict[str, str] = {
        "BASE": "./experiments/Base/RandomWalk/BASE_model_variables.csv",
        "RELS": "./experiments/Base/RandomWalk/RELS_model_variables.csv",
        "HIER": "./experiments/Base/RandomWalk/HIER_model_variables.csv",
        "BOTH": "./experiments/Base/RandomWalk/BOTH_model_variables.csv",
    }

    tester: RandomWalkTester

    # Check for existing saved models and store the relevant information
    save_dirs = list(os.walk("./experiments/Base/RandomWalk"))[0][1]

    directory_missing: bool = False
    existing_savedirs: list[str] = []
    missing_savedirs: list[str] = []

    for model, save_dir in SAVEDIRS.items():
        dir_name: str = deepcopy(save_dir).split("/")[-1]
        if dir_name in save_dirs:
            existing_savedirs.append(model)
        else:
            directory_missing = True
            missing_savedirs.append(model)

    if directory_missing:
        tester = RandomWalkTester()

        if len(existing_savedirs) > 0:  # At least one model exists
            tester.load_models(existing_saves=existing_savedirs)
            tester.setup_models(missing_saves=missing_savedirs)
            tester.run_models(missing_saves=missing_savedirs)
        else:
            tester.setup_models()
            tester.run_models()
    else:
        tester = RandomWalkTester(existing=True)
        tester.load_models()

    # TODO: Add the graph visualisation functions here once those have been implemented...
