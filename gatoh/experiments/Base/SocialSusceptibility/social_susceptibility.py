from __future__ import annotations

import os
import random as rd
from copy import deepcopy
from typing import Any

import numpy as np

import gatoh.agents.agents as agt
import gatoh.graphs.graphs as gr
import gatoh.model.model as md


class SocialSusceptibilityTester:
    """
    A test class that sets up models which are identical, and with each agent in the population having the same social susceptibility,
    but the shared susceptibility value differs across models.

    The models will contain shared social susceptibility values ranging from 0.0 to 1.0 at intervals of 0.2.

    Each instance will use identical random walk parameters, graph structures, agent hierarchy membership,
    and initial conditions, with only the shared social susceptibility value for all agents changing between models.
    """

    def __init__(self, existing: bool = False) -> None:
        """
        :param existing: A flag indicating if the experiment has already been run and saved models are present to inspect.
        """
        self.model_names: list[str] = TEST_PARAMETERS["model_names"]
        self.num_agents: int = AGENT_PARAMETERS["n_agents"]

        self.existing: bool = existing

        # Dynamic model space
        self.models: dict[str, md.ABModel] = {}

        for model_name in self.model_names:
            new_model: md.ABModel = md.ABModel(
                deepcopy(TEST_PARAMETERS["hierarchy_names"]),
                deepcopy(list(TEST_PARAMETERS["hierarchy_rw"].values())),
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
            self.create_agents()
            self.create_graphs()

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
            agent_behaviour: tuple[str, float] = (
                agent_personality,
                0.0,  # Treat ZERO as the base case
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

        # Set the agents for ZERO (base case)
        self.model_agents["ZERO"] = deepcopy(created_agents)

        # Update the social susceptibility to the new value
        for agent in created_agents:
            agent.social_susceptibility += 0.2

        # Set the agents for the next case
        self.model_agents["POINT-TWO"] = deepcopy(created_agents)

        # Repeat...
        for agent in created_agents:
            agent.social_susceptibility += 0.2
        self.model_agents["POINT-FOUR"] = deepcopy(created_agents)

        for agent in created_agents:
            agent.social_susceptibility += 0.2
        self.model_agents["POINT-SIX"] = deepcopy(created_agents)

        for agent in created_agents:
            agent.social_susceptibility += 0.2
        self.model_agents["POINT-EIGHT"] = deepcopy(created_agents)

        for agent in created_agents:
            agent.social_susceptibility += 0.2
        self.model_agents["ONE"] = deepcopy(created_agents)

        print("==== Finished Agent creation ====")

    def create_graphs(self) -> None:
        """
        Generates and sets the shared collection of social hierarchy Graph objects that will be used across the instances.
        """
        print("==== Starting Graph creation ====")
        created_graphs: list[gr.Graph] = []

        for hierarchy in TEST_PARAMETERS["hierarchy_names"]:
            graph: gr.Graph = gr.Graph(hierarchy, TEST_PARAMETERS["relationship_rw"])
            graph.generate_graph(
                deepcopy(self.model_agents["ZERO"]),
                method=TEST_PARAMETERS["graph_generation_alg"],
                relationship_range=AGENT_PARAMETERS["relationships"],
            )

            created_graphs.append(deepcopy(graph))

            # Manual garbage collection
            del graph

        # Set the graphs for all instances
        for model_name in self.model_names:
            self.model_graphs[model_name] = deepcopy(created_graphs)

        # Update the GraphNodes for all instances after ZERO
        self.update_graph_nodes("POINT-TWO")
        self.update_graph_nodes("POINT-FOUR")
        self.update_graph_nodes("POINT-SIX")
        self.update_graph_nodes("POINT-EIGHT")
        self.update_graph_nodes("ONE")

        print("==== Graph creation finished ====")
        return None

    def update_graph_nodes(self, model_name: str) -> None:
        """
        A helper function that updates the GraphNodes of all the hierarchy graphs within a model instance
        so that they contain their model's respective Agent objects.

        :param model_name: The name of the model instance for which the graph nodes are being updated.
        """
        for hierarchy_graph in self.model_graphs[model_name]:
            nodes_in_order: list[str] = []  # To store AgentIDs in order of appearance

            for node in hierarchy_graph.graph.nodes():
                nodes_in_order.append(node.agent.id)

            for idx, agent_id in enumerate(nodes_in_order):
                correct_agent: agt.Agent | None = None

                for agent_obj in self.model_agents[model_name]:
                    if agent_obj.id != agent_id:
                        continue
                    else:
                        correct_agent = deepcopy(agent_obj)
                        break

                if not correct_agent:
                    raise RuntimeError(
                        "No corresponding agent object was found -- Unable to update hierarchy GraphNode"
                    )

                new_graphnode: gr.GraphNode = gr.GraphNode(deepcopy(correct_agent))
                new_graphnode.set_index(idx)

                hierarchy_graph.graph[idx] = deepcopy(new_graphnode)

                # Manual garbage collection
                del correct_agent, new_graphnode
        return None

    def load_models(self, existing_saves: list[str] | None = None) -> None:
        """
        Loads the model objects that have been previously saved in their respective directories.

        :param existing_saves: An optional partial list of the model names representing the models that can be loaded.
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
        Adds the appropriate Agent and Graph objects to all the models.

        :missing_saves: An optional partial list of the model names representing models that should be setup.
        """
        print("==== Setting up the model instances ====")
        if missing_saves:
            for missing_save in missing_saves:
                _ = self.models[missing_save].add_agents(
                    deepcopy(self.model_agents[missing_save])
                )
                _ = self.models[missing_save].add_graphs(
                    deepcopy(self.model_graphs[missing_save]),
                    deepcopy(TEST_PARAMETERS["hierarchy_names"]),
                    deepcopy(TEST_PARAMETERS["hierarchy_rw"]),
                )
            return None

        for model_name in self.model_names:
            _ = self.models[model_name].add_agents(
                deepcopy(self.model_agents[model_name])
            )
            _ = self.models[model_name].add_graphs(
                deepcopy(self.model_graphs[model_name]),
                deepcopy(TEST_PARAMETERS["hierarchy_names"]),
                deepcopy(TEST_PARAMETERS["hierarchy_rw"]),
            )
        return None

    def run_models(self, missing_saves: list[str] | None = None) -> None:
        """
        Runs each model instance in the tester class.

        :missing_saves: An optional partial list of the model names representing models that should be run.
        """
        print("==== Beginning model iterations ====\n\n")
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
    # The relevant parameters that are being applied to this experiment
    TEST_PARAMETERS: dict[str, Any] = {
        "iterations": 100,
        "model_names": [
            "ZERO",
            "POINT-TWO",
            "POINT-FOUR",
            "POINT-SIX",
            "POINT-EIGHT",
            "ONE",
        ],
        "hierarchy_names": ["A", "B", "C", "D"],
        "hierarchy_rw": {
            "A": (0.0, 0.3),
            "B": (0.0, 0.1),
            "C": (0.0, 0.45),
            "D": (0.0, 0.15),
        },
        "relationship_rw": (0.0, 0.1),
        "graph_generation_alg": "small-world",
    }

    # The parameters that will be used to create the Agent population that is shared across models
    AGENT_PARAMETERS: dict[str, Any] = {
        "n_agents": 40,
        "opinions": (-0.8, 0.8),
        "relationships": (-0.8, 0.8),
        "hierarchy_weighting": (-0.6, 0.6),
        "personal_benefit": {True: 0.25, False: 0.75},
        "id_base": "EXSS",
    }

    # The save directories for each model instance
    SAVEDIRS: dict[str, str] = {
        "ZERO": "./experiments/Base/SocialSusceptibility/SocialSusceptibility_ZERO",
        "POINT-TWO": "./experiments/Base/SocialSusceptibility/SocialSusceptibility_POINT-TWO",
        "POINT-FOUR": "./experiments/Base/SocialSusceptibility/SocialSusceptibility_POINT-FOUR",
        "POINT-SIX": "./experiments/Base/SocialSusceptibility/SocialSusceptibility_POINT-SIX",
        "POINT-EIGHT": "./experiments/Base/SocialSusceptibility/SocialSusceptibility_POINT-EIGHT",
        "ONE": "./experiments/Base/SocialSusceptibility/SocialSusceptibility_ONE",
    }

    # The save paths for each model's logger output (must point to a .csv file)
    SAVEFILES: dict[str, str] = {
        "ZERO": "./experiments/Base/SocialSusceptibility/ZERO_model_variables.csv",
        "POINT-TWO": "./experiments/Base/SocialSusceptibility/POINT-TWO_model_variables.csv",
        "POINT-FOUR": "./experiments/Base/SocialSusceptibility/POINT-FOUR_model_variables.csv",
        "POINT-SIX": "./experiments/Base/SocialSusceptibility/POINT-SIX_model_variables.csv",
        "POINT-EIGHT": "./experiments/Base/SocialSusceptibility/POINT-EIGHT_model_variables.csv",
        "ONE": "./experiments/Base/SocialSusceptibility/ONE_model_variables.csv",
    }

    tester: SocialSusceptibilityTester

    # Check for existing saved models and store the relevant information
    save_dirs: list[str] = list(os.walk("./experiments/Base/SocialSusceptibility"))[0][
        1
    ]

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
        tester = SocialSusceptibilityTester()

        if len(existing_savedirs) > 0:  # At least one model exists
            tester.load_models(existing_saves=existing_savedirs)
            tester.setup_models(missing_saves=missing_savedirs)
            tester.run_models(missing_saves=missing_savedirs)
        else:
            tester.setup_models()
            tester.run_models()
    else:
        tester = SocialSusceptibilityTester(existing=True)
        tester.load_models()

    # TODO: Add the graph visualisation functions here when they have been implemented...
