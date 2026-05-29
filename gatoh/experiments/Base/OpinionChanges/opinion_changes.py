from __future__ import annotations

import csv
import os
import pickle
import random as rd
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import gatoh.agents.agents as agt
import gatoh.graphs.graphs as gr
import gatoh.model.model as md


@dataclass
class SaveStruct:
    """
    Dataclass that is used to store the supporting information from a ModelStruct alongside an ABModel's already defined
    save files.
    """

    # All attributes correspond directly to those in ModelStruct (except for the missing `model', which is replaced with the model ID)
    model_id: str
    max_iterations: int
    change_iteration: int
    changed_agents: dict[str, list[str]] = field(default_factory=dict)
    current_iteration: int = 0

    def __init__(self, struct_to_save: ModelStruct, save_dir: str) -> None:
        """
        Extract all the accompanying information from the given ModelStruct, and then pickle the information to the
        ABModel savedir.

        :param struct_to_save: The ModelStruct object that is being saved.
        :param save_dir: The path to the save directory that was written by the ABModel.
        """
        self.model_id = deepcopy(struct_to_save.model.model_id)
        self.max_iterations = struct_to_save.max_iterations
        self.change_iteration = struct_to_save.change_iteration
        self.changed_agents = deepcopy(struct_to_save.changed_agents)
        self.current_iteration = struct_to_save.current_iteration

        # Immediately call self.pickle
        self.pickle_struct(save_dir)

    def pickle_struct(self, save_dir: str) -> None:
        """
        Serialise the accompanying information from the ModelStruct and store it in a pickle file within the
        main ABModel's save directory.

        :param save_dir: The path to the save directory that was written by the ABModel.
        """
        with open(f"{save_dir}/{self.model_id}.pkl", "wb") as pickle_file:
            pickle.dump(self.__dict__, pickle_file)


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
            self.load_agents()
            self.load_graphs()

    def get_struct(self, model_name: str) -> ModelStruct:
        """
        A getter function that iterates over self.models, returning the appropriate ModelStruct object.

        :param model_name: The unique model ID that was generated and assigned for this instance.
        :return: The ModelStruct corresponding to the model ID.
        """
        return_struct: ModelStruct | None = None

        for model_struct in self.models:
            if model_struct.model.model_id == model_name:
                return_struct = model_struct
                break

        if return_struct is None:
            raise RuntimeError(
                f"Model {model_name} was not found in the tester's instances..."
            )
        else:
            return return_struct

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

        # Serialise the created Agent objects so that they remain unchanged across future runs
        self.pickle_agents()

        print("==== Finished Agent creation ====")
        return None

    def pickle_agents(self) -> None:
        """
        Serialises the tester's initial shared Agent population to a subdirectory within the experiment directory.
        """
        agents_path: str = f"{ROOT_DIR}/agents"

        if not os.path.exists(agents_path):
            os.mkdir(agents_path)

        for agent in self.model_agents:
            agent_pickle_path: str = f"{agents_path}/agent_{agent.id}.pkl"
            with open(agent_pickle_path, "wb") as pickle_file:
                pickle.dump(agent, pickle_file)

        return None

    def load_agents(self) -> None:
        """
        Deserialises the tester's initial shared Agent population and loads them into memory.
        """
        agents_path: str = f"{ROOT_DIR}/agents"

        for i in range(self.num_agents):
            agent_id: str = f"{AGENT_PARAMETERS['id_base']}{i + 1:04}"
            agent_pickle_path: str = f"{agents_path}/agent_{agent_id}.pkl"
            agent_obj: agt.Agent
            with open(agent_pickle_path, "rb") as pickle_file:
                agent_obj = pickle.load(pickle_file)

            self.model_agents.append(deepcopy(agent_obj))

            # Manual garbage collection
            del agent_id, agent_pickle_path, agent_obj

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

        # Serialise the created Graph objects so that they remain unchanged across future runs
        self.pickle_graphs()

        print("==== Graph creation finished ====")
        return None

    def pickle_graphs(self) -> None:
        """
        Serialises the tester's initial shared Graph population to a subdirectory within the experiment directory.
        """
        graphs_path: str = f"{ROOT_DIR}/graphs"

        if not os.path.exists(graphs_path):
            os.mkdir(graphs_path)

        for graph in self.model_graphs:
            graph_dir: str = f"{graphs_path}/{graph.name}"
            if not os.path.exists(graph_dir):
                os.mkdir(graph_dir)

            # Write the graphml file for the graph
            graph.save_graph(f"{graph_dir}/graph_{graph.name}.graphml")

            nodes_dir: str = f"{graph_dir}/nodes"
            if not os.path.exists(nodes_dir):
                os.mkdir(nodes_dir)

            for idx, node in enumerate(graph.graph.nodes()):
                node_pickle_path: str = f"{nodes_dir}/node_{idx}.pkl"
                with open(node_pickle_path, "rb") as pickle_file:
                    pickle.dump(node, pickle_file)

            edges_dir: str = f"{graph_dir}/edges"
            if not os.path.exists(edges_dir):
                os.mkdir(edges_dir)

            for idx, edge in enumerate(graph.graph.edges()):
                edge_pickle_path: str = f"{edges_dir}/edge_{idx}.pkl"
                with open(edge_pickle_path, "rb") as pickle_file:
                    pickle.dump(edge, pickle_file)

        return None

    def load_graphs(self) -> None:
        """
        Deserialises the tester's initial shared Graph population and loads it into memory.
        """
        graphs_path: str = f"{ROOT_DIR}/graphs"

        for hierarchy in TEST_PARAMETERS["hierarchy_names"]:
            hierarchy_dir: str = f"{graphs_path}/{hierarchy}"

            new_graph: gr.Graph = gr.Graph("", (0.0, 0.0))
            new_graph.load_graph(
                f"{hierarchy_dir}/graph_{hierarchy}.graphml",
                hierarchy,
                rw_params=TEST_PARAMETERS["hierarchy_rw"][hierarchy],
            )

            nodes_dir: str = f"{hierarchy_dir}/nodes"
            node_paths: list[str] = list(os.walk(nodes_dir))[0][2]

            for node_path in node_paths:
                node_index: int = int(
                    (os.path.basename(node_path).split("_")[-1]).split(".")[0]
                )
                with open(f"{nodes_dir}/{node_path}", "rb") as pickle_file:
                    node_object: gr.GraphNode = pickle.load(pickle_file)
                    new_graph.graph[node_index] = node_object

            edges_dir: str = f"{hierarchy_dir}/edges"
            edge_paths: list[str] = list(os.walk(edges_dir))[0][2]
            for edge_path in edge_paths:
                edge_index: int = int(
                    (os.path.basename(edge_path).split("_")[-1]).split(".")[0]
                )
                with open(f"{edges_dir}/{edge_path}", "rb") as pickle_file:
                    edge_object: gr.GraphEdge = pickle.load(pickle_file)
                    new_graph.graph.update_edge_by_index(edge_index, edge_object)

            self.model_graphs.append(deepcopy(new_graph))
        return None

    def load_models(self, existing_saves: list[str] | None = None) -> None:
        """
        Loads the model objects that have been previously saved in their respective directories.

        :param existing_saves: An optional partial list of the model names representing the models that can be loaded.
        """
        if existing_saves:
            for existing_save in existing_saves:
                # Create an empty dummy model object
                new_model: md.ABModel = md.ABModel(
                    TEST_PARAMETERS["hierarchy_names"], TEST_PARAMETERS["hierarchy_rw"]
                )
                new_model.load_model(SAVEDIRS[existing_save])

                save_struct_path: str = (
                    f"{SAVEDIRS[existing_save]}/{new_model.model_id}.pkl"
                )
                save_struct_dict: dict[str, Any]
                with open(save_struct_path, "rb") as pickle_file:
                    save_struct_dict = pickle.load(pickle_file)

                model_struct: ModelStruct = ModelStruct(
                    deepcopy(new_model),
                    save_struct_dict["max_iterations"],
                    save_struct_dict["change_iteration"],
                    save_struct_dict["changed_agents"],
                )

                self.models.append(deepcopy(model_struct))

                # Manual garbage collection
                del new_model, save_struct_dict, model_struct
            return None

        for model_name, model_savedir in SAVEDIRS.items():
            new_model: md.ABModel = md.ABModel(
                TEST_PARAMETERS["hierarchy_names"],
                TEST_PARAMETERS["hierarchy_rw"],
            )
            new_model.load_model(model_savedir)

            save_struct_path: str = f"{model_savedir}/{model_name}.pkl"
            save_struct_dict: dict[str, Any]
            with open(save_struct_path, "rb") as pickle_file:
                save_struct_dict = pickle.load(pickle_file)

            model_struct: ModelStruct = ModelStruct(
                deepcopy(new_model),
                save_struct_dict["max_iterations"],
                save_struct_dict["change_iteration"],
                save_struct_dict["changed_agents"],
            )

            self.models.append(deepcopy(model_struct))

            # Manual garbage collection
            del new_model, save_struct_dict, model_struct
        return None

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

    def initialise_model_structs(self, missing_saves: list[str] | None = None) -> None:
        """
        Uses the relevant information provided to create appropriate ABModel objects and wrap them in a ModelStruct.

        :param missing_saves: An optional partial list of the model names representing models that should be initialised.
        """
        # Workaround to allow for np random choice
        agent_indices: list[int] = [i for i in range(len(self.model_agents))]

        for interval in self.model_intervals:
            for repeat in range(self.model_repeats):
                # Generate the unique model ID for this instance
                model_name: str = f"ITER-{interval:03}_NUM-{repeat + 1:02}"

                # Only models in missing saves need to be initialised
                if missing_saves:
                    if model_name not in missing_saves:
                        continue

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

    def save_models(self, missing_saves: list[str] | None = None) -> None:
        """
        Saves the model objects along with the information contained in their correpsonnding ModelStruct to allow for future loading.

        :param missing_saves: An optional partial list of model names representing the models that should be saved.
        """
        save_struct: SaveStruct
        if missing_saves is None:
            for model_struct in self.models:
                model_struct.model.save_model()  # Will save the model to a newly created savedir

                # Extract the ModelStruct info (without the ABModel) and immediately pickle it to the Model's newly created save directory
                save_struct = SaveStruct(model_struct, model_struct.model.save_dir)

                # Manual garbage collection
                del save_struct
        else:
            for missing_save in missing_saves:
                struct_to_save: ModelStruct = self.get_struct(missing_save)

                struct_to_save.model.save_model()

                save_struct = SaveStruct(struct_to_save, struct_to_save.model.save_dir)

                # Manual garbage collection
                del save_struct
        return None

    def run_models(self, missing_saves: list[str] | None = None) -> None:
        """
        Runs each model instance in the tester class, calling the custom iteration function, and introducing the
        sudden opinion changes at the correct iteration.

        :param missing_saves: An optional partial list of the model names representing models that should be run.
        """
        print("==== Beginning model iterations ====\n\n")
        if missing_saves:
            for missing_save in missing_saves:
                missing_struct: ModelStruct = self.get_struct(missing_save)
                self.custom_iterate(missing_struct)
            # Only save the models which were missing
            self.save_models(missing_saves=missing_saves)
            return None

        for model_struct in self.models:
            self.custom_iterate(model_struct)
        self.save_models()
        return None

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

    # The root directory of the experiment itself
    ROOT_DIR: str = "./gatoh/experiments/Base/OpinionChanges"

    # The root of the directory in which each instance's save directory will be located
    # (using a /models subdirectory just for this experiment due to significant increase in number of instances)
    SAVEDIR_ROOT: str = "./gatoh/experiments/Base/OpinionChanges/models"

    # A path to which a validation file will be written -- outlining the model name and generated save directory that were generated
    # for each instance during the tester initialisation (to allow for checking of missing saves in the future)
    LOGGED_SAVEDIRS: str = (
        "./gatoh/experiments/Base/OpinionChanges/OpinionChanges_logged_savedirs.csv"
    )

    # A path to which a validation file will be written -- serialising a <model name, [Agent IDs]> mapping that outlines which Agents
    # are having opinion changes introduced for that model
    LOGGED_CHANGED_AGENTS: str = (
        "./gatoh/experiments/Base/OpinionChanges/OpinionChanges_changed_agents.pkl"
    )

    # A <model name, path> mapping of all the model instances that were initially created by the tester
    SAVEDIRS: dict[str, str] = {}
    # A <model name, [Agent IDs]> mapping of all the Agents in which the opinion changes are being introduced
    if os.path.exists(LOGGED_CHANGED_AGENTS):
        with open(LOGGED_CHANGED_AGENTS, "rb") as pickle_file:
            CHANGED_AGENTS: dict[str, dict[str, list[str]]] = pickle.load(pickle_file)

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
        if not os.path.exists(LOGGED_CHANGED_AGENTS):
            # If the changed agents have not been logged, the instances cannot be loaded appropriately...
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
            tester.initialise_model_structs(missing_saves=missing_savedirs)
            tester.run_models(missing_saves=missing_savedirs)
        else:
            tester.initialise_model_structs()
            tester.run_models()
    else:
        tester = OpinionChangesTester(existing=True)
        tester.load_models()

    # TODO: Add the graph visualisation functions here when they have been implemented...
