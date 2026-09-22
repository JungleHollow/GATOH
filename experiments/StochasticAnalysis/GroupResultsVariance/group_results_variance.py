from __future__ import annotations

import csv
import gc
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

    def load_results(self) -> Self:
        """
        Loads results that have been saved following the above format.
        """
        aggregates_path: str = f"{ROOT_DIR}/aggregate_opinions.csv"
        rad_agts_path: str = f"{ROOT_DIR}/radicalised_agents.csv"
        rad_grps_path: str = f"{ROOT_DIR}/radicalised_groups.csv"
        polarisation_path: str = f"{ROOT_DIR}/polarisations.csv"

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

    def calculate_opinions_statistics(self) -> tuple[list[float], list[float]]:
        """
        Calculates the basic statistics of the aggregate opinion across the models.

        :return: The average and standard deviation of the aggregate opinions across the models.
        :rtype: tuple[list[float], list[float]]
        """
        average_opinions: list[float] = []
        opinions_sd: list[float] = []

        for i in range(TEST_PARAMETERS["iterations"]):
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

        for i in range(TEST_PARAMETERS["iterations"]):
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

        for i in range(TEST_PARAMETERS["iterations"]):
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

        for hierarchy in TEST_PARAMETERS["hierarchy_names"]:
            average_polarisation[hierarchy] = []
            polarisation_sd[hierarchy] = []

        for i in range(TEST_PARAMETERS["iterations"]):
            iteration_values: dict[str, list[float]] = {hierarchy: [] for hierarchy in TEST_PARAMETERS["hierarchy_names"]}

            for hierarchy_dict in self.polarisations.values():
                for hierarchy, hierarchy_values in hierarchy_dict.items():
                    iteration_values[hierarchy].append(hierarchy_values[i])

            for hierarchy, values_list in iteration_values.items():
                hierarchy_average: float = np.average(values_list)
                hierarchy_sd: float = float(np.std(values_list))

                average_polarisation[hierarchy].append(hierarchy_average)
                polarisation_sd[hierarchy].append(hierarchy_sd)

        return average_polarisation, polarisation_sd


class VarianceTester:
    """
    The main tester class which will handle the set up, iteration, and persistence of the different models
    that are used in this experiment.

    For this experiment, all models will be identical; the purpose being to inspect the level of variance
    across the output model parameters and emergent behaviour from the stochastic runtime processes.

    A secondary objective being a characterisation of the number of model repetitions that need to be run
    and aggregated before any further reductions to the variance are insignificant.

    This experiment specifically is an extension of the original ResultsVariance, using GATOH's Groups module
    and clustered simulations rather than individual agents; done in order to ensure that the results variance
    within the group simulations are similar to those seen with individual agents.

    :param results_container: The container to which the tester's model results will be stored to.
    :type results_container: AnalysisResults
    :param existing: A flag indicating if the experiment has already been run and saved models are present to inspect.
    :type existing: bool, optional
    """

    def __init__(self, results_container: AnalysisResults, existing: bool = False) -> None:
        self.results: AnalysisResults = results_container
        self.num_agents: int = AGENT_PARAMETERS["n_agents"]

        self.existing: bool = existing
        self.model_saves: dict[str, str] = {}

        # Dynamic model space
        self.models: dict[str, md.ABModel] = {}

        # Define the lists that will contain the populations of Agents, Graphs, and Groups
        self.model_agents: list[agt.Agent] = []
        self.model_graphs: list[gr.Graph] = []
        self.model_groups: list[grp.Group] = []
        self.group_edges: list[tuple[int, int]] = []

        if not self.existing:
            self.create_models()
            self.create_agents()
            self.create_graphs(self.model_agents)
            self.create_groups()
        else:
            self.model_saves = SAVEDIRS
            # load_models is called from __main__ as any missing savefiles are checked for there
            self.load_agents()
            self.load_graphs()
            self.load_groups()

    def create_models(self) -> None:
        """
        Creates the empty model objects that will later be set up and run for the experiment.
        """
        for i in range(TEST_PARAMETERS["repetitions"]):
            model_id: str = f"{TEST_PARAMETERS['model_id_base']}-{i + 1:03}"

            model_savedir: str = f"{SAVEDIR_ROOT}/{model_id}"
            if not os.path.exists(model_savedir):
                os.mkdir(model_savedir)

            model_datafile: str = f"{model_savedir}/{model_id}_variables.csv"

            new_model: md.ABModel = md.ABModel(
                deepcopy(TEST_PARAMETERS["hierarchy_names"]),
                deepcopy(list(TEST_PARAMETERS["hierarchy_rw"].values())),
                save_dir=model_savedir,
                data_file=model_datafile,
                model_id=model_id,
            )
            self.models[model_id] = deepcopy(new_model)

            # Manual garbage collection
            del new_model
            _ = gc.collect()
        return None

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
                AGENT_PARAMETERS["opinions"][0], AGENT_PARAMETERS["opinions"][1],
            )
            agent_personality: str = agt.draw_personality()
            agent_susceptibility: float = rd.uniform(
                AGENT_PARAMETERS["social_susceptibility"][0], AGENT_PARAMETERS["social_susceptibility"][1],
            )
            agent_behaviour: tuple[str, float] = (agent_personality, agent_susceptibility)
            agent_benefit: bool = bool(np.random.choice(benefit_flags, size=1, p=benefit_p)[0])

            hierarchy_weightings: dict[str, float] = {}
            for hierarchy_name in TEST_PARAMETERS["hierarchy_names"]:
                generated_weighting: float = rd.uniform(
                    AGENT_PARAMETERS["hierarchy_weighting"][0], AGENT_PARAMETERS["hierarchy_weighting"][1],
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
            del agent_id, agent_opinion, agent_personality, agent_susceptibility, agent_behaviour, agent_benefit
            del hierarchy_weightings, created_agent
            _ = gc.collect()

        self.model_agents = deepcopy(created_agents)

        # Manual garbage collection
        del created_agents
        _ = gc.collect()

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
            _ = gc.collect()

        return None

    def create_graphs(self, agents: list[agt.Agent]) -> None:
        """
        Generates and sets the shared collection of social hierarchy Graph objects that will be used across the instances.

        :param agents: The population of agents to use for graph creation.
        :type agents: list[:class:`~gatoh.agents.Agent`]
        """
        print("==== Starting Graph creation ====")

        # Workaround to allow for np random choice
        agent_indices: list[int] = [i for i in range(len(agents))]

        for hierarchy in TEST_PARAMETERS["hierarchy_names"]:
            graph: gr.Graph = gr.Graph(
                hierarchy, TEST_PARAMETERS["relationship_rw"], suppress_warnings=True,
            )

            # Ensures that every agent in the population belongs to at least one hierarchy
            if hierarchy == "B":
                _ = graph.generate_graph(
                    deepcopy(agents),
                    method=TEST_PARAMETERS["graph_generation_alg"],
                    relationship_range=AGENT_PARAMETERS["relationships"],
                )
            else:
                hierarchy_n_agents: int = rd.randint(
                    AGENT_PARAMETERS["subset_size_range"][0],
                    AGENT_PARAMETERS["subset_size_range"][1],
                )
                selected_agents: list[int] = list(np.random.choice(agent_indices, size=hierarchy_n_agents, replace=False))

                agent_sample: list[agt.Agent] = []
                for index in selected_agents:
                    agent_sample.append(deepcopy(agents[index]))

                _ = graph.generate_graph(
                    deepcopy(agent_sample),
                    method=TEST_PARAMETERS["graph_generation_alg"],
                    relationship_range=AGENT_PARAMETERS["relationships"],
                )

                # Manual garvage collection
                del hierarchy_n_agents, selected_agents, agent_sample
                _ = gc.collect()

            self.model_graphs.append(deepcopy(graph))

            # Manual garbage collection
            del graph
            _ = gc.collect()

        # Serialise the created Graph objects so that they remain unchanged across future runs
        self.pickle_graphs()

        print("==== Graph creatiron finished ====")
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
                with open(node_pickle_path, "wb") as pickle_file:
                    pickle.dump(node, pickle_file)

            edges_dir: str = f"{graph_dir}/edges"
            if not os.path.exists(edges_dir):
                os.mkdir(edges_dir)

            for idx, edge in enumerate(graph.graph.edges()):
                edge_pickle_path: str = f"{edges_dir}/edge_{idx}.pkl"
                with open(edge_pickle_path, "wb") as pickle_file:
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

            # Manual garbage collection
            del new_graph
            _ = gc.collect()
        return None

    def create_groups(self) -> None:
        """
        Generates and sets the shared population of Group objects that will be used across the instances.
        """
        print("==== Starting Group creation ====")
        created_groups: list[grp.Group] = []
        group_relationships: list[tuple[int, int]] = []

        group_count: int = 0

        for graph in self.model_graphs:
            clustered_nodes: dict[gr.GraphNode, int] = graph.cluster_nodes(k=GROUP_PARAMETERS["n_groups"])

            # Re-organise the nodes in clusters into clustered agents
            group_members: dict[int, list[agt.Agent]] = {}
            for node, cluster in clustered_nodes.items():
                group_members.setdefault(cluster, []).append(node.agent)

            graph_groups: list[grp.Group] = []

            for cluster, members in group_members.items():
                new_group: grp.Group = grp.Group()
                _ = new_group.generate_group(
                    f"{GROUP_PARAMETERS['id_base']}{group_count + 1:04}",
                    cluster,
                    hierarchy=graph.name,
                    members=members,
                )
                new_group.set_index(group_count)
                group_count += 1
                graph_groups.append(new_group)

            created_groups.extend(deepcopy(graph_groups))
            group_relationships.extend(graph.generate_group_edges(graph_groups))

            # Manual garbage collection
            del clustered_nodes, group_members, graph_groups
            _ = gc.collect()

        self.group_edges = deepcopy(group_relationships)
        self.model_groups = deepcopy(created_groups)

        # Manual garbage collection
        del group_relationships, created_groups
        _ = gc.collect()

        # Serialise the created Group objects
        self.pickle_groups()

        print("==== Finished Group creation ====")
        return None

    def pickle_groups(self) -> None:
        """
        Serialise the tester's initial shared Group population to a subdirectory within the experiment directory.
        """
        groups_path: str = f"{ROOT_DIR}/groups"

        if not os.path.exists(groups_path):
            os.mkdir(groups_path)

        for idx, group in enumerate(self.model_groups):
            group_pickle_path: str = f"{groups_path}/group_{idx}.pkl"
            with open(group_pickle_path, "wb") as pickle_file:
                pickle.dump(group, pickle_file)

        # Also pickle the group edges
        with open(f"{groups_path}/group_edges.pkl", "wb") as pickle_file:
            pickle.dump(self.group_edges, pickle_file)

        return None

    def load_groups(self) -> None:
        """
        Deserialises the tester's initial shared Group population and loads them into memory.
        """
        groups_path: str = f"{ROOT_DIR}/groups"

        for i in range(GROUP_PARAMETERS["n_groups"]):
            group_pickle_path: str = f"{groups_path}/group_{i}.pkl"
            group_obj: grp.Group
            with open(group_pickle_path, "rb") as pickle_file:
                group_obj = pickle.load(pickle_file)

            self.model_groups.append(deepcopy(group_obj))

            # Manual garbage collection
            del group_pickle_path, group_obj
            _ = gc.collect()

        # Also load the group edges
        with open(f"{groups_path}/group_edges.pkl", "rb") as pickle_file:
            self.group_edges = pickle.load(pickle_file)

        return None

    def load_models(self, existing_saves: list[str] | None = None) -> None:
        """
        Loads the model objects that have been previously saved in their respective subdirectories.

        :param missing_saves: A potentially partial list of model names indicating models which can be loaded.
        :type missing_saves: list[str], optional
        """
        if existing_saves is not None:
            for existing_save in existing_saves:
                # Create a dummy model for now
                new_model: md.ABModel = md.ABModel(
                    TEST_PARAMETERS["hierarchy_names"],
                    list(TEST_PARAMETERS["hierarchy_rw"].values()),
                )
                new_model.load_model(SAVEDIRS[existing_save])

                self.models[existing_save] = deepcopy(new_model)

                # Manual garbage collection
                del new_model
                _ = gc.collect()
            return None

        for model_name, model_savedir in SAVEDIRS.items():
            new_model: md.ABModel = md.ABModel(
                TEST_PARAMETERS["hierarchy_names"],
                list(TEST_PARAMETERS["hierarchy_rw"].values()),
            )
            new_model.load_model(model_savedir)

            self.models[model_name] = deepcopy(new_model)

            # Manual garbage collection
            del new_model
            _ = gc.collect()
        return None

    def create_savedir_validation(self) -> None:
        """
        Writes a csv file with columns ["model_name", "model_savedir"] containing the relevant information
        for the models of all instances that have been initialised during model setup.

        This is done in order to allow for checking of missing instance save directories if the tester is
        being loaded from an existing run.
        """
        with open(LOGGED_SAVEDIRS, "w", newline="") as csv_file:
            field_names: list[str] = ["model_name", "model_savedir"]

            csv_writer: csv.DictWriter[str] = csv.DictWriter(
                csv_file, fieldnames=field_names,
            )
            csv_writer.writeheader()

            for model in self.models.values():
                csv_row: dict[str, str] = {
                    "model_name": model.model_id,
                    "model_savedir": model.save_dir,
                }
                csv_writer.writerow(csv_row)

        return None

    def setup_models(self, missing_saves: list[str] | None = None) -> None:
        """
        Creates the model objects and adds copies of the shared Agent, Graph, and Group populations to them.

        :param missing_saves: A potentially partial list of model names which must be setup.
        :type missing_saves: list[str], optional
        """
        print("==== Setting up the model instances ====")
        if missing_saves is not None:
            for missing_save in missing_saves:
                _ = self.models[missing_save].add_agents(deepcopy(self.model_agents))
                _ = self.models[missing_save].add_graphs(
                    deepcopy(self.model_graphs),
                    TEST_PARAMETERS["hierarchy_names"],
                    list(TEST_PARAMETERS["hierarchy_rw"].values()),
                )
                _ = self.models[missing_save].add_groups(deepcopy(self.model_groups))

                for edge in self.group_edges:
                    from_group: grp.Group = self.model_groups[edge[0]]
                    to_group: grp.Group = self.model_groups[edge[1]]
                    self.models[missing_save].add_group_graph_edge(from_group, to_group)
            print("==== Finished setting up model instances ====")
            return None

        for model in self.models.values():
            _ = model.add_agents(deepcopy(self.model_agents))
            _ = model.add_graphs(
                deepcopy(self.model_graphs),
                TEST_PARAMETERS["hierarchy_names"],
                list(TEST_PARAMETERS["hierarchy_rw"].values()),
            )
            _ = model.add_groups(deepcopy(self.model_groups))

            for edge in self.group_edges:
                from_group: grp.Group = self.model_groups[edge[0]]
                to_group: grp.Group = self.model_groups[edge[1]]
                model.add_group_graph_edge(from_group, to_group)
        print("==== Finished setting up model instances ====")
        return None

    def store_model_results(self, model: md.ABModel) -> None:
        """
        A helper class that takes a model which has finished iterating and calls AnalysisResults.init_model
        appropriately.

        :param model: The model object that is being initialised in the analysis results.
        :type model: :class:`~gatoh.model.ABModel`
        """
        self.results.init_model(
            model.model_id,
            model.logger.variables.aggregate_opinions,
            model.logger.variables.radicalised_agents,
            model.logger.variables.radicalised_groups,
            model.logger.variables.layer_polarisations,
        )
        return None

    def save_results(self) -> None:
        """
        Stores the parameters stored within the AnalysisResults into separate .csv files for each parameter.
        """
        self.create_savedir_validation()
        self.results.save_results()
        return None

    def load_results(self) -> None:
        """
        Loads existing AnalysisResults from their corresponding .csv files.
        """
        self.results = self.results.load_results()
        return None

    def calculate_results_statistics(self) -> OutputDict:
        """
        Calculate and return all of the AnalysisResults statistics.

        :return: A <parameter name : statistics> mapping containing the means and standard deviations for all parameters.
        :rtype: dict[str, Any]
        """
        opinion_statistics: tuple[list[float], list[float]] = self.results.calculate_opinions_statistics()
        radical_agt_statistics: tuple[list[float], list[float]] = self.results.calculate_rad_agt_statistics()
        radical_grp_statistics: tuple[list[float], list[float]] = self.results.calculate_rad_grp_statistics()
        polarisation_statistics: tuple[dict[str, list[float]], dict[str, list[float]]] = self.results.calculate_polarisation_statistics()

        output_dict: OutputDict = {
            "opinion_statistics": opinion_statistics,
            "rad_agts_statistics": radical_agt_statistics,
            "rad_grps_statistics": radical_grp_statistics,
            "polarisation_statistics": polarisation_statistics,
        }

        return output_dict

    def run_models(self, missing_saves: list[str] | None = None, worker_pool: WorkerPool | None = None) -> None:
        """
        Runs each model instance in the tester class.

        :param missing_saves: A potentially partial list of model names which must be run.
        :type missing_saves: list[str], optional
        :param worker_pool: A pool of workers that can distribute the iteration processsing amongst themselves.
        :type worker_pool: :class:`~multiprocessing.pool.Pool`, optional
        """
        if missing_saves is not None:
            for missing_save in missing_saves:
                self.models[missing_save].iterate(worker_pool=worker_pool)
                self.models[missing_save].save_model()
                self.store_model_results(self.models[missing_save])
            return None

        for model in self.models.values():
            model.iterate(worker_pool=worker_pool)
            model.save_model()
            self.store_model_results(model)

        return None


def plot_var_over_models(analysis_results: AnalysisResults) -> None:
    """
    Plots the variance of the opinions across the models.

    :param analysis_results: The result data from the experiment.
    :type analysis_results: AnalysisResults
    """
    current_opinion_values: list[float] = []

    y_values: list[float] = []  # Variance of n models' values
    x_values: list[int] = []  # Number of models

    for idx, values in enumerate(analysis_results.aggregate_opinions.values()):
        current_opinion_values += deepcopy(values)

        current_vals_variance: float = float(np.var(current_opinion_values))

        y_values.append(current_vals_variance)
        x_values.append(idx + 1)

    fig, ax = plt.subplots()

    _ = ax.plot(x_values, y_values, "--k")
    _ = ax.set_xlabel("Number of Models")
    _ = ax.set_ylabel("Aggregate Opinion Variance")
    _ = ax.set_title("Variance of Aggregate Opinions by Number of Models Used")

    save_path: str = f"{ROOT_DIR}/GroupVarianceOverModels.png"

    plt.savefig(save_path, dpi=300.0)

    return None


def plot_model_runtimes(analysis_results: AnalysisResults, analysis_statistics: OutputDict) -> None:
    """
    Plot the runtime opinion values for all models in the experiment, along with an overall average trend.

    :param analysis_results: The results data from the experiment.
    :type analysis_results: AnalysisResults
    :param analysis_statistics: A dictionary containing the per-iteration means and standard deviations of the model parameters.
    :type analysis_statistics: dict[str, Any]
    """
    iterations: list[int] = [i + 1 for i in range(TEST_PARAMETERS["iterations"])]
    fig, ax = plt.subplots()

    for _, values in analysis_results.aggregate_opinions.items():
        _ = ax.plot(iterations, values, "-k", linewidth=0.7, alpha=0.25)

    _ = ax.plot(
        iterations,
        analysis_statistics["opinion_statistics"][0]m
        "-r",
        linewidth=0.8,
        alpha=1.0,
        label="Average",
    )

    _ = ax.legend()
    _ = ax.set_xlabel("Iterations")
    _ = ax.set_ylabel("Aggregate Opinion")
    _ = ax.set_title("Model Aggregate Opinions over Iterations")

    save_path: str = f"{ROOT_DIR}/GroupModelRuntimes.png"

    plt.savefig(save_path, dpi=300.0)

    return None


def plot_parameter_whiskers(analysis_statistics: OutputDict) -> None:
    """
    Plot box and whiskers for each of the model parameters.

    :param analysis_statistics: A dictionary containing the per-iteration means and standard deviations for all parameters.
    :type analysis_statistics: dict[str, Any]
    """
    unpacked_results: list[list[float | int]] = []
    tick_labels: list[str] = []

    for hierarchy, statistics in analysis_statistics["polarisation_statistics"][0].items():
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

    save_path: str = f"{ROOT_DIR}/GroupPolarisationStatistics.png"

    plt.savefig(save_path, dpi=300.0)

    unpacked_opinion: list[float] = deepcopy(analysis_statistics["opinion_statistics"][0])
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

    save_path = f"{ROOT_DIR}/GroupOpinionStatistics.png"

    plt.savefig(save_path, dpi=300.0)

    unpacked_rad_agt: list[float] = deepcopy(analysis_statistics["rad_agts_statistics"][0])
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

    save_path = f"{ROOT_DIR}/GroupRadicalAgentStatistics.png"

    plt.savefig(save_path, dpi=300.0)

    unpacked_rad_grp: list[float] = deepcopy(analysis_statistics["rad_grps_statistics"][0])
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

    save_path = f"{ROOT_DIR}/GroupRadicalGroupStatistics.png"

    plt.savefig(save_path, dpi=300.0)

    return None


def create_analysis_plots(analysis_results: AnalysisResults, analysis_statistics: OutputDict) -> None:
    """
    Create all relevant analysis plots and then store them to the experiment's save directory.

    :param analysis_results: The result data from the experiment.
    :type analysis_results: AnalysisResults
    :param analysis_statistics: A dictionary containing per-iteration means and standard deviations for each parameter.
    :type analysis_statistics: dict[str, Any]
    """
    plot_var_over_models(analysis_results)
    plot_model_runtimes(analysis_results, analysis_statistics)
    plot_parameter_whiskers(analysis_statistics)
    return None


if __name__ == "__main__":
    MULTIPROCESSING: bool = True
    WORKER_POOL: WorkerPool | None = Pool() if MULTIPROCESSING else None

    class TestParameters(TypedDict):
        """
        A helper class used for typechecking of TEST_PARAMETERS.
        """
        iterations: int
        repetitions: int
        hierarchy_names: list[str]
        hierarchy_rw: dict[str, tuple[float, float]]
        relationship_rw: tuple[float, float]
        graph_generation_alg: str
        model_id_base: str

    # The relevant parameters that are defined for the identical model instances
    TEST_PARAMETERS: TestParameters = {
        "iterations": 100,
        "repetitions": 100,
        "hierarchy_names": ["A", "B", "C"],
        "hierarchy_rw": {
            "A": (0.0, 0.3),
            "B": (0.0, 0.1),
            "C": (0.0, 0.45),
        },
        "relationship_rw": (0.0, 0.1),
        "graph_generation_alg": "small-world",
        "model_id_base": "GSARV-Model",
    }

    class AgentParameters(TypedDict):
        """
        A helper class used for typechecking of AGENT_PARAMETERS.
        """
        n_agents: int
        subset_size_range: tuple[int, int]
        opinions: tuple[float, float]
        relationships: tuple[float, float]
        hierarchy_weighting: tuple[float, float]
        personal_benefit: dict[bool, float]
        social_susceptibility: tuple[float, float]
        id_base: str

    # The parameters that will be used to create the Agent population that is shared across models
    AGENT_PARAMETERS: AgentParameters = {
        "n_agents": 100,
        "subset_size_range": (25, 50),
        "opinions": (-1.0, 1.0),
        "relationships": (-1.0, 1.0),
        "hierarchy_weighting": (-1.0, 1.0),
        "personal_benefit": {True: 0.25, False: 0.75},
        "social_susceptibility": (0.0, 1.0),
        "id_base": "GSARVA",  # (Group Stochastic Analysis Results Variance Agent)
    }

    class GroupParameters(TypedDict):
        """
        A helper class used for typechecking of GROUP_PARAMETERS.
        """
        n_groups: int
        id_base: str

    # The parameters that will be used to create the Group population that is shared across models
    GROUP_PARAMETERS: GroupParameters = {
        "n_groups": 5,
        "id_base": "GSARVG",  # (Group Stochastic Analysis Results Variance Group)
    }

    # The root directory of the experiment itself
    ROOT_DIR: str = "./experiments/StochasticAnalysis/GroupResultsVariance"

    # The root directory in which each instance's save directory will be located
    # (using a /models subdirectory for this experiment due to large number of instances)
    SAVEDIR_ROOT: str = f"{ROOT_DIR}/models"

    # A path to which a validation file will be written -- outlining the model name and save directory that were generated
    # for each instance during the tester initialisation (to allow for reduced, partial runtimes in the future if some files are missing)
    LOGGED_SAVEDIRS: str = f"{ROOT_DIR}/GroupResultsVariance_logged_savedirs.csv"

    # A <model_name : path> mapping of all the model instances that were initially created by the tester
    SAVEDIRS: dict[str, str] = {}

    results: AnalysisResults
    tester: VarianceTester

    # Ensure the subdirectory exists
    if not os.path.exists(SAVEDIR_ROOT):
        os.mkdir(SAVEDIR_ROOT)

    # Check for existing saved models and store the relevant information
    save_dirs: list[str] = list(os.walk(SAVEDIR_ROOT))[0][1]

    directory_missing: bool = False
    existing_savedirs: list[str] = []
    missing_savedirs: list[str] = []

    # The tester has not yet been run or the validation file was removed
    if not os.path.exists(LOGGED_SAVEDIRS):
        directory_missing = True
    else:
        with open(LOGGED_SAVEDIRS, "r", newline="") as csv_file:
            csv_reader: csv.DictReader[str] = csv.DictReader(csv_file)
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
        print("One or more model subdirectories are missing; initialising and running the missing models...")
        results = AnalysisResults()
        tester = VarianceTester(results)

        # At least one model exists
        if len(existing_savedirs) > 0:
            tester.load_models(existing_saves=existing_savedirs)
            tester.setup_models(missing_saves=missing_savedirs)
            tester.run_models(missing_saves=missing_savedirs, worker_pool=WORKER_POOL)
            tester.save_results()
        else:
            tester.setup_models()
            tester.run_models(worker_pool=WORKER_POOL)
            tester.save_results()
    else:
        print("Loading a saved, previously run instance of the Group ResultsVariance experiment...")
        results = AnalysisResults()
        tester = VarianceTester(results, existing=True)
        tester.load_models()
        tester.load_results()

        # Calculate the means and standard deviations for all relevant model parameters
        analysis_statistics: OutputDict = tester.calculate_results_statistics()

        # Create all relevant output plots for this experiment
        create_analysis_plots(tester.results, analysis_statistics)

    # Ensure the worker pool is terminated if it exists once all processing is finished
    if WORKER_POOL is not None:
        WORKER_POOL.terminate()
