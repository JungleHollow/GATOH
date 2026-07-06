from __future__ import annotations

import os
from copy import deepcopy
from datetime import datetime
from random import choices, randint
from shutil import rmtree
from typing import Any

import numpy as np
import yaml
from matplotlib import pyplot as plt

# Used for type declarations in ABModel __init__
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rustworkx.rustworkx import NoEdgeBetweenNodes

from gatoh.agents.agents import Agent, AgentSet
from gatoh.graphs.graphs import Graph, GraphEdge, GraphSet
from gatoh.logging.logging import GATOHLogger
from gatoh.utils.utils import (
    EdgeChanges,
    NodeChanges,
    YamlLoader,
    create_config_file,
)
from gatoh.visualisation.visualisation import ABVisualiser


class ABModel:
    """
    An agent-based model class that is capable of handling multiple layers that affect agent behaviour.

    :param hierarchy_names: The names of all social hierarchies that will exist in the model.
    :type hierarchy_names: list[str]
    :param hierarchy_rw_distributions: (mean, variance) parameters used for random walk effects within each hierarchy.
    :type hierarchy_rw_distributions: list[tuple[float, float]]
    :param agent_opinion_rw: Shared (mean, variance) parameters used for stochastic opinion changes across all agents at each timestep.
    :type agent_opinion_rw: tuple[float, float], optional
    :param iterations: The number of iterations that the model will run for.
    :type iterations: int, optional
    :param silencing_threshold: A threshold that, when surpassed by agents, will cause them to cease expressing their opinions in a given hierarchy.
    :type silencing_threshold: float, optional
    :param negation_threshold: A threshold that, when surpassed by agents, will cause their opinion to become its additive inverse.
    :type negation_threshold: float, optional
    :param radicalisation_threshold: A threshold that determines how strong of an absolute opinion an agent must hold before they begin to consider becoming radicalised.
    :type radicalisation_threshold: float, optional
    :param suppress_warnings: A flag indicating if non-critical runtime warnings should be suppressed.
    :type suppress_warnings: bool, optional
    :param visualise: A flag indicating if the model should visualise emergent behaviour in real time.
    :type visualise: bool, optional
    :param visualisation_dir: The path to a directory in which all of this model's visualiser outputs should be saved to.
    :type visualisation_dir: str, optional
    :param vis_aggregation_method: The aggregation method that should be used when relevant for visualisation (i.e. ``median'', ``mean'', etc.).
    :type vis_aggregation_method: str, optional
    :param checkpointing: A flag indicating if the model's progress should be saved at the end of each iteration (useful in case of interrupted runtimes).
    :type checkpointing: bool, optional
    :param save_dir: The path to a directory in which all of this model's non-logger data should be saved to.
    :type save_dir: str, optional
    :param data_file: The path to which the logger's data should be saved to after iterations are run.
    :type data_file: str, optional
    :param model_id: A unique ID assigned to this model to make it referencable amongst other models.
    :type model_id: str, optional
    """

    def __init__(
        self,
        hierarchy_names: list[str],
        hierarchy_rw_distributions: list[tuple[float, float]],
        agent_opinion_rw: tuple[float, float] = (0.0, 0.1),
        iterations: int = 100,
        silencing_threshold: float = 0.95,
        negation_threshold: float = 0.999,
        radicalisation_threshold: float = 0.99,
        suppress_warnings: bool = False,
        visualise: bool = True,
        visualisation_dir: str = "",
        vis_aggregation_method: str = "median",
        checkpointing: bool = True,
        save_dir: str = "",
        data_file: str = "",
        model_id: str = "",
    ) -> None:
        self.hierarchy_information: dict[str, tuple[float, float]] = {}
        for idx, hierarchy in enumerate(hierarchy_names):
            self.hierarchy_information[hierarchy] = hierarchy_rw_distributions[idx]

        self.agent_opinion_rw: tuple[float, float] = agent_opinion_rw

        self.agents: AgentSet = AgentSet(self)
        self.graphs: GraphSet = GraphSet(self)

        # A model-handled 'base' Graph that keeps track of all relationships across the social hierarchies
        # (Used to greatly simplify network-level graph calculations)
        self.base_graph: Graph = Graph("base", (0.0, 0.0))

        self.logger: GATOHLogger = GATOHLogger(self, iterations, hierarchy_names)

        self.visualise: bool = visualise
        self.visualisation_dir: str = visualisation_dir
        self.visualiser: ABVisualiser
        self.fig: Figure
        self.ax: Axes

        # Only create the visualisation objects if visualisation is required
        if self.visualise:
            self.visualiser = ABVisualiser(
                self, self.visualisation_dir, aggregation_method=vis_aggregation_method
            )
            self.fig, self.ax = plt.subplots()

        self.current_iteration: int = 0
        self.max_iterations: int = iterations

        self.silencing_threshold: float = silencing_threshold
        self.negation_threshold: float = negation_threshold
        self.radicalisation_threshold: float = radicalisation_threshold

        self.suppress_warnings: bool = suppress_warnings

        self.checkpointing: bool = checkpointing
        self.save_dir: str = save_dir
        self.data_file: str = data_file
        self.model_id: str = model_id

    def save_model(self) -> None:
        """
        Saves the model's GraphSet and AgentSet objects, including all objects recursively contained.
        """
        if self.save_dir == "":
            # An empty save directory is assumed to mean that no saving is desired.
            # This will automatically override a checkpointing = True flag.
            return None

        # In the case when attempting to save the model, an existing directory with the specified name will be deleted and newly created.
        if os.path.isdir(self.save_dir):
            rmtree(self.save_dir)

        # Create the save directory
        os.mkdir(self.save_dir)

        # Save the AgentSet and GraphSet
        self.agents.save_agentset(self.save_dir)
        self.graphs.save_graphset(self.save_dir)

        # Save the model's base graph (uncompressed and in the root of the save directory)
        base_graph_path: str = f"{self.save_dir}/graph_base_graph.graphml"
        self.base_graph.save_graph(base_graph_path)

        # Store the model's configurations in a YAML config file
        config_path: str
        if self.model_id != "":
            config_path = f"{self.save_dir}/model_{self.model_id}.yaml"
        else:
            config_path = f"{self.save_dir}/model_{datetime.now().strftime('%y-%m-%d %H-%M')}.yaml"
        config_data: dict[str, Any] = {
            "hierarchy_information": self.hierarchy_information,
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "silencing_threshold": self.silencing_threshold,
            "negation_threshold": self.negation_threshold,
            "radicalisation_threshold": self.radicalisation_threshold,
            "visualise": self.visualise,
            "visualisation_dir": self.visualisation_dir,
            "suppress_warnings": self.suppress_warnings,
            "checkpointing": self.checkpointing,
            "save_dir": self.save_dir,
            "data_file": self.data_file,
            "model_id": self.model_id,
        }
        create_config_file(config_path, config_data)
        return None

    def load_model(self, load_dir: str) -> None:
        """
        Loads a model and all its components which have been saved following the processes in the save_model() function.

        :param load_dir: The path to the directory where all model data was saved.
        :type load_dir: str
        :raises FileNotFoundError: If the input load_dir is invalid.
        """
        if not os.path.isdir(load_dir):
            raise FileNotFoundError(f"The input load directory {load_dir} is invalid.")

        graphset_exists: bool = False
        agentset_exists: bool = False

        # Recursively scan the files in load_dir
        for file_name in os.listdir(load_dir):
            file_path: str = f"{load_dir}/{file_name}"
            file_type: str = deepcopy(file_name).split(".")[-1]
            match file_type:
                case "zip":
                    # Simply flag that any zip files exist, as these will then be decompressed and loaded by their respective parent modules
                    if file_name == "_agentset.zip":
                        agentset_exists = True
                    elif file_name == "_graphset.zip":
                        graphset_exists = True
                    else:
                        # Currently unknown how/if to handle edge cases here
                        pass
                case "yaml":
                    # The 'yaml' case may grow to include more config files beyond just the model's...
                    config_prefix: str = file_name.split("_")[0]
                    if config_prefix == "model":
                        with open(file_path, "r") as config_file:
                            config_data: dict[str, Any] = yaml.load(
                                config_file, Loader=YamlLoader
                            )
                            self.hierarchy_information = config_data[
                                "hierarchy_information"
                            ]
                            self.current_iteration = config_data["current_iteration"]
                            self.max_iterations = config_data["max_iterations"]
                            self.silencing_threshold = config_data[
                                "silencing_threshold"
                            ]
                            self.negation_threshold = config_data["negation_threshold"]
                            self.radicalisation_threshold = config_data[
                                "radicalisation_threshold"
                            ]
                            self.visualise = config_data["visualise"]
                            self.visualisation_dir = config_data["visualisation_dir"]
                            self.suppress_warnings = config_data["suppress_warnings"]
                            self.checkpointing = config_data["checkpointing"]
                            self.save_dir = config_data["save_dir"]
                            self.data_file = config_data["data_file"]
                            self.model_id = config_data["model_id"]
                case "graphml":
                    # For now, only the model's base graph should exist as an uncompressed graphml file in the root of the save directory
                    self.base_graph.load_graph(file_path, "base_graph", (0.0, 0.0))
                case _:
                    # Currently unknown how/if to handle edge cases here
                    pass

        # Check if any compressed files exist and handle them from their parent modules
        if agentset_exists:
            self.agents.load_agentset(load_dir)
        if graphset_exists:
            self.graphs.load_graphset(load_dir, self.hierarchy_information)

        return None

    def add_graph(self, graph: Graph) -> GraphSet:
        """
        Add a new Graph to the Model's GraphSet. It is assumed that this is a generated Graph object which already has
        a name and rw_params assigned to it.

        :param graph: The graph to be added to the Model's GraphSet.
        :type graph: Graph
        :return: The model's newly updated graph set.
        :rtype: GraphSet
        """
        self.graphs.add_graph(graph)

        # Also add new edges to the model's base graph
        self.add_base_graph_edges(graph)
        return self.graphs

    def add_graphs(
        self,
        graphs: list[Graph | str],
        names: list[str],
        rw_params: list[tuple[float, float]],
    ) -> GraphSet:
        """
        Add new Graphs to the Model's GraphSet.

        :param graphs: Graph objects or filepaths to stored GraphML objects.
        :type graphs: list[Graph | str]
        :param names: The corresponding social hierarchy names to give to the Graphs.
        :type names: list[str]
        :param rw_params: The (mean, variance) to assign to the hierarchy when determining normal distributions for random walk dynamic relationships.
        :type rw_params: list[tuple[float, float]]
        :return: The model's newly updated graph set.
        :rtype: GraphSet
        """
        for idx, graph in enumerate(graphs):
            if type(graph) is Graph:
                self.graphs.add_graph(graph)
            elif type(graph) is str:
                new_graph: Graph = Graph(names[idx], rw_params[idx])
                new_graph.load_graph(graph, names[idx])
                self.graphs.add_graph(new_graph)
        self.init_base_graph()
        return self.graphs

    def generate_graphs(
        self,
        hierarchies: list[str],
        agents: list[Agent] | AgentSet,
        method: str = "small-world",
        agent_subsetting: bool = False,
        rw_params: list[tuple[float, float]] | None = None,
        individual_methods: dict[str, str] | None = None,
    ) -> None:
        """
        Randomly generates graphs for the given social hierarchy names using the specified method.
        Hierarchies will only contain the agents whose names are passed to the function.

        :param hierarchies: The names of the social hierarchy graphs to be created.
        :type hierarchies: list[str]
        :param agents: The agents to be included in the hierarchies.
        :type agents: list[Agent] | AgentSet
        :param method: The social network graph generation method to use. Options include: 'small-world', 'scale-free', 'random', 'blockmodel'. Defaults to 'small-world'.
        :type method: str, optional
        :param agent_subsetting: A flag indicating if the agents should be sampled into random subsets when generating each graph.
        :type agent_subsetting: bool, optional
        :param rw_params: (mean, variance) parameters containing the random-walk distributions for each of the generated graphs.
        :type rw_params: list[tuple[float, float]], optional
        :param individual_methods: A <hierarchy : generation method> mapping indicating the per-hierarchy generation methods that should be used.
        :type individual_methods: dict[str, str], optional
        """
        agent_array: np.ndarray = np.array(agents)
        agent_sample: list[Agent] = []

        for idx, hierarchy in enumerate(hierarchies):
            if agent_subsetting:
                random_k: int = randint(len(agents) // 4, len(agents))
                if type(agents) is list:
                    agent_sample = list(
                        np.random.choice(agent_array, size=random_k, replace=False)
                    )
                elif type(agents) is AgentSet:
                    agent_sample = agents.sample(random_k)

            hierarchy_rw_param: tuple[float, float] = (0.0, 0.1)
            if rw_params:
                hierarchy_rw_param = rw_params[idx]

            hierarchy_graph: Graph = Graph(
                hierarchy, hierarchy_rw_param, suppress_warnings=self.suppress_warnings
            )

            if individual_methods is not None:
                hierarchy_graph = hierarchy_graph.generate_graph(
                    agent_sample, method=individual_methods[hierarchy]
                )
            else:
                hierarchy_graph = hierarchy_graph.generate_graph(
                    agent_sample, method=method
                )

            _ = self.add_graph(hierarchy_graph)
        return None

    def add_agent(self, agent: Agent) -> int:
        """
        Add a single new Agent to the model's AgentSet, returning its index within the AgentSet.

        :param agent: The agent to add to the AgentSet.
        :type agent: Agent
        :return: The index of the newly added Agent in the AgentSet.
        :rtype: int
        """
        # Add the Agent object to the model-handled 'base' graph
        self.base_graph.add_nodes([agent])
        return self.agents.add(agent)

    def add_agents(self, agents: list[Agent]) -> AgentSet:
        """
        Add new Agents to the Model's AgentSet.

        :param agents: The agents to be added to the AgentSet.
        :type agents: list[Agent]
        :return: The model's newly updated agent set.
        :rtype: AgentSet
        """
        for agent in agents:
            _ = self.agents.add(agent)

        # Add all the new Agent objects to the model-handled 'base' graph
        self.base_graph.add_nodes(agents)
        return self.agents

    def generate_agents(
        self,
        id_base: str,
        personality_probs: dict[str, float],
        distribution: str = "gaussian",
        parameters: dict[str, Any] | None = None,
        number: int = 100,
    ) -> None:
        """
        Randomly generates a number of Agent objects.

        :param id_base: a 4-character alphabetic string that serves as the base of the XXXXnnnn id for each Agent.
        :type id_base: str
        :param personality_probs: A <personality : probability> mapping specifying the probability of an Agent having any given personality.
        :type personality_probs: dict[str, float]
        :param distribution: The distribution from which any random values will be drawn.
        :type distribution: str, optional
        :param parameters: Any explicit parameters that the distribution should use when being created.
        :type parameters: dict, optional
        :param number: Number of agents to be randomly created.
        :type number: int, optional
        """
        # Convert to separate lists for use in random.choices()
        personalities: list[str] = list(personality_probs.keys())
        probabilities: list[float] = list(personality_probs.values())

        # Extract the hierarchy names from the information dictionary
        hierarchies: list[str] = list(self.hierarchy_information.keys())

        for i in range(number):
            new_agent: Agent = Agent()
            agent_id: str = f"{id_base}{i:04}"
            agent_index: int = self.add_agent(new_agent)
            agent_personality: str = choices(personalities, weights=probabilities, k=1)[
                0
            ]
            _ = new_agent.generate_agent(
                agent_id,
                agent_index,
                hierarchies,
                distribution=distribution,
                personality=agent_personality,
                parameters=parameters,
            )
        return None

    def iterate(self) -> None:
        """
        Handles the main model iteration loop
        """
        while self.current_iteration < self.max_iterations:
            # Initialise the logger state for the current iteration
            if self.current_iteration == 0:
                self.logger.new_iteration(init=True)
            else:
                self.logger.new_iteration()

            # Initialise a dictionary to keep track of agent opinion changes
            # (this is done to prevent recursive updating of opinions during the evolution of opinions)
            new_agent_opinions: dict[str, tuple[float, list[float], list[bool]]] = {}

            # First each agent looks at its neighbours to see how their opinion will evolve this iterations
            for agent in self.agents:
                agent.previous_opinion = agent.opinion
                for hierarchy in self.graphs:
                    # Update the previous opinion across all hierarchies
                    hierarchy.agent_previous_opinion(agent)

                # TODO: Implement a check and handling for if the current agent is radicalised
                collective_changes: list[float] = []
                for hierarchy in self.graphs:
                    neighbour_influences: float | None = hierarchy.neighbour_influences(
                        agent
                    )
                    if neighbour_influences is not None:
                        collective_changes.append(neighbour_influences)
                total_change: float = sum(collective_changes)

                # Check for the existence of personal benefit across all of the agent's neighbours
                all_neighbour_indices: list[int] = list(
                    self.base_graph.graph.neighbors(agent.index)
                )
                all_neighbour_benefits: list[bool] = []
                for neighbour_index in all_neighbour_indices:
                    neighbour_object: Agent = self.base_graph.graph[neighbour_index]
                    all_neighbour_benefits.append(neighbour_object.personal_benefit)

                # Constrain to [-1, 1]
                # 100.0 and -100.0 are used as key delta values indicating that the opinion needs to be constrained
                if agent.opinion + total_change < -1.0:
                    new_agent_opinions[agent.id] = (
                        -100.0,
                        deepcopy(collective_changes),
                        deepcopy(all_neighbour_benefits),
                    )
                elif agent.opinion + total_change > 1.0:
                    new_agent_opinions[agent.id] = (
                        100.0,
                        deepcopy(collective_changes),
                        deepcopy(all_neighbour_benefits),
                    )
                else:
                    new_agent_opinions[agent.id] = (
                        total_change,
                        deepcopy(collective_changes),
                        deepcopy(all_neighbour_benefits),
                    )
            self.iteration_opinion_changes(new_agent_opinions)
            self.step()
            self.update()

            self.logger_iteration()  # Handle the logger's iteration() calculations and call its method

            # Get this iteration's print string (will be formatted appropriately based on the print interval)
            iteration_print_string: str = self.logger.iteration_print()
            print(iteration_print_string)

            if self.visualise:
                self.visualiser.visualiser_iteration()
            if self.checkpointing:
                self.save_model()

            self.current_iteration += 1
        # Call the logger's save_data function which handles data persistence appropriately
        data_saved: bool = self.logger.save_data(self.data_file)
        if data_saved:
            print(
                f"\n\nGATOH logger data was successfully written to the file at path: {self.data_file}\n\n"
            )
        if self.visualise:
            # Make sure that the pyplot figure is closed after iterations to prevent excess memory usage
            plt.close(self.fig)
            del self.fig, self.ax
        return None

    def iteration_opinion_changes(
        self, changes_dict: dict[str, tuple[float, list[float], list[bool]]]
    ) -> None:
        """
        A helper function for iterate that simply applies all agent opinion changes and then checks
        for radicalisation.

        :param changes_dict: A <agent ID : opinion change information> mapping of the opinion values to apply.
        :type changes_dict: dict[str, tuple[float, list[float], list[bool]]]
        """
        for agent_id, opinion_change_info in changes_dict.items():
            agent_object: Agent | None = self.agents.get_agent_by_id(agent_id)
            if agent_object is not None:
                for hierarchy in self.graphs:
                    # Update the current opinion across all hierarchies
                    hierarchy.agent_opinion_change(agent_object, opinion_change_info[0])

                # After the opinion change, determine if the agent has become radicalised
                was_radicalised: bool = agent_object.radicalisation(
                    opinion_change_info[1],
                    opinion_change_info[2],
                    list(self.hierarchy_information.keys()),
                    self.radicalisation_threshold,
                )

                for hierarchy in self.graphs:
                    # Update the radicalisation status of the agent across all hierarchies
                    hierarchy.agent_radicalisation_change(agent_object, was_radicalised)

                # Update the nodes in the base graph
                self.base_graph.agent_opinion_change(
                    agent_object, opinion_change_info[0]
                )
                self.base_graph.agent_radicalisation_change(
                    agent_object, was_radicalised
                )

                # Update the radicalisation count in the logger as needed
                self.logger.variables.increment_radicalised(was_radicalised)
        return None

    def step(self) -> None:
        """
        Steps the model forward one iteration.

        This does not handle agent opinion changes, but rather dynamic agent relationships
        and hierarchy weightings.
        """
        for graph in self.graphs:
            graph.step()
        for agent in self.agents:
            agent.step(
                self.hierarchy_information,
                self.agent_opinion_rw,
            )
        return None

    def update(self) -> None:
        """
        Updates the agents' internal states to match the model step. This mainly handles the construction of agents'
        perceived opinion climates within their hierarchies, and the simulation of opinion silencing behaviours depending
        on these climates.
        """
        for agent in self.agents:
            silenced: dict[str, bool] = {}
            was_silenced: bool = False
            negation: bool = False
            for graph in self.graphs:
                if not graph.agent_in_graph(agent):
                    # The Agent does not have membership in a specific hierarchy
                    continue
                else:
                    est_opinion_climate: float = graph.estimate_opinion_climate(agent)
                    is_silenced: tuple[bool, float] = agent.opinion_silencing(
                        graph.name, est_opinion_climate
                    )
                    silenced[graph.name] = is_silenced[0]

                    if is_silenced[0]:
                        was_silenced = True

                    if not negation:
                        negation = agent.opinion_negation(
                            graph.name, is_silenced[1], self.negation_threshold
                        )

            # Update the logger variables as needed
            self.logger.variables.increment_silenced(was_silenced)
            self.logger.variables.increment_negated(negation)

            # Update the Agent object
            agent.update(silenced, negation)
        return None

    def logger_iteration(self) -> None:
        """
        Calculate any relevant aggregate statistics and then pass these to the logger's iteration() function to be stored.

        Statistics calculated currently:
            1. Aggregate network opinion
            2. Network radicalisation log odds
            3. Layer navigability for each hierarchy
            4. Layer interdependence for each hierarchy
        """
        aggregate_opinion: float = self.calculate_aggregate_opinion()
        radicalisation_logodds: float = self.calculate_radicalisation_logodds()
        layer_interdependences: dict[str, float] = {}
        for hierarchy in self.hierarchy_information.keys():
            layer_interdependences[hierarchy] = self.calculate_interdependence(
                self.graphs.get_index(hierarchy)
            )

        layers_polarisation: dict[str, float] = self.calculate_layers_polarisation()

        self.logger.iteration(
            aggregate_opinion,
            radicalisation_logodds,
            layer_interdependences,
            layers_polarisation,
        )
        return None

    def calculate_aggregate_opinion(self) -> float:
        """
        Calculates the aggregate network opinion by iterating over each Agent in the model.

        :return: The aggregate network opinion value.
        :rtype: float
        """
        all_opinions: list[float] = []
        for agent in self.agents:
            all_opinions.append(agent.opinion)

        opinion_sum: float = sum(all_opinions)
        average_opinion: float = opinion_sum / len(all_opinions)

        return average_opinion

    def calculate_radicalisation_logodds(self) -> float:
        """
        Calculates the log odds of an Agent being radicalised within the model.

        :return: The log odds of agent radicalisation.
        :rtype: float
        """
        radicalised_count: int = 0
        for agent in self.agents:
            if agent.radicalised:
                radicalised_count += 1

        radicalisation_p: float = radicalised_count / len(self.agents)
        if 1.0 - radicalisation_p != 0.0:
            log_odds: float = np.log1p(radicalisation_p / (1.0 - radicalisation_p))
            return log_odds
        return 0.0

    def calculate_layers_polarisation(self) -> dict[str, float]:
        r"""
        Calculate the polarisation of the opinion climate within each hierarchy by calling each graph's calculate_polarisation() method.

        :return: A <hierarchy : value> mapping containing the polarisation value for each hierarchy.
        :rtype: dict[str, float]
        """
        layers_polarisation: dict[str, float] = {}

        for hierarchy in self.hierarchy_information.keys():
            layers_polarisation[hierarchy] = self.graphs.calculate_polarisation(
                hierarchy
            )

        return layers_polarisation

    def calculate_navigability(
        self, from_node: tuple[int, int], to_node: tuple[int, int]
    ) -> float:
        r"""
        Calculate the difficulty of navigating from an arbitrary node :math:`s` in some layer :math:`a` to another arbitrary
        node :math:`t` in some layer :math:`b`, where :math:`a` and :math:`b` may or may not be the same layer.

        The formulae for the navigatability are defined by:

        .. math::

            S(s \rightarrow t) = -\log_{2}\sum_{\{p(s,t)\}} P[p(s,t)]

            P[p(s,t)] = \frac{1}{k_{s}} \prod_{j \in p(s,t)} \frac{1}{k_{j} - 1}

        :param from_node: (agent_index, graph_index) for the starting node.
        :type from_node: tuple[int, int]
        :param to_node: (agent_index, graph_index) for the end node.
        :type to_node: tuple[int, int]
        :return: The navigability value for the specified path.
        :rtype: float
        """
        # TODO: Implement this function
        raise NotImplementedError(
            "Navigability measure calculation is not yet implemented..."
        )
        return 0.0

    def calculate_interdependence(self, layer: int) -> float:
        r"""
        Calculate the layer interdependence; a measure of how much impact a specific layer has in the overall
        social network.

        The general formula for layer interdependence is defined as:

        .. math::

            \lambda^{a} = \frac{\sum_{i}\sum_{j \neq i}\Psi^{a}_{ij}}{\sum_{i}\sum_{j \neq i}\Psi_{ij}}

        where :math:`\Psi^{a}_{ij}` describes the number of shortest paths between nodes :math:`i` and :math:`j`
        using two or more layers, where at least one of the layers passed through is :math:`a`.

        For the case of multilayer social contagion modeling, it has been defined here as:

        .. math::

            \lambda^{a} = \frac{\sum_{i}\sum_{j \neq i}|OC'_{i}(j)^{a}|}{\sum_{i}\sum_{j \neq i}|OC'_{i}(j)|}

        where :math:`|OC'_{i}(j)^{a}|` is Agent :math:`j`'s opinion climate value as perceived by Agent :math:`i`
        in the social hierarchy layer :math:`a`. Although, the absolute of this value should be taken, as this
        is representative of the real `strength' of a layer.

        :param layer: The index of the layer of interest.
        :type layer: int
        :return: The layer interdependence measure for the layer of interest.
        :rtype: float
        """
        # Update the base graph's edge weights before performing any calculations (possibility for future features requiring this)
        self.update_base_graph()

        layer_of_interest: str = self.graphs.graphs[layer].name
        observed_opinions_all: dict[str, dict[str, dict[str, float]]] = {}

        for hierarchy in self.graphs:
            observed_opinions_layer: dict[str, dict[str, float]] = {}
            for agent_i in hierarchy.graph.nodes():
                agent_i_oc: dict[str, float] = hierarchy.estimate_neighbour_opinions(
                    agent_i.agent
                )
                observed_opinions_layer[agent_i.agent.id] = agent_i_oc
            observed_opinions_all[hierarchy.name] = observed_opinions_layer

        interdep_numerator: float = 0.0
        # Get the sum of all the estimated opinion values, only for the layer of interest (a)
        oc_a: dict[str, dict[str, float]] = observed_opinions_all[layer_of_interest]
        for agent_a_i, oc_a_i in oc_a.items():
            for oc_val in oc_a_i.values():
                interdep_numerator += abs(oc_val)

        interdep_denominator: float = 0.0
        # Get the sum of all the estimated opinion values for all layers (k)
        for layer_k, oc_k in observed_opinions_all.items():
            for agent_k_i, oc_k_i in oc_k.items():
                for oc_val in oc_k_i.values():
                    interdep_denominator += abs(oc_val)

        # Calculate the interdependence value for the layer
        if interdep_denominator != 0:
            layer_interdependence: float = interdep_numerator / interdep_denominator
        else:
            layer_interdependence = 0.0
        return layer_interdependence

    def init_base_graph(self) -> None:
        """
        Iterates over all the relationships in the existing social hierarchies and creates corresponding
        edges within the model's base graph.
        """
        for hierarchy in self.graphs:
            for idx, edge in hierarchy.graph.edge_index_map().items():
                graph_edge: GraphEdge = deepcopy(edge[2])

                # Get the index of the Agent objects within the model's AgentSet (not the graph's node set)
                base_from_idx, base_to_idx = self.get_base_indices_from_edge(
                    hierarchy, graph_edge
                )

                # Update the weigting in base graph if an edge exists and the weighting is different from the hierarchy's
                # The try, except is included for cases where the base graph may already contain explicitly initialised relationships
                try:
                    base_edge: GraphEdge = self.base_graph.graph.get_edge_data(
                        base_from_idx, base_to_idx
                    )
                    if graph_edge.weighting != base_edge.weighting:
                        self.base_graph.change_weights(
                            base_from_idx, base_to_idx, graph_edge.weighting
                        )
                # If the edge does not exist in the base graph, create it and add it to the base graph
                except NoEdgeBetweenNodes:
                    new_edge: dict[str, list[Any]] = {
                        "from_node": [base_from_idx],
                        "to_node": [base_to_idx],
                        "weighting": [graph_edge.weighting],
                        "name": [hierarchy.name],
                    }
                    self.base_graph.add_edges(deepcopy(new_edge))

                    # Manual garbage collection
                    del new_edge
        return None

    def get_base_indices_from_edge(
        self, hierarchy_graph: Graph, edge: GraphEdge
    ) -> tuple[int, int]:
        """
        A helper function for the base graph that takes in a GraphEdge object from a hierarchy graph and transforms
        the node indices from hierarchy graph indices to the respective index of the Agent objects in the model's
        AgentSet.

        :param hierarchy_graph: The corresponding hierarchy graph that the graph edge belongs in.
        :type hierarchy_graph: Graph
        :param edge: A graph edge from one of the model's hierarchy graphs in the GraphSet.
        :type edge: GraphEdge
        :return: The index in the AgentSet of the parent and child nodes involved in the hierarchy graph's relationship.
        :rtype: tuple[int, int]
        """
        # Actually GraphNode objects, but must be declared as "Any" for cases where a non-existent node index is passed to the function...
        from_node: Any = hierarchy_graph.get_node(edge.from_node)
        to_node: Any = hierarchy_graph.get_node(edge.to_node)

        from_index_base: int = self.agents.get_index(from_node.agent)
        to_index_base: int = self.agents.get_index(to_node.agent)

        return from_index_base, to_index_base

    def add_base_graph_edges(self, graph: Graph) -> None:
        """
        A function that takes a Graph object and adds all of its weighted edges to the model's base graph.

        :param graph: The new graph that is being added to self.graphs.
        :type graph: Graph
        """
        new_edges: dict[str, list[Any]] = {
            "from_node": [],
            "to_node": [],
            "weighting": [],
            "name": [],
        }

        for idx, edge in graph.graph.edge_index_map().items():
            graph_edge: GraphEdge = deepcopy(edge[2])

            # Get the index of the Agent objects within the model's AgentSet (not the graph's node set)
            base_from_idx, base_to_idx = self.get_base_indices_from_edge(
                graph, graph_edge
            )

            new_edges["name"].append(graph_edge.hierarchy)
            new_edges["from_node"].append(base_from_idx)
            new_edges["to_node"].append(base_to_idx)
            new_edges["weighting"].append(graph_edge.weighting)

            # Manual garbage collection
            del graph_edge, base_from_idx, base_to_idx

        self.base_graph.add_edges(deepcopy(new_edges))

        # Manual garbage collection
        del new_edges

        return None

    def update_base_graph(self) -> None:
        """
        Iterates over all the social hierarchy graphs and checks for pending edge changes.

        If pending changes exist, use all of the given information to apply the changes appropriately
        to the relationships present in the model's base graph.
        """
        for hierarchy in self.graphs:
            pending_changes: dict[str, EdgeChanges] = hierarchy.get_edge_changes()

            if len(pending_changes.keys()) == 0:
                continue

            for agents, change in pending_changes.items():
                from_id, to_id = agents.split(",")

                from_agent = self.agents.get_agent_by_id(from_id)
                to_agent = self.agents.get_agent_by_id(to_id)

                # Included for type checking
                if from_agent is not None and to_agent is not None:
                    from_idx = from_agent.index
                    to_idx = to_agent.index

                    edge_indices = self.base_graph.graph.edge_indices_from_endpoints(
                        from_idx, to_idx
                    )

                    for edge_index in edge_indices:
                        edge_data: GraphEdge = (
                            self.base_graph.graph.get_edge_data_by_index(edge_index)
                        )
                        if edge_data.hierarchy != change.hierarchy:
                            continue
                        else:
                            edge_data.set_weighting(change.weighting)
                            self.base_graph.graph.update_edge_by_index(
                                edge_index, deepcopy(edge_data)
                            )
        return None
