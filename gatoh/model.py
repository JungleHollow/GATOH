from __future__ import annotations

import gc
import os
from copy import deepcopy
from datetime import datetime
from multiprocessing.pool import Pool
from random import choices, randint
from shutil import rmtree
from typing import Any, TypedDict

import numpy as np
import yaml

# Used for type declarations in ABModel __init__
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rustworkx.rustworkx import NoEdgeBetweenNodes
from rustworkx import all_shortest_paths as rx_shortest_paths

from gatoh.agents import Agent, AgentSet, OPINION_MAX
from gatoh.graphs import Graph, GraphNode, GraphEdge, GraphSet
from gatoh.logging import GATOHLogger
from gatoh.utils import (
    EdgeChanges,
    YamlLoader,
    create_config_file,
)
from gatoh.visualisation import ABVisualiser


# Define global constants to avoid using "magic numbers" throughout the code

# The valid range of silencing threshold values
MIN_SILENCING_THRESH: float = 0.0
MAX_SILENCING_THRESH: float = 1.0
# The valid range of negation threshold values
MIN_NEGATION_THRESH: float = 0.0
MAX_NEGATION_THRESH: float = 1.0
# The valid range of radicalisation threshold values
MIN_RADICAL_THRESH: float = 0.0
MAX_RADICAL_THRESH: float = 1.0
# The default divisor to use for agent subsetting in generate_graphs
DEFAULT_SUBSETTING_DIV: int = 4
# The absolute threshold value to use when determining 'like-minded' collective opinion changes
LIKE_MINDED_THRESH: float = 0.05


class ConfigData(TypedDict):
    """
    A helper class used to type check the config data for :class:`~gatoh.model.ABModel`.
    """
    hierarchy_information: dict[str, tuple[float, float]]
    current_iteration: int
    max_iterations: int
    silencing_threshold: float
    negation_threshold: float
    radicalisation_threshold: float
    visualise: bool
    visualisation_dir: str
    suppress_warnings: bool
    checkpointing: bool
    save_dir: str
    data_file: str
    model_id: str


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
    :param print_interval: The iteration interval at which the model's logger should be printing detailed variable reports.
    :type print_interval: int, optional
    :param debug: A flag indicating if additional developer statistics should be tracked and reported during runtime.
    :type debug: bool, optional
    :param visualise: A flag indicating if the model should visualise emergent behaviour in real time.
    :type visualise: bool, optional
    :param visualisation_dir: The path to a directory in which all of this model's visualiser outputs should be saved to.
    :type visualisation_dir: str, optional
    :param vis_aggregation_method: The aggregation method that should be used when relevant for visualisation (i.e. "median", "mean", etc.).
    :type vis_aggregation_method: str, optional
    :param checkpointing: A flag indicating if the model's progress should be saved at the end of each iteration (useful in case of interrupted runtimes).
    :type checkpointing: bool, optional
    :param init_graphs: A flag indicating if empty graphs should be initialised with the input hierarchy information.
    :type init_graphs: bool, optional
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
        print_interval: int = 10,
        debug: bool = False,
        visualise: bool = True,
        visualisation_dir: str = "",
        vis_aggregation_method: str = "median",
        checkpointing: bool = True,
        init_graphs: bool = False,
        save_dir: str = "",
        data_file: str = "",
        model_id: str = "",
    ) -> None:
        self.hierarchy_information: dict[str, tuple[float, float]] = {}
        for idx, hierarchy in enumerate(hierarchy_names):
            self.hierarchy_information[hierarchy] = hierarchy_rw_distributions[idx]

        self.agent_opinion_rw: tuple[float, float] = agent_opinion_rw

        self.agents: AgentSet = AgentSet()
        self.graphs: GraphSet = GraphSet()

        if init_graphs:
            # Ensure that the input hierarchy information always produce at least empty graphs in the graphset
            # (Although, it is expected that add_graphs() or generate_graphs() will be called normally)
            for hierarchy, rw_distrib in self.hierarchy_information.items():
                hierarchy_graph: Graph = Graph(
                    hierarchy, rw_distrib, suppress_warnings=suppress_warnings
                )
                self.graphs.add_graph(hierarchy_graph)

        # A model-handled 'base' Graph that keeps track of all relationships across the social hierarchies
        # (Used to greatly simplify network-level graph calculations)
        self.base_graph: Graph = Graph(
            "base", (0.0, 0.0), suppress_warnings=suppress_warnings
        )

        self.debug: bool = debug
        self.logger: GATOHLogger = GATOHLogger(
            iterations,
            hierarchy_names,
            print_interval=print_interval,
            debug=self.debug,
        )

        self.visualise: bool = visualise
        self.visualisation_dir: str = visualisation_dir
        self.visualiser: ABVisualiser
        self.fig: Figure
        self.ax: Axes

        # Only create the visualisation objects if visualisation is required
        if self.visualise:
            self.visualiser = ABVisualiser(
                self.visualisation_dir,
                aggregation_method=vis_aggregation_method,
                save_visualisations=self.visualise,
            )

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

    def set_hierarchy_information(self, hierarchy: str, rw_params: tuple[float, float]) -> None:
        """
        A setter function that adds a new entry into the model's hierarchy information attribute.

        This function will always overwrite the parameters for an existing hierarchy in the model.

        :param hierarchy: The name of the new hierarchy being recorded.
        :type hierarchy: str
        :param rw_params: The random-walk (mean, variance) parameters for the new hierarchy.
        :type rw_params: tuple[float, float]
        """
        self.hierarchy_information[hierarchy] = rw_params
        if self.debug:
            self.logger.log_function_call("ABModel.set_hierarchy_information")
        return None

    def set_agent_opinion_rw(self, agent_opinion_rw: tuple[float, float]) -> None:
        """
        A setter function that changes the model's agent_opinion_rw attribute.

        :param agent_opinion_rw: The (mean, variance) parameters used for the agent opinions random-walk effect.
        :type agent_opinion_rw: tuple[float, float]
        """
        self.agent_opinion_rw = agent_opinion_rw
        if self.debug:
            self.logger.log_function_call("ABModel.set_agent_opinion_rw")
        return None

    def set_visualise(self, visualise: bool) -> None:
        """
        A setter function that changes the model's visualise flag.

        :param visualise: A flag indicating if the model should visualise its runtime and outputs.
        :type visualise: bool
        """
        # Assume that flipping from False to True means that the visualiser has not been initialised yet
        if not self.visualise and visualise:
            self.visualiser = ABVisualiser(self.visualisation_dir)
        # Assume that flipping from True to False means that the visualiser will not be used any more, or requires a full reset
        elif self.visualise and not visualise:
            del self.visualiser
        self.visualise = visualise
        if self.debug:
            self.logger.log_function_call("ABModel.set_visualise")
        return None

    def set_visualisation_dir(self, visualisation_dir: str, force: bool = False) -> None:
        """
        A setter function that changes the model's visualisation directory.

        :param visualisation_dir: The path to the directory where model visualisations should be saved.
        :type visualisation_dir: str
        :param force: A flag indicating if the function should explicitly create the directory if it does not exist.
        :type force: bool, optional
        :raises NotADirectoryError: If the input directory is not valid and the operation is not being forced.
        """
        if os.path.exists(visualisation_dir):
            self.visualisation_dir = visualisation_dir
        elif not os.path.exists(visualisation_dir) and force:
            os.mkdir(visualisation_dir)
            self.visualisation_dir = visualisation_dir
        elif not os.path.exists(visualisation_dir) and not force:
            raise NotADirectoryError(f"The path {visualisation_dir} does not point to a valid directory -- change the path or set 'force=True' to fix this")

        if self.debug:
            self.logger.log_function_call("ABModel.set_visualisation_dir")

        return None

    def override_current_iteration(self, current_iteration: int) -> None:
        """
        A setter function that overrides the model's current iteration value with a new one.

        This function should not be used if a custom iteration function is not in use, as it will cause unexpected simulation behaviours
        if not handled appropriately.
        """
        self.current_iteration = current_iteration
        if self.debug:
            self.logger.log_function_call("ABModel.override_current_iteration")
        return None

    def set_max_iterations(self, max_iterations: int) -> None:
        """
        A setter function that changes the model's max iterations attribute.

        :param max_iterations: The total number of iterations that the model should run for.
        :type max_iterations: int
        :raises ValueError: If max_iterations is not a valid integer.
        """
        if max_iterations < 0:
            raise ValueError(f"The max_iterations value {max_iterations} is invalid -- Use a positive integer")
        self.max_iterations = max_iterations
        if self.debug:
            self.logger.log_function_call("ABModel.set_max_iterations")
        return None

    def set_silencing_threshold(self, silencing_thresh: float) -> None:
        """
        A setter function that changes the model's silencing threshold.

        :param silencing_thresh: The threshold that the model should use for the opinion silencing effect.
        :type silencing_thresh: float
        :raises ValueError: If the silencing threshold is outside the range [0.0, 1.0].
        """
        if MIN_SILENCING_THRESH <= silencing_thresh <= MAX_SILENCING_THRESH:
            self.silencing_threshold = silencing_thresh
        else:
            raise ValueError(f"The silencing threshold value of {silencing_thresh} is outside the valid range of [0.0, 1.0]")

        if self.debug:
            self.logger.log_function_call("ABModel.set_silencing_threshold")

        return None

    def set_negation_threshold(self, negation_thresh: float) -> None:
        """
        A setter function that changes the model's negation threshold.

        :param negation_thresh: The threshold that the model should use for the opinion negation effect.
        :type negation_thresh: float
        :raises ValueError: If the negation threshold is outside the range [0.0, 1.0].
        """
        if MIN_NEGATION_THRESH <= negation_thresh <= MAX_NEGATION_THRESH:
            self.negation_threshold = negation_thresh
        else:
            raise ValueError(f"The negation threshold value of {negation_thresh} is outside the valid range of [0.0, 1.0]")

        if self.debug:
            self.logger.log_function_call("ABModel.set_negation_threshold")

        return None

    def set_radicalisation_threshold(self, radical_thresh: float) -> None:
        """
        A setter function that changes the model's radicalisation threshold.

        :param radical_thresh: The threshold that the model should use when determining agent (de)radicalisation.
        :type radical_thresh: float
        :raises ValueError: If the radicalisation threshold is outside the range [0.0, 1.0].
        """
        if MIN_RADICAL_THRESH <= radical_thresh <= MAX_RADICAL_THRESH:
            self.radicalisation_threshold = radical_thresh
        else:
            raise ValueError(f"The radicalisation threshold value of {radical_thresh} is outside the valid range of [0.0, 1.0]")

        if self.debug:
            self.logger.log_function_call("ABModel.set_radicalisation_threshold")

        return None

    def set_suppress_warnings(self, suppress_warnings: bool) -> None:
        """
        A setter function that changes the model's suppress warnings flag.

        :param suppress_warnings: A flag indicating if non-critical runtime warnings should be suppressed.
        :type suppress_warnings: bool
        """
        self.suppress_warnings = suppress_warnings
        if self.debug:
            self.logger.log_function_call("ABModel.set_suppress_warnings")
        return None

    def set_checkpointing(self, checkpointing: bool) -> None:
        """
        A setter function that changes the model's checkpointing flag.

        :param checkpointing: A flag indicating if the model should save checkpoints at the end of every iteration.
        :type checkpointing: bool
        """
        self.checkpointing = checkpointing
        if self.debug:
            self.logger.log_function_call("ABModel.set_checkpointing")
        return None

    def set_save_dir(self, save_dir: str, force: bool = False) -> None:
        """
        A setter function that changes the model's save directory.

        :param save_dir: The path to the root directory where all model components and sub-components should be saved.
        :type save_dir: str
        :param force: A flag indicating if the function should explicitly create the directory if it does not exist.
        :type force: bool, optional
        :raises NotADirectoryError: If the input directory is not valid and the operation is not being forced.
        """
        if os.path.exists(save_dir):
            self.save_dir = save_dir
        elif not os.path.exists(save_dir) and force:
            os.mkdir(save_dir)
            self.save_dir = save_dir
        elif not os.path.exists(save_dir) and not force:
            raise NotADirectoryError(f"The path {save_dir} does not point to a valid directory -- change the path or set 'force=True' to fix this")

        if self.debug:
            self.logger.log_function_call("ABModel.set_save_dir")

        return None

    def set_data_file(self, data_file: str) -> None:
        """
        A setter function that changes the path to which the final data file should be written.

        :param data_file: The path to where the model's final data .csv file should be written.
        :type data_file: str
        """
        self.data_file = data_file
        if self.debug:
            self.logger.log_function_call("ABModel.set_data_file")
        return None

    def set_model_id(self, model_id: str) -> None:
        """
        A setter function that changes the model's unique ID.

        :param model_id: The new identifier to assign to the model.
        :type model_id: str
        """
        self.model_id = model_id
        if self.debug:
            self.logger.log_function_call("ABModel.set_model_id")
        return None

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

        if self.debug:
            self.logger.log_function_call("AgentSet.save_agentset")
            self.logger.log_function_call("GraphSet.save_graphset")
            self.logger.log_function_call("Graph.save_graph")

        # Store the model's configurations in a YAML config file
        config_path: str
        if self.model_id != "":
            config_path = f"{self.save_dir}/model_{self.model_id}.yaml"
        else:
            config_path = f"{self.save_dir}/model_{datetime.now().strftime('%y-%m-%d %H-%M')}.yaml"
        config_data: ConfigData = {
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

        if self.debug:
            self.logger.log_function_call("ABModel.save_model")

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
                            config_data: ConfigData = yaml.load(
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
                    if self.debug:
                        self.logger.log_function_call("Graph.load_graph")
                case _:
                    # Currently unknown how/if to handle edge cases here
                    pass

        # Check if any compressed files exist and handle them from their parent modules
        if agentset_exists:
            self.agents.load_agentset(load_dir)
            if self.debug:
                self.logger.log_function_call("AgentSet.load_agentset")
        if graphset_exists:
            self.graphs.load_graphset(load_dir, self.hierarchy_information)
            if self.debug:
                self.logger.log_function_call("GraphSet.load_graphset")

        if self.debug:
            self.logger.log_function_call("ABModel.load_model")

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

        if self.debug:
            self.logger.log_function_call("GraphSet.add_graph")
            self.logger.log_function_call("ABModel.add_graph")

        return self.graphs

    def add_graphs(
        self,
        graphs: list[Graph] | list[str],
        names: list[str],
        rw_params: list[tuple[float, float]],
    ) -> GraphSet:
        """
        Add new Graphs to the Model's GraphSet.

        :param graphs: Graph objects or filepaths to stored GraphML objects.
        :type graphs: list[Graph] | list[str]
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
                if self.debug:
                    self.logger.log_function_call("GraphSet.add_graph")
            elif type(graph) is str:
                new_graph: Graph = Graph(names[idx], rw_params[idx])
                new_graph.load_graph(graph, names[idx])
                self.graphs.add_graph(new_graph)
                if self.debug:
                    self.logger.log_function_call("Graph.load_graph")
                    self.logger.log_function_call("GraphSet.add_graph")

        self.init_base_graph()

        if self.debug:
            self.logger.log_function_call("ABModel.add_graphs")

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
                random_k: int = randint(len(agents) // DEFAULT_SUBSETTING_DIV, len(agents))
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

            if self.debug:
                self.logger.log_function_call("Graph.generate_graph")

        if self.debug:
            self.logger.log_function_call("ABModel.generate_graphs")

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
        if self.debug:
            self.logger.log_function_call("Graph.add_nodes")
            self.logger.log_function_call("ABModel.add_agent")
            # Preemptively logging the agentset add
            self.logger.log_function_call("AgentSet.add")
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
            if self.debug:
                self.logger.log_function_call("AgentSet.add")

        # Add all the new Agent objects to the model-handled 'base' graph
        self.base_graph.add_nodes(agents)

        if self.debug:
            self.logger.log_function_call("Graph.add_nodes")
            self.logger.log_function_call("ABModel.add_agents")

        return self.agents

    def add_agents_to_hierarchy(self, agents: list[Agent], hierarchy: str) -> None:
        """
        A helper function that can directly add a collection of agents to a specific hierarchy in the model's
        graph set.

        This function will also add any input agents to the model's overarching AgentSet if they do not already
        exist in it.

        :param agents: The collection of agents that should be added as nodes to the hierarchy.
        :type agents: list[Agent]
        :param hierarchy: The hierarchy that the agents are being added to.
        :type hierarchy: str
        :raises KeyError: If the specified hierarchy does not exist in the graph set.
        :raises ValueError: If any object in the agents iterable is of an invalid type.
        """
        for idx, agent in enumerate(agents):
            if not isinstance(agent, Agent):
                raise TypeError(f"The object at index {idx} of the input iterable is not a valid Agent object -- cannot add it to the hierarchy graph '{hierarchy}'")
        hierarchy_to_extend: Graph | None = self.graphs.get_hierarchy(hierarchy)
        if hierarchy_to_extend is None:
            raise KeyError(f"The specified hierarchy '{hierarchy}' does not exist in the GraphSet -- cannot add agents to it")
        else:
            # Add the agents to the AgentSet if they do not exist
            for agent in agents:
                if agent not in self.agents:
                    self.add_agent(agent)
            # Finally, add the agents as nodes to the specified hierarchy
            hierarchy_to_extend.add_nodes(agents)
        return None

    def add_relationships_to_hierarchy(self, relationships: dict[str, list[Any]], hierarchy: str) -> None:
        """
        A helper function that can directly add relationships to a specific hierarchy in the model's graph set.

        Validation of from_node and to_node indices is performed within the called Graph function when adding the relationships.

        :param relationships: A <key, values> mapping providing 'to_node', 'from_node', and 'weighting' information.
        :type relationships: dict[str, list[int | float]]
        :param hierarchy: The hierarchy that the relationships are being added to.
        :type hierarchy: str
        :raises KeyError: If the specified hierarchy does not exist in the graph set.
        :raises ValueError: If any required key is missing, or the information is of a mismatching data type.
        """
        if "to_node" not in relationships or "from_node" not in relationships:
            raise ValueError("The relationships information is missing one of the required 'from_node' or 'to_node' keys")
        specified_hierarchy: Graph | None = self.graphs.get_hierarchy(hierarchy)
        if specified_hierarchy is None:
            raise KeyError(f"The specified hierarchy '{hierarchy}' does not exist in the GraphSet -- cannot add relationships to it")
        else:
            specified_hierarchy.add_edges(relationships)
        return None

    def generate_agents(
        self,
        id_base: str,
        personality_probs: dict[str, float],
        distribution: str = "gaussian",
        parameters: dict[str, float] | None = None,
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
        if self.debug:
            self.logger.log_function_call("model.generate_agents")
        return None

    def iterate(self, worker_pool: Pool | None = None) -> None:
        """
        Handles the main model iteration loop.

        :param worker_pool: A pool of workers that can distribute the iteration processing amongst themselves.
        :type worker_pool: :class:`~multiprocessing.pool.Pool`, optional
        """
        while self.current_iteration < self.max_iterations:
            # Initialise the logger state for the current iteration
            if self.current_iteration == 0:
                self.logger.new_iteration(init=True)
            else:
                self.logger.new_iteration()

            # Get and print the formatted debug string if appropriate
            if self.debug:
                self.logger_debug_iteration()
                debug_print_string: str = self.logger.debug_iteration_print()
                print(debug_print_string)

            # Initialise a dictionary to keep track of agent opinion changes
            # (this is done to prevent recursive updating of opinions during the evolution of opinions)
            new_agent_opinions: dict[str, tuple[float, list[float], list[bool]]] = {}

            # First each agent looks at its neighbours to see how their opinion will evolve this iteration
            if worker_pool is not None:
                opinion_results = worker_pool.imap(
                    self.iteration_opinion_calculation,
                    self.agents,
                    chunksize=10,
                )
                for opinion_result in opinion_results:
                    new_agent_opinions[opinion_result[0]] = opinion_result[1]
                    if self.debug:
                        self.logger.log_function_call("ABModel.iteration_opinion_calculation")

                # Manual garbage collection
                del opinion_results
                _ = gc.collect()
            else:
                for agent in self.agents:
                    opinion_result = self.iteration_opinion_calculation(agent)
                    new_agent_opinions[opinion_result[0]] = opinion_result[1]

                    if self.debug:
                        self.logger.log_function_call("ABModel.iteration_opinion_calculation")

                    # Manual garbage collection
                    del opinion_result
                    _ = gc.collect()

            self.iteration_opinion_changes(new_agent_opinions)
            self.step()
            self.update()

            self.logger_iteration()  # Handle the logger's iteration() calculations and call its method

            # Get this iteration's print string (will be formatted appropriately based on the print interval)
            iteration_print_string: str = self.logger.iteration_print()
            print(iteration_print_string)

            if self.visualise:
                self.visualiser.visualiser_iteration(
                    self.base_graph, self.current_iteration, model_name=self.model_id
                )
            if self.checkpointing:
                self.save_model()

            self.current_iteration += 1
        # Call the logger's save_data function which handles data persistence appropriately
        data_saved: bool = self.logger.save_data(self.data_file)
        if data_saved:
            print(
                f"\n\nGATOH logger data was successfully written to the file at path: {self.data_file}\n\n"
            )
        if self.debug:
            self.logger.log_function_call("ABModel.iterate")
        return None

    def iteration_opinion_calculation(
        self,
        agent: Agent,
    ) -> tuple[str, tuple[float, list[float], list[bool]]]:
        """
        A helper function that calculates the per-agent, per-hierarchy changes to opinions for the
        iteration, returning all necessary information for :meth:`~self.iteration_opinion_changes`
        to apply the opinion changes.

        This function was primarily created to allow for multiprocessing in the main :meth:`~self.iterate`
        function.

        :param agent: The agent for which the opinion changes are being calculated.
        :type agent: Agent
        :return: An <Agent ID : Changes info> mapping that provides all necessary information to apply the opinion changes for a specific agent.
        :rtype: tuple[str, tuple[float, list[float], list[bool]]]
        """
        agent.previous_opinion = agent.opinion
        for hierarchy in self.graphs:
            # Update the previous opinion across all hierarchies
            hierarchy.agent_previous_opinion(agent)

        collective_changes: list[float] = []
        for hierarchy in self.graphs:
            neighbour_influences: float | None = hierarchy.neighbour_influences(agent)
            if neighbour_influences is not None:
                collective_changes.append(neighbour_influences)
        collective_changes_sum: float = sum(collective_changes)

        # Account for the idea that a collection of like-minded agents will push each other towards more extreme opinions
        # even if all neighbours are already averaging around the same opinion value
        total_change: float
        if agent.previous_opinion < 0.0 and -LIKE_MINDED_THRESH < collective_changes_sum < 0.0:
            total_change = -LIKE_MINDED_THRESH
        elif agent.previous_opinion > 0.0 and 0.0 < collective_changes_sum < LIKE_MINDED_THRESH:
            total_change = LIKE_MINDED_THRESH
        else:
            # The previous checks mean that a "minor" collective changes sum that is going in the opposite direction
            # to the agent's opinion will still have the effect of moving the agent towards the neighbour average
            total_change = collective_changes_sum

        # Check for the existence of personal benefit across all of the agent's neighbours
        all_neighbour_indices: list[int] = list(
            self.base_graph.graph.neighbors(agent.index)
        )
        all_neighbour_benefits: list[bool] = []
        for neighbour_index in all_neighbour_indices:
            neighbour_object: Agent = self.base_graph.graph[neighbour_index].agent
            all_neighbour_benefits.append(neighbour_object.personal_benefit)

        # Define the type of the return
        opinion_result: tuple[str, tuple[float, list[float], list[bool]]]

        # Constrain to [-1, 1]
        # 100.0 and -100.0 are used as key delta values indicating that the opinion needs to be constrained
        if agent.opinion + total_change < -OPINION_MAX:
            opinion_result = (
                agent.id,
                (
                    -100.0,
                    collective_changes,
                    all_neighbour_benefits,
                ),
            )
        elif agent.opinion + total_change > OPINION_MAX:
            opinion_result = (
                agent.id,
                (100.0, collective_changes, all_neighbour_benefits),
            )
        else:
            opinion_result = (
                agent.id,
                (
                    total_change,
                    collective_changes,
                    all_neighbour_benefits,
                ),
            )
        return opinion_result

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
            if self.debug:
                self.logger.log_function_call("AgentSet.get_agent_by_id")

            if agent_object is not None:
                # Flag if the Agent was already radicalised
                existing_radicalisation: bool = agent_object.radicalised

                # After the opinion change, determine if the agent has become radicalised or deradicalised
                if not existing_radicalisation:
                    was_radicalised: bool = agent_object.radicalisation(
                        opinion_change_info[1],
                        opinion_change_info[2],
                        self.radicalisation_threshold,
                    )
                    if self.debug:
                        self.logger.log_function_call("Agent.radicalisation")
                    for hierarchy in self.graphs:
                        # Update the radicalisation status of the agent across all hierarchies
                        hierarchy.agent_radicalisation_change(
                            agent_object, was_radicalised
                        )
                        if self.debug:
                            self.logger.log_function_call("Graph.agent_radicalisation_change")

                    # Update the node in the base graph
                    self.base_graph.agent_opinion_change(
                        agent_object, opinion_change_info[0]
                    )
                    self.base_graph.agent_radicalisation_change(
                        agent_object, was_radicalised
                    )

                    # Update the radicalisation count in the logger as needed
                    # (was_radicalised will always be False if the agent was already radicalised)
                    self.logger.variables.increment_radicalised(was_radicalised)

                    if self.debug:
                        self.logger.log_function_call("Graph.agent_opinion_change")
                        self.logger.log_function_call("Graph.agent_radicalisation_change")
                        self.logger.log_function_call("LoggerVariables.increment_radicalised")

                    # Update the current opinion across all hierarchies
                    for hierarchy in self.graphs:
                        hierarchy.agent_opinion_change(agent_object, opinion_change_info[0])
                        if self.debug:
                            self.logger.log_function_call("Graph.agent_opinion_change")
                else:
                    was_deradicalised: bool = agent_object.deradicalisation(
                        opinion_change_info[1],
                        opinion_change_info[2],
                        self.radicalisation_threshold,
                    )

                    if self.debug:
                        self.logger.log_function_call("Agent.deradicalisation")

                    for hierarchy in self.graphs:
                        # Update the radicalisation status of the agent across all hierarchies
                        hierarchy.agent_radicalisation_change(
                            agent_object, not was_deradicalised
                        )
                        if self.debug:
                            self.logger.log_function_call("Graph.agent_radicalisation_change")

                    # Update the node in the base graph (flagging for deradicalisation)
                    self.base_graph.agent_opinion_change(
                        agent_object, opinion_change_info[0], deradicalisation=True
                    )
                    self.base_graph.agent_radicalisation_change(
                        agent_object, not was_deradicalised
                    )

                    # Update the deradicalisation count in the logger as needed
                    # (was_deradicalised will always be False if the agent was not already radicalised)
                    self.logger.variables.increment_deradicalised(was_deradicalised)

                    if self.debug:
                        self.logger.log_function_call("Graph.agent_opinion_change")
                        self.logger.log_function_call("Graph.agent_radicalisation_change")
                        self.logger.log_function_call("LoggerVariables.increment_deradicalised")

                    # Update the current opinion across all hierarchies (flagging for deradicalisation)
                    for hierarchy in self.graphs:
                        hierarchy.agent_opinion_change(agent_object, opinion_change_info[0], deradicalisation=True)
                        if self.debug:
                            self.logger.log_function_call("Graph.agent_opinion_change")
        if self.debug:
            self.logger.log_function_call("ABModel.iteration_opinion_changes")
        return None

    def step(self) -> None:
        """
        Steps the model forward one iteration.

        This does not handle agent opinion changes, but rather dynamic agent relationships
        and hierarchy weightings.
        """
        for graph in self.graphs:
            graph.step()
            if self.debug:
                self.logger.log_function_call("Graph.step")
        for agent in self.agents:
            agent.step(
                self.hierarchy_information,
                self.agent_opinion_rw,
            )
            if self.debug:
                self.logger.log_function_call("Agent.step")

        if self.debug:
            self.logger.log_function_call("ABModel.step")
        return None

    def update(self, worker_pool: Pool | None = None) -> None:
        """
        Updates the agents' internal states to match the model step. This mainly handles the construction of agents'
        perceived opinion climates within their hierarchies, and the simulation of opinion silencing behaviours depending
        on these climates.

        :param worker_pool: A pool of workers that can distribute the processing of the update function amongst themselves.
        :type worker_pool: :class:`~multiprocessing.pool.Pool`, optional
        """
        if worker_pool is not None:
            agent_updates = worker_pool.map(self.update_multi, self.agents)

            # Update the agent object, and the logger variables as needed
            for idx, agent_update in enumerate(agent_updates):
                self.agents.agents[idx].update(agent_update[0], agent_update[2])

                self.logger.variables.increment_silenced(agent_update[1])
                self.logger.variables.increment_negated(agent_update[2])
        else:
            for agent in self.agents:
                agent_update = self.update_multi(agent)

                agent.update(agent_update[0], agent_update[2])

                # Update the logger variables as needed
                self.logger.variables.increment_silenced(agent_update[1])
                self.logger.variables.increment_negated(agent_update[2])

        if self.debug:
            for _ in range(len(self.agents)):
                self.logger.log_function_call("ABModel.update_multi")
                self.logger.log_function_call("Agent.update")
                self.logger.log_function_call("LoggerVariables.increment_silenced")
                self.logger.log_function_call("LoggerVariables.increment_negated")
            self.logger.log_function_call("ABModel.update")

        return None

    def update_multi(self, agent: Agent) -> tuple[dict[str, bool], bool, bool]:
        """
        A helper function that allows for multiprocessing of the :meth:`~self.update` function.

        :param agent: The agent being updated.
        :type agent: Agent
        :return: The new is_silenced flags for the agent, and flags indicating if opinion silencing and negation ocurred this iteration.
        :rtype: tuple[dict[str, bool], bool, bool]
        """
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
                    est_opinion_climate
                )
                silenced[graph.name] = is_silenced[0]

                if is_silenced[0]:
                    was_silenced = True

                if not negation:
                    negation = agent.opinion_negation(
                        graph.name, is_silenced[1], self.negation_threshold
                    )
        return (silenced, was_silenced, negation)

    def logger_debug_iteration(self) -> None:
        """
        A helper function that handles the iteration of the logger's debugging component.
        """
        self.logger.debug_iteration()
        if self.debug:
            self.logger.log_function_call("ABModel.logger_debug_iteration")
        return None

    def logger_iteration(self, worker_pool: Pool | None = None) -> None:
        """
        Calculate any relevant aggregate statistics and then pass these to the logger's iteration() function to be stored.

        Statistics calculated currently:
            1. Aggregate network opinion
            2. Network radicalisation log odds
            3. Layer navigability for each hierarchy
            4. Layer interdependence for each hierarchy

        :param worker_pool: A pool of workers that can distribute the processing of the logger iteration between them.
        :type worker_pool: :class:`~multiprocessing.pool.Pool`
        """
        aggregate_opinion: float = self.calculate_aggregate_opinion()
        radicalisation_logodds: float = self.calculate_radicalisation_logodds()

        layer_interdependences: dict[str, float] = {}
        interdepencence_results: list[tuple[str, float]] = self.calculate_interdependences(worker_pool=worker_pool)
        for interdependence_result in interdepencence_results:
            layer_interdependences[interdependence_result[0]] = interdependence_result[1]

        layers_polarisation: dict[str, float] = self.calculate_layers_polarisation(worker_pool=worker_pool)

        self.logger.iteration(
            aggregate_opinion,
            radicalisation_logodds,
            layer_interdependences,
            layers_polarisation,
        )
        if self.debug:
            self.logger.log_function_call("ABModel.logger_iteration")
        return None

    def calculate_aggregate_opinion(self) -> float:
        """
        Calculates the aggregate network opinion by iterating over each Agent in the model.

        :return: The aggregate network opinion value.
        :rtype: float
        """
        opinion_sum: float = 0.0
        opinion_count: int = 0

        for agent in self.agents:
            opinion_sum += agent.opinion
            opinion_count += 1

        average_opinion: float = opinion_sum / opinion_count

        if self.debug:
            self.logger.log_function_call("ABModel.calculate_aggregate_opinion")

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

            if self.debug:
                self.logger.log_function_call("ABModel.calculate_radicalisation_logodds")

            return log_odds

        if self.debug:
            self.logger.log_function_call("ABModel.calculate_radicalisation_logodds")

        return 0.0

    def calculate_layers_polarisation(self, worker_pool: Pool | None = None) -> dict[str, float]:
        r"""
        Calculate the polarisation of the opinion climate within each hierarchy by calling each graph's calculate_polarisation() method.

        :param worker_pool: A pool of workers that can distribute the polarisation calculations amongst themselves.
        :type worker_pool: :class:`~multiprocessing.pool.Pool`
        :return: A <hierarchy : value> mapping containing the polarisation value for each hierarchy.
        :rtype: dict[str, float]
        """
        layers_polarisation: dict[str, float] = {}

        if worker_pool is None:
            for hierarchy in self.hierarchy_information:
                layers_polarisation[hierarchy] = self.graphs.calculate_polarisation(
                    hierarchy
                )
        else:
            polarisation_results = worker_pool.map(self.layers_polarisation_multi, self.hierarchy_information.keys())
            for polarisation_result in polarisation_results:
                layers_polarisation[polarisation_result[0]] = polarisation_result[1]

        if self.debug:
            for _ in range(len(self.hierarchy_information.keys())):
                self.logger.log_function_call("GraphSet.calculate_polarisation")
            self.logger.log_function_call("ABModel.calculate_layers_polarisation")

        return layers_polarisation

    def layers_polarisation_multi(self, hierarchy_name: str) -> tuple[str, float]:
        """
        A helper function that allows for multiprocessing of :meth:`~gatoh.model.model.ABModel.calculate_layers_polarisation`.

        :param hierarchy_graph: The name of the hierarchy graph for which the polarisation is being calculated.
        :type hierarchy_graph: str
        :return: The name and polarisation value for the given hierarchy.
        :rtype: tuple[str, float]
        """
        polarisation_value: float = self.graphs.calculate_polarisation(hierarchy_name)
        return (hierarchy_name, polarisation_value)

    def calculate_density(self) -> float:
        r"""
        Calculate the density of the entire network.

        The density of a graph is defined as:

        .. math::

            D = \frac{l}{\frac{n(n-1)}{2}}

        which simply refers to the proportion of existing relationships versus the total possible relationships.

        Given that gatoh is multilayer, the density calculation must instead be defined as:

        .. math::

            D = \frac{l}{\sum \frac{n(n-1)}{2}}

        where the summation is the total possible relationships across all layers.

        :return: The density metric for the entire social network.
        :rtype: float
        """
        # Base graph edge count represents the total existing relationships across all hierarchies
        total_l: int = self.base_graph.edge_count
        final_n: float = 0.0

        for hierarchy in self.graphs:
            hierarchy_n: float = (hierarchy.node_count * (hierarchy.node_count - 1)) / 2
            final_n += hierarchy_n

        if self.debug:
            self.logger.log_function_call("ABModel.calculate_density")

        return total_l / final_n

    def calculate_navigability(
        self, from_node: tuple[int, int], to_node: tuple[int, int]
    ) -> float:
        r"""
        Calculate the difficulty of navigating from an arbitrary node :math:`s` in some layer :math:`a` to another arbitrary
        node :math:`t` in some layer :math:`b`, where :math:`a` and :math:`b` may or may not be the same layer.

        The general formulae for the navigatability are defined by:

        .. math::

            S(s \rightarrow t) = -\log_{2}\sum_{\{p(s,t)\}} P[p(s,t)]

            P[p(s,t)] = \frac{1}{k_{s}} \prod_{j \in p(s,t)} \frac{1}{k_{j} - 1}

        Although, the relationships across the layers in gatoh are bidirectional and weighted, and as such,
        :math:`P[p(s,t)]` is redefined as:

        .. math::

            P[p(s,t)] = \frac{w_{s \rightarrow j_{o}}}{k_{s}} \prod_{j \in p(s,t)} \frac{w_{j \rightarrow j + 1}}{k_{j} - 1}

        :param from_node: (agent_index, graph_index) for the starting node.
        :type from_node: tuple[int, int]
        :param to_node: (agent_index, graph_index) for the end node.
        :type to_node: tuple[int, int]
        :return: The navigability value for the specified path.
        :rtype: float
        """
        # First, calculate all the possible shortest paths (from_node -> to_node) using the all-encompassing base graph.
        all_shortest_paths: list[list[int]] = rx_shortest_paths(self.base_graph.graph, from_node[1], to_node[1])

        if self.debug:
            # This DOES NOT indicate that rx_shortest_paths was authored in GATOH Graphs, simply that it is being
            # grouped with other Graph-related functions...
            self.logger.log_function_call("Graph.rx_shortest_paths")

        navigability_summation: float = 0.0

        # Next, for each shortest path, find P[p(s,t)]
        for shortest_path in all_shortest_paths:
            path_product: float = 0.0
            for i in range(len(shortest_path) - 1):
                current_edge: GraphEdge = self.base_graph.graph.get_edge_data(shortest_path[i], shortest_path[i + 1])
                node_out_degree: int = self.base_graph.graph.out_degree(shortest_path[i])
                if i == 0:
                    path_product = float(current_edge.weighting / node_out_degree)
                else:
                    # Node degree should always be >= 2, as there must exist at least one ingoing and one outgoing edge for the path to exist
                    path_product *= float(current_edge.weighting / node_out_degree - 1)

                if self.debug:
                    self.logger.log_function_call("Graph.get_edge_data")
                    self.logger.log_function_call("Graph.out_degree")

            navigability_summation += path_product

        # -log2(x) == log2(1 / x)
        navigability_value: float = float(np.log2(1.0 / navigability_summation))

        if self.debug:
            self.logger.log_function_call("ABModel.calculate_navigability")

        return navigability_value

    def calculate_interdependences(self, worker_pool: Pool | None = None) -> list[tuple[str, float]]:
        """
        A helper function that allows for multiprocessing of :meth:`~gatoh.model.model.ABModel.calculate_interdependence`.

        :param worker_pool: A pool of workers that can distribute the interdependence calculations amongst themselves.
        :type worker_pool: :class:`~multiprocessing.pool.Pool`
        :return: A list containing the necessary results.
        :rtype: list[tuple[str, float]]
        """
        interdependence_results: list[tuple[str, float]] = []
        if worker_pool is None:
            for hierarchy in self.hierarchy_information:
                interdependence_result: float = self.calculate_interdependence(self.graphs.get_index(hierarchy))
                interdependence_results.append((hierarchy, interdependence_result))
        else:
            result_values = worker_pool.map(self.calculate_interdependence, list(range(len(self.graphs))))
            for idx, result_value in enumerate(result_values):
                interdependence_results.append((self.graphs.graphs[idx].name, result_value))

        if self.debug:
            for _ in range(len(self.hierarchy_information.keys())):
                self.logger.log_function_call("GraphSet.get_index")
                self.logger.log_function_call("ABModel.calculate_interdependence")
            self.logger.log_function_call("ABModel.calculate_interdependences")

        return interdependence_results


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
        is representative of the real "strength" of a layer.

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
                if self.debug:
                    self.logger.log_function_call("Graph.estimate_neighbour_opinions")
                observed_opinions_layer[agent_i.agent.id] = agent_i_oc
            observed_opinions_all[hierarchy.name] = observed_opinions_layer

        interdep_numerator: float = 0.0
        # Get the sum of all the estimated opinion values, only for the layer of interest (a)
        oc_a: dict[str, dict[str, float]] = observed_opinions_all[layer_of_interest]
        for oc_a_i in oc_a.values():
            for oc_val in oc_a_i.values():
                interdep_numerator += abs(oc_val)

        interdep_denominator: float = 0.0
        # Get the sum of all the estimated opinion values for all layers (k)
        for oc_k in observed_opinions_all.values():
            for oc_k_i in oc_k.values():
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
            for edge in hierarchy.graph.edge_index_map().values():
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
                        if self.debug:
                            self.logger.log_function_call("Graph.change_weights")
                    if self.debug:
                        self.logger.log_function_call("Graph.get_edge_data")
                # If the edge does not exist in the base graph, create it and add it to the base graph
                except NoEdgeBetweenNodes:
                    new_edge: dict[str, list[Any]] = {
                        "from_node": [base_from_idx],
                        "to_node": [base_to_idx],
                        "weighting": [graph_edge.weighting],
                        "name": [hierarchy.name],
                    }
                    self.base_graph.add_edges(new_edge)

                    if self.debug:
                        self.logger.log_function_call("Graph.add_edges")

                    # Manual garbage collection
                    del new_edge
                    _ = gc.collect()

        if self.debug:
            self.logger.log_function_call("ABModel.init_base_graph")

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
        from_node: GraphNode | None = hierarchy_graph.get_node(edge.from_node)
        to_node: GraphNode | None = hierarchy_graph.get_node(edge.to_node)

        if self.debug:
            self.logger.log_function_call("Graph.get_node")
            self.logger.log_function_call("Graph.get_node")

        if from_node is not None and to_node is not None:
            from_index_base: int = self.agents.get_index(from_node.agent)
            to_index_base: int = self.agents.get_index(to_node.agent)

            if self.debug:
                self.logger.log_function_call("AgentSet.get_index")
                self.logger.log_function_call("AgentSet.get_index")
                self.logger.log_function_call("ABModel.get_base_indices_from_edge")

            return from_index_base, to_index_base

        # Including here for return checking
        raise RuntimeError("This line should not have been reached...")

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

        for edge in graph.graph.edge_index_map().values():
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
            _ = gc.collect()

        self.base_graph.add_edges(new_edges)

        # Manual garbage collection
        del new_edges
        _ = gc.collect()

        if self.debug:
            self.logger.log_function_call("Graph.add_edges")
            self.logger.log_function_call("ABModel.add_base_graph_edges")

        return None

    def update_base_graph(self) -> None:
        """
        Iterates over all the social hierarchy graphs and checks for pending edge changes.

        If pending changes exist, use all of the given information to apply the changes appropriately
        to the relationships present in the model's base graph.
        """
        for hierarchy in self.graphs:
            pending_changes: dict[str, EdgeChanges] = hierarchy.get_edge_changes()

            if self.debug:
                self.logger.log_function_call("Graph.get_edge_changes")

            if len(pending_changes.keys()) == 0:
                continue

            for agents, change in pending_changes.items():
                from_id, to_id = agents.split(",")

                from_agent = self.agents.get_agent_by_id(from_id)
                to_agent = self.agents.get_agent_by_id(to_id)

                if self.debug:
                    # Log twice as two calls were made
                    self.logger.log_function_call("AgentSet.get_agent_by_id")
                    self.logger.log_function_call("AgentSet.get_agent_by_id")

                # Included for type checking
                if from_agent is not None and to_agent is not None:
                    from_idx = from_agent.index
                    to_idx = to_agent.index

                    edge_indices = self.base_graph.graph.edge_indices_from_endpoints(
                        from_idx, to_idx
                    )

                    if self.debug:
                        self.logger.log_function_call("Graph.edge_indices_from_endpoints")

                    for edge_index in edge_indices:
                        edge_data: GraphEdge = (
                            self.base_graph.graph.get_edge_data_by_index(edge_index)
                        )
                        if self.debug:
                            self.logger.log_function_call("Graph.get_edge_data_by_index")

                        if edge_data.hierarchy != change.hierarchy:
                            continue
                        else:
                            edge_data.set_weighting(change.weighting)
                            self.base_graph.graph.update_edge_by_index(
                                edge_index, deepcopy(edge_data)
                            )
                            if self.debug:
                                self.logger.log_function_call("GraphEdge.set_weighting")
                                self.logger.log_function_call("Graph.update_edge_by_index")

        if self.debug:
            self.logger.log_function_call("ABModel.update_base_graph")

        return None
