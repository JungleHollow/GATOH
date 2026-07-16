from __future__ import annotations

import gc
import os
from typing import Any
from typing_extensions import deprecated

import matplotlib
import numpy as np
import seaborn as sns

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from rustworkx.visualization import mpl_draw

from gatoh.graphs.graphs import Graph


class ABVisualiser:
    """
    The visualisations will include:
        - Graph structure plotting
        - Realtime model runtime display
        - Post-runtime summarisation graphs
        - Per-agent lifetime information
        - Per-graph lifetime information

    :param visualisation_dir: A path to a subdirectory to which all relevant visualisation outputs should be saved to.
    :type visualisation_dir: str
    :param aggregation_method: The method to use for aggregating parameters (i.e. "mean", "median", etc.).
    :type aggregation_method: str, optional
    :param save_visualisations: A flag indicating if the visualisation outputs should be saved.
    :type save_visualisations: bool, optional
    """

    def __init__(
        self,
        visualisation_dir: str,
        aggregation_method: str = "median",
        save_visualisations: bool = True,
    ) -> None:
        self.visualisation_dir: str = visualisation_dir
        self.save_visualisations: bool

        self.aggregation_method: str = aggregation_method

        # If no visualisation save directory is provided, assume this always means no saving is desired
        if self.visualisation_dir == "":
            self.save_visualisations = False
        else:
            self.save_visualisations = save_visualisations

            # Ensure that the visualisations save directory exists
            if not os.path.exists(self.visualisation_dir):
                os.mkdir(self.visualisation_dir)

    def graph_node_opinion(self, graph_node: Any) -> str:
        """
        A helper function that takes in a GraphNode object (node payloads in the Graphs),
        and reports the string conversion of the agent's opinion in the GraphNode.

        :param graph_node: The graph node for which the agent opinion is being reported.
        :type graph_node: :class:`~gatoh.graphs.graphs.GraphNode`
        :return: A text representation of the node's agent opinion.
        :rtype: str
        """
        agent_opinion: str = str(graph_node.agent.opinion)
        return agent_opinion

    def graph_node_values(self, graph_node: Any) -> tuple[str, bool, float]:
        """
        A helper function that takes in a GraphNode object (node payloads in the Graphs),
        and reports the plotting information for the agent in the GraphNode.

        :param graph_node: The graph node for which the agent opinion is being reported.
        :type graph_node: :class:`~gatoh.graphs.graphs.GraphNode`
        :raises ValueError: If any of the agent values are of the incorrect type for plotting.
        :return: The radicalisation status of the agent and the float value of the node's agent opinion.
        :rtype: tuple[bool, float]
        """
        agent_id: str = graph_node.agent.id
        radicalised: bool = graph_node.agent.radicalised
        opinion: float = graph_node.agent.opinion
        if type(radicalised) is not bool or type(opinion) is not float or type(agent_id) is not str:
            raise ValueError("An Agent's radicalisation or opinion attributes are of an incorrect type to plot")
        return (agent_id, radicalised, opinion)

    def visualise_hierarchy(self, hierarchy_graph: Graph, save_graph: bool = True, show_graph: bool = True) -> None:
        """
        A standalone function to visualise the graph for a specific social hierarchy within the ABModel.

        :param hierarchy_graph: The hierarchy graph that is being visualised.
        :type hierarchy_graph: :class:`~gatoh.graphs.graphs.Graph`
        :param save_graph: A flag indicating if the hierarchy visualisation should be saved.
        :type save_graph: bool, optional
        :param show_graph: A flag indicating if the hierarchy visualisation should be shown in the output terminal.
        :type show_graph: bool, optional
        """
        num_agents: int = hierarchy_graph.node_count

        # First, extract all necessary information for plotting the agents
        agents_info: dict[str, tuple[bool, float]] = {}
        for node in hierarchy_graph.graph.nodes():
            agents_info[node.agent.id] = (node.agent.radicalised, node.agent.opinion)

        # Determine the array size so that a (10, x) 2D array shape is always used
        array_size: int
        if num_agents % 10 != 0:
            array_size = num_agents + (10 - num_agents % 10)
        else:
            array_size = num_agents

        # Create the agent labels and the data array
        agent_labels: np.ndarray = np.empty([array_size], dtype="S20")
        agent_data: np.ndarray = np.empty([array_size], dtype=np.int8)
        for idx, item in enumerate(agents_info.items()):
            agent_label: str = f"{item[1][1]:.2f}"
            agent_labels[idx] = agent_label
            agent_data[idx] = int(item[1][0])

        # Set any of the remainder cells to null values
        if array_size > num_agents:
            for i in range(num_agents, array_size):
                agent_labels[i] = "_"
                agent_data[i] = 2

        # Create the listed colour map
        colour_map: ListedColormap = ListedColormap(["red", "green"])

        # Reshape the arrays into 2D shapes (10, x)
        agent_labels = agent_labels.reshape(10, int(array_size / 10))
        agent_data = agent_data.reshape(10, int(array_size / 10))

        fig, ax = plt.subplots()

        ax = sns.heatmap(agent_data, cmap=colour_map, ax=ax, annot=agent_labels, fmt="s", cbar=False, linewidth=0.5, mask=agent_data==2)
        _ = ax.set_title(f"Visualisation of hierarchy '{hierarchy_graph.name}'")
        _ = ax.set(xlabel="", ylabel="")

        if save_graph:
            plt.savefig(f"{self.visualisation_dir}/{hierarchy_graph.name}_graph.png", dpi=300.0)
        if show_graph:
            plt.show()
            _ = plt.waitforbuttonpress()

        plt.close(fig)

        # Manual garbage collection
        del fig, ax, agent_labels, agent_data
        _ = gc.collect()

        return None

    @deprecated("Use visualise_hierarchy instead")
    def visualise_hierarchy_mpl(
        self, hierarchy_graph: Graph, save_graph: bool = True, show_graph: bool = True
    ) -> None:
        """
        **DEPRECATED**
        This function was originally used when :func:`~rustworkx.visualization.mpl_draw` was the graph plotting function in use.

        The visualisation module was reworked in July 2026 to work on a more flexible, custom visualisation system developed
        specifically for GATOH.

        The direct counterpart of this function that should be used in the new system is
        :meth:`~gatoh.visualisation.visualisation.ABVisualiser.visualise_hierarchy`.

        ---

        A standalone function to visualise the graph for a specific social hierarchy within the ABModel.

        :param hierarchy_graph: The hierarchy graph that is being visualised.
        :type hierarchy_graph: :class:`~gatoh.graphs.graphs.Graph`
        :param save_graph: A flag indicating if the hierarchy visualisation should be saved.
        :type save_graph: bool, optional
        :param show_graph: A flag indicating if the hierarchy visualisation should be shown in the output terminal.
        :type show_graph: bool, optional
        """
        # Determine node colours based on radicalisation
        node_colours: list[str] = []
        for node in hierarchy_graph.graph.nodes():
            if node.agent.radicalised:
                node_colours.append("red")
            else:
                node_colours.append("green")

        fig, ax = plt.subplots()

        # Draw the hierarchy Graph
        _ = mpl_draw(
            hierarchy_graph.graph,
            ax=ax,
            with_labels=True,
            node_color=node_colours,
            labels=self.graph_node_opinion,
        )

        plt.draw()

        if save_graph:
            plt.savefig(
                f"{self.visualisation_dir}/{hierarchy_graph.name}_graph.png", dpi=300.0
            )
        if show_graph:
            plt.show()
            _ = plt.waitforbuttonpress()

        plt.close(fig)

        # Manual garbage collection
        del fig, ax
        _ = gc.collect()
        return None

    def visualiser_iteration(self, base_graph: Graph, current_iteration: int, model_name: str | None = None) -> None:
        """
        Visualise the radicalisation of agents across the entire agent population;
        called from the parent model's iterate function at every model iteration.

        :param base_graph: The model's base graph to visualise.
        :type base_graph: :class:`~gatoh.graphs.graphs.Graph`
        :param current_iteration: The model's current runtime iteration number.
        :type current_iteration: int
        :param model_name: The name or identification assigned to the model that is being visualised.
        :type model_name: str, optional
        """
        num_agents: int = base_graph.node_count

        # First, extract all necessary information for plotting the agents
        agents_info: dict[str, tuple[bool, float]] = {}
        for node in base_graph.graph.nodes():
            agents_info[node.agent.id] = (node.agent.radicalised, node.agent.opinion)

        # Determine the array size so that a (10, x) 2D array shape is always used
        array_size: int
        if num_agents % 10 != 0:
            array_size = num_agents + (10 - num_agents % 10)
        else:
            array_size = num_agents

        # Create the agent labels and the data array
        agent_labels: np.ndarray = np.empty([array_size], dtype="S20")
        agent_data: np.ndarray = np.empty([array_size], dtype=np.int8)
        for idx, item in enumerate(agents_info.items()):
            agent_label: str = f"{item[1][1]:.2f}"
            agent_labels[idx] = agent_label
            agent_data[idx] = int(item[1][0])

        # Set any of the remainder cells to null values
        if array_size > num_agents:
            for i in range(num_agents, array_size):
                agent_labels[i] = "_"
                agent_data[i] = 2

        # Create the listed colour map
        colour_map: ListedColormap = ListedColormap(["red", "green"])

        # Reshape the arrays into 2D shapes (10, x)
        agent_labels = agent_labels.reshape(10, int(array_size / 10))
        agent_data = agent_data.reshape(10, int(array_size / 10))

        fig, ax = plt.subplots()

        ax = sns.heatmap(agent_data, cmap=colour_map, ax=ax, annot=agent_labels, fmt="s", cbar=False, linewidth=0.5, mask=agent_data==2)
        if model_name is not None:
            _ = ax.set_title(f"Visualisation of model {model_name} at iteration {current_iteration}")
        else:
            _ = ax.set_title(f"Visualisation of a GATOH model at iteration {current_iteration}")

        _ = ax.set(xlabel="", ylabel="")

        if self.save_visualisations:
            # Ensure that the model_runtime subdirectory exists
            if not os.path.exists(f"{self.visualisation_dir}/model_runtime"):
                os.mkdir(f"{self.visualisation_dir}/model_runtime")
            plt.savefig(f"{self.visualisation_dir}/model_runtime/iteration_{current_iteration}.png", dpi=300.0)

        plt.close(fig)

        # Manual garbage collection
        del fig, ax, agent_labels, agent_data
        _ = gc.collect()

        return None

    @deprecated("Use visualiser_iteration instead")
    def visualiser_iteration_mpl(self, base_graph: Graph, current_iteration: int) -> None:
        """
        **DEPRECATED**
        This function was originally used when :func:`~rustworkx.visualization.mpl_draw` was the graph plotting function in use.

        The visualisation module was reworked in July 2026 to work on a more flexible, custom visualisation system developed
        specifically for GATOH.

        The direct counterpart of this function that should be used in the new system is
        :meth:`~gatoh.visualisation.visualisation.ABVisualiser.visualiser_iteration`.

        ---

        Visualise the radicalisation of agents across the entire agent population;
        called from the parent model's iterate function at every model iteration.

        :param base_graph: The model's base graph to visualise.
        :type base_graph: :class:`~gatoh.graphs.graphs.Graph`
        :param current_iteration: The model's current runtime iteration number.
        :type current_iteration: int
        """
        # Determine the colours based on radicalisation
        node_colours: list[str] = []
        for node in base_graph.graph.nodes():
            if node.agent.radicalised:
                node_colours.append("red")
            else:
                node_colours.append("green")

        fig, ax = plt.subplots()

        _ = mpl_draw(
            base_graph.graph,
            ax=ax,
            with_labels=True,
            node_color=node_colours,
            labels=self.graph_node_opinion,
        )

        fig.canvas.draw()

        if self.save_visualisations:
            # Ensure that the model_runtime subdirectory exists
            if not os.path.exists(f"{self.visualisation_dir}/model_runtime"):
                os.mkdir(f"{self.visualisation_dir}/model_runtime")
            plt.savefig(
                f"{self.visualisation_dir}/model_runtime/iteration_{current_iteration}.png",
                dpi=300.0,
            )

        plt.close(fig)

        # Manual garbage collection
        del fig, ax
        _ = gc.collect()

        return None
