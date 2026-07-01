from __future__ import annotations

import os
from typing import Any

import matplotlib
import numpy as np
import rustworkx as rx

matplotlib.use("gtk4agg")
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from rustworkx.visualization import mpl_draw


class ABVisualiser:
    """
    The visualisations will include:
        - Graph structure plotting
        - Realtime model runtime display
        - Post-runtime summarisation graphs
        - Per-agent lifetime information
        - Per-graph lifetime information

    :param model: The model that this visualiser is being attached to.
    :type model: :py:class:`gatoh.model.model.ABModel`
    :param visualisation_dir: A path to a subdirectory to which all relevant visualisation outputs should be saved to.
    :type visualisation_dir: str
    :param aggregation_method: The method to use for aggregating parameters (i.e. ``mean'', ``median'', etc.).
    :type aggregation_method: str, optional
    :param save_visualisations: A flag indicating if the visualisation outputs should be saved.
    :type save_visualisations: bool, optional
    """

    def __init__(
        self,
        model: Any,
        visualisation_dir: str,
        aggregation_method: str = "median",
        save_visualisations: bool = True,
    ) -> None:
        self.parent_model: Any = model
        self.visualisation_dir: str = visualisation_dir
        self.save_visualisations: bool

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
        :type graph_node: :py:class:`gatoh.graph.graph.GraphNode`
        :return: A text representation of the node's agent opinion.
        :rtype: str
        """
        agent_opinion: str = str(graph_node.agent.opinion)
        return agent_opinion

    def visualise_hierarchy(
        self, hierarchy_name: str, save_graph: bool = True, show_graph: bool = True
    ) -> None:
        """
        A standalone function to visualise the graph for a specific social hierarchy within the ABModel.

        :param hierarchy_name: The name that has been assigned to the specified hierarchy.
        :type hierarchy_name: str
        :param save_graph: A flag indicating if the hierarchy visualisation should be saved.
        :type save_graph: bool, optional
        :param show_graph: A flag indicating if the hierarchy visualisation should be shown in the output terminal.
        :type show_graph: bool, optional
        """
        # Get the relevant Graph object from the GraphSet
        hierarchy_graph = self.parent_model.graphs.get_hierarchy(hierarchy_name)

        # Determine node colours based on radicalisation
        node_colours: list[str] = []
        for node in hierarchy_graph.graph.nodes():
            if node.agent.radicalised:
                node_colours.append("red")
            else:
                node_colours.append("blue")

        fig, ax = plt.subplots()

        # Draw the hierarchy Graph
        _ = mpl_draw(
            hierarchy_graph,
            ax=ax,
            with_labels=True,
            node_color=node_colours,
            labels=self.graph_node_opinion,
        )

        plt.draw()

        if save_graph:
            plt.savefig(
                f"{self.visualisation_dir}/{hierarchy_name}_graph.png", dpi=300.0
            )
        if show_graph:
            plt.show()
            _ = plt.waitforbuttonpress()

        plt.close(fig)
        return None

    def visualiser_iteration(self) -> None:
        """
        Visualise the radicalisation of agents across the entire agent population;
        called from the parent model's iterate function at every model iteration.
        """
        base_graph = self.parent_model.base_graph

        # Determine the colours based on radicalisation
        node_colours: list[str] = []
        for node in base_graph.graph.nodes():
            if node.agent.radicalised:
                node_colours.append("red")
            else:
                node_colours.append("blue")

        _ = mpl_draw(
            base_graph,
            ax=self.parent_model.ax,
            with_labels=True,
            node_color=node_colours,
            labels=self.graph_node_opinion,
        )

        self.parent_model.fig.draw()

        if self.save_visualisations:
            # Ensure that the model_runtime subdirectory exists
            if not os.path.exists(f"{self.visualisation_dir}/model_runtime"):
                os.mkdir(f"{self.visualisation_dir}/model_runtime")
            plt.savefig(
                f"{self.visualisation_dir}/model_runtime/iteration_{self.parent_model.current_iteration}.png",
                dpi=300.0,
            )

        return None
