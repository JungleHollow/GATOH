from __future__ import annotations

from typing import Any

import numpy as np
import rustworkx as rx
from matplotlib import pyplot as plt
from rustworkx.visualization import mpl_draw


class ABVisualiser:
    """
    The visualisations will include:
        - Graph structure plotting
        - Realtime model runtime display
        - Post-runtime summarisation graphs
        - Per-agent lifetime information
        - Per-graph lifetime information
    """

    # TODO: Implement visualisation features

    def __init__(self, model: Any, visualisation_dir: str) -> None:
        """
        :param model: The ABModel object that this visualiser is being attached to.
        :param visualisation_dir: A path to a subdirectory to which all relevant visualisation outputs should be saved to.
        """
        self.parent_model: Any = model
        self.visualisation_dir: str = visualisation_dir

    def graph_node_opinion(self, graph_node: Any) -> str:
        """
        A helper function that takes in a GraphNode object (node payloads in the Graphs),
        and reports the string conversion of the agent's opinion in the GraphNode.

        :param graph_node: The GraphNode for which the agent opinion is being reported.
        :return: A string representation of the node's agent opinion.
        """
        agent_opinion: str = str(graph_node.agent.opinion)
        return agent_opinion

    def visualise_hierarchy(
        self, hierarchy_name: str, save_graph: bool = True, show_graph: bool = True
    ) -> None:
        """
        Visualise the graph for a specific social hierarchy within the ABModel.

        :param hierarchy_name: A string of the name that has been assigned to the specified hierarchy.
        :param save_graph: A boolean flag indicating if the hierarchy visualisation should be saved.
        :param show_graph: A boolean flag indicating if the hierarchy visualisation should be shown in the output terminal.
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
        return None
