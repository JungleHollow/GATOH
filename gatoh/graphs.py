from __future__ import annotations

import contextlib
import concurrent.futures
import gc
import os
import pickle
import warnings
import zipfile
from collections.abc import Iterable, Iterator
from copy import deepcopy
from multiprocessing.pool import Pool
from random import Random
from shutil import rmtree
from typing import Any, Self
from typing import TypedDict
from typing import override

import numpy as np
import rustworkx as rx

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gatoh.agents import Agent

from gatoh.utils import (
    EdgeChanges,
    beta_value_attenuation,
    connected_watts_strogatz_graph,
    random_coinflip,
    value_rw_delta,
    watts_strogatz_graph,
    pygraph_to_pydigraph
)


class GenerationParam(TypedDict):
    """
    A data class used for typechecking the graph generation parameters.
    """
    m: int
    p: float
    sbm_sizes: int


class GraphNode:
    """
    A helper class that allows rustworx to more efficiently store information about Agents in the graph nodes.

    :param agent: The agent that is being associated with this GraphNode.
    :type agent: Agent
    """

    def __init__(self, agent: Agent) -> None:
        self.index: int
        self.agent: Agent = agent

    def set_index(self, idx: int) -> None:
        """
        A setter method to set the GraphNode's index value.

        :param idx: The index to set for this GraphNode.
        :type idx: int
        """
        self.index = idx
        return None

    @override
    def __str__(self) -> str:
        """
        An override of what calling 'print()' on a GraphNode object will output.

        :return: A printable representation of the GraphNode.
        :rtype: str
        """
        return f"Agent ({self.agent.id}) at graph node ({self.index})"


class GraphEdge:
    """
    A helper class that allows rustworx to more efficiently store information about Agent relationships in the graph edges.
    As the social hierarchies are assumed to be DiGraphs, each GraphEdge is directional, and the social weighting that Agent
    A places on Agent B will not necessarilly be equally reciprocated.

    :param hierarchy: The name of the social hierarchy that this edge belongs to.
    :type hierarchy: str
    :param from_node: The index of the origin node.
    :type from_node: int
    :param to_node: The index of the destination node.
    :type to_node: int
    :param weighting: The relationship opinion value that is being assigned (range [-1, 1]).
    :type weighting: float, optional
    :param rw_params: The normal distribution (mean, variance) to assign for relationship-specific random walk effects.
    :type rw_params: tuple[float, float], optional
    """

    def __init__(
        self,
        hierarchy: str,
        from_node: int,
        to_node: int,
        weighting: float = 0.0,
        rw_params: tuple[float, float] | None = None,
    ) -> None:
        self.index: int
        self.weighting: float = weighting
        self.from_node: int = from_node
        self.to_node: int = to_node
        self.hierarchy: str = hierarchy
        self.rw_params: tuple[float, float] | None = rw_params

    def set_index(self, idx: int) -> None:
        """
        A setter function that changes this GraphEdge's index value.

        :param idx: The index to store for this GraphEdge.
        :type idx: int
        """
        self.index = idx
        return None

    def set_weighting(self, value: float) -> None:
        """
        A setter function that changes this GraphEdge's weighting value.

        :param value: The new weighting to store for this GraphEdge.
        :type value: float
        """
        self.weighting = value
        return None

    def set_rw_params(self, rw_params: tuple[float, float]) -> None:
        """
        A setter function that changes this GraphEdge's rw_params value.

        :param rw_params: A (mean, variance) tuple specifying this relationship's unique random walk distribution.
        :type rw_params: tuple[float, float]
        """
        self.rw_params = rw_params
        return None

    def update_from_node(self, idx: int) -> None:
        """
        A setter function that updates the from_node's index for this GraphEdge.

        :param idx: The from_node's new index value to update to.
        :type idx: int
        """
        self.from_node = idx
        return None

    def update_to_node(self, idx: int) -> None:
        """
        A setter function that updates the to_node's index for this GraphEdge.

        :param idx: The to_node's new index value to update to.
        :type idx: int
        """
        self.to_node = idx
        return None

    def has_rw_params(self) -> bool:
        """
        A function that checks if this relationship has explicit random walk parameters.

        :return: A flag indicating if random walk parameters exist.
        :rtype: bool
        """
        if self.rw_params:
            return True
        return False

    @override
    def __str__(self) -> str:
        """
        An override of what calling 'print()' on a GraphEdge object will output.

        :return: A printable representation of this GraphEdge.
        :rtype: str
        """
        return f"GraphEdge of weight ({self.weighting}) from node ({self.from_node}) to node ({self.to_node}) in the {self.hierarchy} social layer"


class Graph:
    """
    A graph class that defines a single agent-based model layer.

    This corresponds to the agents' attitudes towards one another
    with respect to different social hierarchies.

    :param name: The name of the social hierarchy that this Graph object will be representing.
    :type name: str
    :param rw_params: The (mean, variance) of the normal distribution used for the dynamic relationships random walk.
    :type rw_params: tuple[float, float]
    :param generation_method: The random graph generation method that should be used where relevant.
    :type generation_method: str, optional
    :param suppress_warnings: A flag indicating if non-critical warnings should be suppressed.
    :type suppress_warnings: bool, optional
    :param dynamic_rels: A flag indicating if dynamic relationships should be modelled for this social hierarchy.
    :type dynamic_rels: bool, optional
    """

    def __init__(
        self,
        name: str,
        rw_params: tuple[float, float],
        generation_method: str = "",
        suppress_warnings: bool = False,
        dynamic_rels: bool = True,
    ) -> None:
        # Defined as DiGraph as it is common in social networks for relationships to be unidirectional or unbalanced
        self.graph: rx.PyDiGraph[GraphNode, GraphEdge] = rx.PyDiGraph()
        self.node_count: int = 0
        self.edge_count: int = 0
        self.name: str = name
        self.generation_method: str = generation_method
        self.dynamic_rels: bool = dynamic_rels
        self.suppress_warnings: bool = suppress_warnings
        self.rw_params: tuple[float, float] = rw_params
        self.generation_params: GenerationParam = {  # Used for random graph generation, can be manually set by the user if desired
            "p": 0.25,
            "m": 3,
            "sbm_sizes": 10,
        }
        self.pending_edge_changes: dict[str, EdgeChanges] = {}

    def change_generation_params(self, **params: int | float) -> None:
        """
        Setter function which outlines the existing generation parameters used in generate_graph()
        and allows the user to alter them.

        :param p: The probability of edge rewiring (small-world) or edge creation (random).
        :type p: float, optional
        :param m: The number of nearest neighbours that each node is connected to initially (scale-free).
        :type m: int, optional
        :param: sbm_sizes: The size of generated blocks (blockmodel).
        :type sbm_sizes: int, optional
        :raises UserWarning: If invalid parameter keys or data types are input to the function.
        """
        for key, value in params.items():
            if (
                key not in self.generation_params.keys()
            ):  # Skip any invalid parameters which have been passed
                warnings.warn(
                    f"WARNING: Invalid graph generation parameter ({key}) specified when trying to modify parameter values.",
                    category=UserWarning,
                )
                continue
            elif not isinstance(self.generation_params[key], type(value)):  # Skip altering any parameters which have been assigned invalid data types
                warnings.warn(
                    f"WARNING: Invalid data type detected for the value when modifying parameter {key}.",
                    category=UserWarning,
                )
                continue
            self.generation_params[key] = value
        return None

    def load_graph(
        self, path: str, name: str, rw_params: tuple[float, float] | None = None
    ) -> None:
        """
        Loads a Graph object stored in the GraphML format from the given path.
        The social hierarchy name must be explicitly passed with this call.

        :param path: Path to a stored graph file.
        :type path: str
        :param name: The name of the hierarchy that the stored Graph belongs to.
        :type name: str
        :param rw_params: The mean and variance of the Graph's random walk distribution.
        :type rw_params: tuple[float, float], optional
        """
        graph: list[rx.PyDiGraph | rx.PyGraph] = rx.read_graphml(path)
        if isinstance(graph[0], rx.PyDiGraph):
            self.graph = graph[0]
        else:
            converted_graph: rx.PyDiGraph = pygraph_to_pydigraph(graph[0])
            self.graph = converted_graph
        self.node_count = len(self.graph.nodes())
        self.edge_count = len(self.graph.edges())
        self.name = name

        if rw_params:
            self.rw_params = rw_params

        return None

    def save_graph(self, path: str) -> None:
        """
        Saves the existing Graph object in the GraphML format to the given path.

        :param path: Path to which the Graph will be saved.
        :type path: str
        """
        rx.write_graphml(self.graph, path)

        if not os.path.exists(path):
            raise EnvironmentError(f"Failed to write graph {self.name} to path: {path}")

        return None

    def get_node(self, node_index: int) -> GraphNode | None:
        """
        A getter function to access GraphNode objects.

        :param node_index: The index of the node to access.
        :type node_index: int
        :raises RuntimeWarning: If the node index is out of bounds for this Graph.
        :return: The graph node if the index was valid, or None otherwise.
        :rtype: GraphNode | None
        """
        try:
            return self.graph.nodes()[node_index]
        except IndexError:
            warnings.warn(
                f"WARNING: Node with index {node_index} is out of bounds for graph {self.name} with {self.node_count} total nodes.",
                category=RuntimeWarning,
            )
            return None

    def get_edge(self, edge_index: int) -> GraphEdge | None:
        """
        A getter function to access GraphEdge objects.

        :param edge_index: The index of the edge to access.
        :type edge_index: int
        :raises RuntimeWarning: If the edge index is out of bounds for this Graph.
        :return: The graph edge if the index was valid, or None otherwise.
        :rtype: GraphEdge | None
        """
        try:
            return self.graph.edges()[edge_index]
        except IndexError:
            warnings.warn(
                f"WARNING: Edge with index {edge_index} is out of bounds for graph {self.name} with {self.edge_count} total edges.",
                category=RuntimeWarning,
            )
            return None

    def update_node_indices(self) -> None:
        """
        Iterates over all the existing nodes in the graph and updates their stored indices to reflect the current graph state.

        Will also update the graph node_count attribute.
        """
        for index in self.graph.node_indices():
            self.graph[index].set_index(index)
        self.update_edge_indices()
        self.node_count = len(self.graph.nodes())
        return None

    def add_nodes(self, agents: Iterable[Agent]) -> None:
        """
        Creates appropriate GraphNodes from the given Agents, and then adds these to the graph.

        :param agents: The agents that will be converted to GraphNodes and added to the graph.
        :type agents: Iterable[Agent]
        """
        nodes: list[GraphNode] = []
        for agent in agents:
            agent_node = GraphNode(agent)
            nodes.append(agent_node)

        _ = self.graph.add_nodes_from(nodes)
        self.update_node_indices()
        return None

    def update_edge_indices(self) -> None:
        """
        Iterates over all the existing edges in the graph and updates their stored indices to reflect the current graph state.

        Will also update the graph edge_count attribute.
        """
        for idx, data in self.graph.edge_index_map().items():
            graph_edge: GraphEdge | list[GraphEdge] = data[2]
            # Workaround for undetermined error where a list of a single GraphEdge is added to the base graph at some point
            if isinstance(graph_edge, list):
                edge_object: GraphEdge = graph_edge[0]
                edge_object.set_index(idx)
            else:
                graph_edge.set_index(idx)
                self.graph.update_edge_by_index(idx, graph_edge)
        self.edge_count = len(self.graph.edges())
        return None

    def add_edges(self, edges: dict[str, list[Any]]) -> None:
        """
        Creates appropriate GraphEdges from the given dictionary and then adds these to the graph.

        The parameter :attr:`edges` has been typed as :class:`~typing.Any` to simplify typechecking.

        :param edges: A mapping of <key : list> where each key corresponds to (from_node, to_node, [optional] weighting, [optional] name).
        :type edges: dict[str, list[int | float | tuple[float, float]]]
        """
        graph_edges: list[tuple[int, int, GraphEdge]] = []
        from_nodes: list[int] = edges["from_node"]
        to_nodes: list[int] = edges["to_node"]
        weightings: list[float] | None = None
        rw_params: list[tuple[float, float]] | None = None

        if "weighting" in edges.keys():
            weightings = edges["weighting"]
        if "rw_param" in edges.keys():
            rw_params = edges["rw_param"]

        # Used in case that explicit hierarchy names are set per edge (in the case of the mixed-hierarchy base graph in the model for example)
        names: list[str] | None = None
        if "name" in edges.keys():
            names = edges["name"]

        # Declare the data type of 'edge'
        edge: GraphEdge

        if weightings:
            if names:
                if rw_params:
                    for i in range(len(from_nodes)):
                        edge = GraphEdge(
                            names[i],
                            from_nodes[i],
                            to_nodes[i],
                            weighting=weightings[i],
                            rw_params=rw_params[i],
                        )
                        graph_edges.append((from_nodes[i], to_nodes[i], edge))
                else:
                    for i in range(len(from_nodes)):
                        edge = GraphEdge(
                            names[i],
                            from_nodes[i],
                            to_nodes[i],
                            weighting=weightings[i],
                        )
                        graph_edges.append((from_nodes[i], to_nodes[i], edge))
            else:
                if rw_params:
                    for i in range(len(from_nodes)):
                        edge = GraphEdge(
                            self.name,
                            from_nodes[i],
                            to_nodes[i],
                            weighting=weightings[i],
                            rw_params=rw_params[i],
                        )
                else:
                    for i in range(len(from_nodes)):
                        edge = GraphEdge(
                            self.name, from_nodes[i], to_nodes[i], weightings[i]
                        )
                        graph_edges.append((from_nodes[i], to_nodes[i], edge))
        else:
            if names:
                if rw_params:
                    for i in range(len(from_nodes)):
                        edge = GraphEdge(
                            names[i], from_nodes[i], to_nodes[i], rw_params=rw_params[i]
                        )
                        graph_edges.append((from_nodes[i], to_nodes[i], edge))
                else:
                    for i in range(len(from_nodes)):
                        edge = GraphEdge(names[i], from_nodes[i], to_nodes[i])
                        graph_edges.append((from_nodes[i], to_nodes[i], edge))
            else:
                if rw_params:
                    for i in range(len(from_nodes)):
                        edge = GraphEdge(
                            self.name,
                            from_nodes[i],
                            to_nodes[i],
                            rw_params=rw_params[i],
                        )
                        graph_edges.append((from_nodes[i], to_nodes[i], edge))
                else:
                    for i in range(len(from_nodes)):
                        edge = GraphEdge(self.name, from_nodes[i], to_nodes[i])
                        graph_edges.append((from_nodes[i], to_nodes[i], edge))

        _ = self.graph.add_edges_from(graph_edges)
        self.update_edge_indices()
        return None

    def generate_graph(
        self,
        agents: list[Agent],
        method: str = "",
        relationship_range: tuple[float, float] = (-1.0, 1.0),
        ensure_complete: bool = True,
    ) -> Self:
        """
        Randomly generate edges between existing Graph nodes and add them to the graph.

        :param agents: The subset of Agents in the base model that are being used as the nodes for this graph.
        :type agents: list[Agent]
        :param method: The random generation method to use. Possible choices include: 'small-world', 'scale-free', 'random', 'blockmodel'; Defaults to 'small-world'.
        :type method: str, optional
        :param relationship_range: The valid range of generated relationship strengths (at most, constrained to [-1.0, 1.0]).
        :type relationship_range: tuple[float, float], optional
        :param ensure_complete: A flag indicating if the generated graph should be complete or not (in the case of 'small-world').
        :type ensure_complete: bool, optional
        :raises ValueError: If no agents are being passed to this function.
        :raises ValueError: If an invalid random generation method is being passed to this function.
        :return: A reference to this Graph object.
        :rtype: Graph
        """
        if len(agents) <= 0:
            raise ValueError(
                f"Attempting to generate random graph for hierarchy '{self.name}' without passing any valid Agents."
            )

        if self.generation_method != "":
            method = self.generation_method
        elif method == "":
            # Default to small-world if no explicit method was passed during initialisation or to this function
            method = "small-world"
            self.generation_method = "small-world"
        else:
            # Update self.generation_method with the explicit method that was passed
            self.generation_method = method

        n: int = len(agents)
        generated_graph: rx.PyDiGraph = rx.PyDiGraph()  # Initialise an empty graph for predictable behaviour in case of assignation errors
        random_gen: Random = (
            Random()
        )  # Initialise a random generator instance for this function

        match method:
            case "small-world":
                # Watts-Strogatz
                k: int = int(
                    np.ceil(np.log(n))
                )  # The smallest integer which is larger than log(n) to guarantee graph connectivity
                if ensure_complete:
                    generated_graph = connected_watts_strogatz_graph(
                        n, k, self.generation_params["p"]
                    )
                else:
                    generated_graph = watts_strogatz_graph(
                        n, k, self.generation_params["p"]
                    )

                # Explicitly set the generation method again to mark the method that has been used (for clarity)
                self.generation_method = "small-world"
            case "scale-free":
                # Barbasi-Albert
                generated_graph = rx.directed_barabasi_albert_graph(
                    n, self.generation_params["m"]
                )
                self.generation_method = "scale-free"
            case "random":
                # Erdos-Renyi
                generated_graph = rx.directed_gnp_random_graph(
                    n, self.generation_params["p"]
                )
                self.generation_method = "random"
            case "blockmodel":
                # Holland et al.
                sbm_remainder: int = (
                    n % self.generation_params["sbm_sizes"]
                )  # Determine if there will be any remainder with the specified block size
                sbm_n_blocks: int = (
                    len(agents) // self.generation_params["sbm_sizes"]
                )  # Determine how many blocks will be created
                sbm_sizes: list[int] = [
                    self.generation_params["sbm_sizes"] for _ in range(sbm_n_blocks)
                ]
                sbm_sizes[-1] += (
                    sbm_remainder  # If any agents are left over, add them all to the last block
                )

                sbm_probabilities: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.zeros(
                    (sbm_n_blocks, sbm_n_blocks), dtype=np.float64
                )  # Initialise a BxB array to hold the probabilities for inter-block connections
                for i in range(sbm_probabilities.shape[0]):
                    for j in range(sbm_probabilities.shape[1]):
                        sbm_probabilities[i, j] = (
                            random_gen.random()
                        )  # Set a random probability for edge connectivity from block i to block j (directed, asymmetrical)

                generated_graph = rx.directed_sbm_random_graph(
                    sbm_sizes, sbm_probabilities, False
                )  # "False" to disallow existence of self loops in the graph
                self.generation_method = "blockmodel"

                # Manual garbage collection
                del sbm_sizes, sbm_probabilities
                _ = gc.collect()

            case _:
                self.generation_method = "INVALID"
                raise ValueError(
                    f"Attempting to generate random graph with a non-supported method ({method}).\n\nUse one of the supported methods: 'small-world', 'scale-free', 'random', or 'blockmodel'..."
                )

        graph_nodes: list[GraphNode] = []
        for index in range(len(generated_graph.nodes())):
            graph_node: GraphNode = GraphNode(agents[index])
            graph_node.set_index(index)
            graph_nodes.append(graph_node)
        for idx, graph_node in enumerate(graph_nodes):
            generated_graph[idx] = (
                graph_node  # Update all the graph nodes with the new GraphNode data objects
            )

        self.graph = generated_graph  # Store the generated graph as the object's "graph" attribute (with 0.0 relationship weights currently)

        for index, edge in generated_graph.edge_index_map().items():
            generated_value = random_gen.uniform(
                relationship_range[0], relationship_range[1]
            )  # Generate a random value in the specified range (default is [-1.0, 1.0])

            graph_edge: GraphEdge = GraphEdge(
                self.name, edge[0], edge[1], weighting=generated_value
            )

            self.graph.update_edge_by_index(
                index, graph_edge
            )  # Update the edge with a GraphEdge object

        # Update the node and edge counts manually as no call to update_x_indices() have been made
        self.node_count = len(self.graph.nodes())
        self.edge_count = len(self.graph.edges())

        return self

    def relationship_exists(self, from_node: int, to_node: int) -> int | None:
        """
        Checks for the existence of a relationship (weighted edge) between two Agents (nodes).

        :param from_node: the node index of the parent node.
        :type from_node: int
        :param to_node: the node index of the child node.
        :type to_node: int
        :return: The index of the edge if the relationship exists, or None otherwise.
        :rtype: int | None
        """
        for edge in self.graph.edges():
            if edge.from_node == from_node and edge.to_node == to_node:
                return edge.index
        return None

    def get_relationships(
        self, node_1: int, node_2: int
    ) -> dict[tuple[int, int], float] | None:
        """
        Retrieves and reports the bidirectional relationship weightings between two nodes in the Graph.

        :param node_1: the node index of Agent 1.
        :type node_1: int
        :param node_2: the node index of Agent 2.
        :type node_2: int
        :return: The bidirectional edge weightings between two nodes (if they exist).
        :rtype: dict[tuple[int, int], float] | None
        """
        if not self.relationship_exists(node_1, node_2):
            return None

        relationships_dict: dict[tuple[int, int], float] = {}

        with contextlib.suppress(KeyError):
            relationships_dict[(node_2, node_1)] = self.graph.adj_direction(
                node_1, True
            )[node_2].weighting

        with contextlib.suppress(KeyError):
            relationships_dict[(node_1, node_2)] = self.graph.adj_direction(
                node_1, False
            )[node_2].weighting

        return relationships_dict

    def get_relationship(self, from_node: Agent, to_node: Agent) -> float:
        """
        Return a directed relationship from one node to another.

        :param from_node: The node that the relationship originates from.
        :type from_node: Agent
        :param to_node: The node that the relationship points to.
        :type to_node: Agent
        :return: The weighting of the directed relationship (from_node -> to_node).
        :rtype: float
        """
        from_index: int | None = self.get_agent_index(from_node)
        if from_index is None:
            # This should never be reached and is included for type checking purposes
            return 0.0

        relationship_dict: dict[int, GraphEdge] = self.graph.adj_direction(from_index, False)

        to_index: int | None = self.get_agent_index(to_node)
        if to_index is None:
            return 0.0

        graph_edge: GraphEdge = relationship_dict[to_index]
        return graph_edge.weighting

    def change_weights(self, node_1: int, node_2: int, value: float) -> None:
        """
        Updates the weight of the relationship between two agents in the graph.

        If no relationship previously exists, a new one is created.

        :param node_1: The index of some Agent in the graph.
        :type node_1: int
        :param node_2: The index  of some other Agent in the graph.
        :type node_2: int
        :param value: The new weight to assign.
        :type value: float
        """
        edge_index: int | None = self.relationship_exists(node_1, node_2)
        updated_edge: tuple[int, int, GraphEdge] = (node_1, node_2, GraphEdge(self.name, node_1, node_2, value))
        if edge_index is not None:
            self.graph.update_edge_by_index(edge_index, updated_edge[-1])
        else:
            _ = self.graph.add_edges_from([updated_edge])

        self.update_edge_indices()
        self.register_edge_change(node_1, node_2, value)
        return None

    def register_edge_change(
        self, from_node: int, to_node: int, weighting: float
    ) -> None:
        """
        A helper function that records a pending edge change for this graph.

        :param from_node: The index of the origin node.
        :type from_node: int
        :param to_node: The index of the destination node.
        :type to_node: int
        :param weighting: The new weighting that is being assigned to the edge.
        :type weighting: float
        """
        from_agent = self.get_node(from_node)
        to_agent = self.get_node(to_node)
        # Included for type checking
        if from_agent is not None and to_agent is not None:
            edge_change: EdgeChanges = EdgeChanges(self.name, weighting)
            self.pending_edge_changes[f"{from_agent.agent.id},{to_agent.agent.id}"] = (
                edge_change
            )

    def get_edge_changes(self) -> dict[str, EdgeChanges]:
        """
        A getter function that returns the current register of pending edge changes for this graph,
        and then resets the attribute to an empty entry.

        :return: A <"ID1,ID2" : edge change> mapping that outlines the edge changes being made between two agents in the graph.
        :rtype: dict[str, EdgeChanges]
        """
        changes_register: dict[str, EdgeChanges] = deepcopy(self.pending_edge_changes)

        # Manual garbage collection
        del self.pending_edge_changes
        _ = gc.collect()

        self.pending_edge_changes = {}

        return changes_register

    def remove_node(self, node: int) -> None:
        """
        Removes a node from the graph, along with any relationships involving it.

        :param node: The node index to remove from the graph.
        :type node: int
        """
        self.graph.remove_node(node)

        edges_to_remove: list[tuple[int, int]] = []
        for edge in self.graph.edges():
            if edge.from_node == node or edge.to_node == node:
                edges_to_remove.append((edge.from_node, edge.to_node))

        for edge in edges_to_remove:
            self.remove_edge(edge[0], edge[1])
        # No need to update indices, as rustworkx will automatically add new nodes/edges into the largest empty index
        return None

    def node_relationships_count(self, node_index: int) -> int:
        """
        Report the number of ingoing and outgoing relationships that a specific node is involved in.

        :param node_index: The index of the node that is being inspected.
        :type node_index: int
        :return: The total number of relationships involving the input node.
        :rtype: int
        """
        ingoing_relationships: int = self.graph.in_degree(node_index)
        outgoing_relationships: int = self.graph.out_degree(node_index)
        return ingoing_relationships + outgoing_relationships

    def remove_edge(self, from_node: int, to_node: int) -> None:
        """
        Removes a single edge from the graph.

        Throws a warning without interrupting the runtime if the edge did not exist in the first place.

        :param from_node: the parent node in the edge.
        :type from_node: int
        :param to_node: the child node in the edge.
        :type to_node: int
        :raises UserWarning: If the edge (from_node -> to_node) does not exist in the Graph.
        """
        edge_exists: int | None = self.relationship_exists(from_node, to_node)
        if edge_exists:
            self.graph.remove_edge(from_node, to_node)

            # Check the number of relationships that each node now has
            from_rels_count: int = self.node_relationships_count(from_node)
            to_rels_count: int = self.node_relationships_count(to_node)

            # If either node now has no relationships, it is removed from the graph entirely
            if from_rels_count == 0:
                self.remove_node(from_node)
            if to_rels_count == 0:
                self.remove_node(to_node)
        else:
            warnings.warn(
                f"WARNING: Attempted to remove edge ({from_node} -> {to_node}) which does not exist in the graph.",
                category=UserWarning,
            )
        return None

    def agent_in_graph(self, agent: Agent) -> bool:
        """
        A simple function that checks wether an Agent exists within a Graph.

        :param agent: the agent whose existence in the Graph is being checked for.
        :type agent: Agent
        :return: A flag indicating if the Agent exists in the Graph.
        :rtype: bool
        """
        for node in self.graph.nodes():
            if agent.id == node.agent.id:
                return True
        return False

    def agent_previous_opinion(self, agent: Agent) -> None:
        """
        Set the specified Agent's previous opinion to be equal to the current opinion (before the current opinion changes in the current iteration).

        :param agent: The agent whose previous opinion is being set.
        :type agent: Agent
        """
        agent_node: GraphNode | None = self.node_from_agent(agent)
        if agent_node:
            agent_node.agent.store_previous_opinion()
        return None

    def agent_opinion_change(self, agent: Agent, change_delta: float) -> None:
        """
        Changes the specified Agent's current opinion by the given delta.

        :param agent: The agent whose current opinion is being changed.
        :type agent: Agent
        :param change_delta: The value by which to change the Agent's current opinion.
        :type change_delta: float
        """
        agent_node: GraphNode | None = self.node_from_agent(agent)
        if agent_node:
            agent_node.agent.change_opinion(change_delta)
        return None

    def agent_radicalisation_change(self, agent: Agent, radicalisation: bool) -> None:
        """
        Change the specified Agent's radicalisation status.

        :param agent: The agent whose radicalisation status is being changed.
        :type agent: Agent
        :param radicalisation: The new radicalisation status.
        :type radicalisation: bool
        """
        agent_node: GraphNode | None = self.node_from_agent(agent)
        if agent_node:
            agent_node.agent.change_radicalisation(radicalisation)
        return None

    def node_from_agent(self, agent: Agent) -> GraphNode | None:
        """
        Returns the GraphNode object corresponding to the given Agent object.

        :param agent: The agent being searched for in the GraphNodes.
        :type agent: Agent
        :return: The graph node corresponding to the input Agent if it exists, or None otherwise.
        :rtype: GraphNode | None
        """
        agent_index: int | None = self.get_agent_index(agent)
        if agent_index is not None:
            agent_node: GraphNode | None = self.get_node(agent_index)
            return agent_node
        return None

    def get_agent_index(self, agent: Agent) -> int | None:
        """
        Searches for the node index in the Graph which corresponds to the input Agent object.

        :param agent: The agent whose index is being searched for.
        :type agent: Agent
        :return: The agent's node index within the social hierarchy Graph if it exists, or None otherwise.
        :rtype: int | None
        """
        for idx, node in enumerate(self.graph.nodes()):
            if agent.id == node.agent.id:
                return idx
        return None

    def get_neighbours(self, agent: Agent) -> list[GraphNode] | None:
        """
        Finds all the nodes in the graph with direct relationships to the specified Agent.

        :param agent: The agent for which the neighbours are being examined.
        :type agent: Agent
        :raises UserWarning: If the input agent does not exist in the Graph.
        :return: A list of the graph nodes belonging to the direct neighbours of the agent.
        :rtype: list[GraphNode]
        """
        neighbour_nodes: list[GraphNode] = []
        agent_index: int | None = self.get_agent_index(agent)
        if not agent_index:
            if not self.suppress_warnings:
                warnings.warn(
                    f"Input Agent does not exist in this hierarchy ({self.name})",
                    category=UserWarning,
                )
            return None
        neighbour_indices: rx.rustworkx.NodeIndices = self.graph.neighbors(agent_index)
        for index in neighbour_indices:
            neighbour_node: GraphNode | None = self.get_node(index)
            if neighbour_node is None:
                # Should never be reached and is just included for type checking purposes
                continue
            neighbour_nodes.append(neighbour_node)
        return neighbour_nodes

    def step(self) -> None:
        """
        Step the individual Graph object:
            1. Handle dynamic relationships within the graph.
        """
        self.dynamic_relationships()
        return None

    def neighbour_influences(self, agent: Agent) -> float | None:
        """
        Looks at all the neighbours for an Agent and uses the neighbours' own opinions plus the
        weight of the relationship between Agents to return a final value by which the given
        Agent's opinion value will increment or decrement.

        :param agent: The agent for which the strength of opinion change is being determined.
        :type agent: Agent
        :raises UserWarning: If the input agent does not exist in the Graph.
        :return: The final change in the Agent's opinion caused by their neighbours in this hierarchy.
        :rtype: float
        """
        agent_hierarchy_weighting: float = agent.social_weightings[self.name]
        agent_index: int | None = self.get_agent_index(agent)
        if agent_index is None:
            if not self.suppress_warnings:
                warnings.warn(
                    f"Input Agent {agent.id} does not exist in this hierarchy ({self.name})",
                    category=UserWarning,
                )
            return None
        neighbour_indices: rx.NodeIndices = self.graph.neighbors(agent_index)

        weighted_deltas: list[float] = []
        delta_weightings: list[float] = []
        for neighbour_index in neighbour_indices:
            neighbour_node: GraphNode | None = self.get_node(neighbour_index)

            if neighbour_node is None:
                # This should never be reached and is only included for type checking purposes
                continue

            relationship_strength: float = self.get_relationship(
                agent, neighbour_node.agent
            )

            average_opinion: float = (
                agent.opinion + neighbour_node.agent.opinion
            ) / 2.0  # Simple average of own and neighbour opinions
            distance_from_avg: float = (
                average_opinion - agent.opinion
            )  # The delta that must be applied to own opinion to reach the average
            weighted_delta: float = (
                distance_from_avg * agent_hierarchy_weighting * relationship_strength
            )  # The final opinion change

            # Account for neighbour radicalisation
            # (neutral personality means that the existence or lack of neighbour radicalisation will have no effect)
            relative_weighting: float = 1.0
            if (
                neighbour_node.agent.radicalised
                and agent.personality != "neutral"
                and not agent.radicalised
            ):
                if agent.personality in ["rational", "social"]:
                    # "rational" or "social" agents that are not radicalised will have a generally lesser view of radicalised opinions
                    relative_weighting = 0.5
                elif agent.personality == "impulsive":
                    # "impulsive" agents will always view radicalised opinions more favourably
                    relative_weighting = 2.0
                else:  # Agent personality is "erratic"
                    # "erratic" agents act randomly...
                    erratic_coinflip: bool = random_coinflip("bool")
                    if erratic_coinflip:
                        relative_weighting = 2.0
                    else:
                        relative_weighting = 0.5
            elif neighbour_node.agent.radicalised and agent.radicalised:
                if distance_from_avg <= 0.25:
                    # Both agents are radicalised towards the same opinion
                    relative_weighting = 4.0
                else:
                    # Both agent are radicalised in opposing opinions
                    relative_weighting = 0.25

            weighted_deltas.append(weighted_delta)
            delta_weightings.append(relative_weighting)

        # Calculate the final change
        final_change: float = 0.0
        total_weightings: float = sum(delta_weightings)
        for idx, weighted_delta in enumerate(weighted_deltas):
            final_change += weighted_delta * (delta_weightings[idx] / total_weightings)

        return final_change

    def dynamic_relationships(self) -> None:
        """
        Uses the (mean, variance) passed at initialisation to draw random walk values by which each edge (relationship)
        in the hierarchy will be shifted. Aims to simulate dynamic relationships between agents across timesteps.
        """
        if not self.dynamic_rels:
            # No need to iterate the edges...
            return None

        for edge in self.graph.edges():
            new_weighting: float

            if not edge.has_rw_params():
                new_weighting = value_rw_delta(
                    edge.weighting, self.rw_params[0], self.rw_params[1]
                )
            elif edge.rw_params is not None:
                new_weighting = value_rw_delta(
                    edge.weighting, edge.rw_params[0], edge.rw_params[1]
                )
            else:  # Default to not changing the weighting at all
                new_weighting = edge.weighting

            # Constrain the weighting back to [-1.0, 1.0] as needed
            if new_weighting < -1.0:
                new_weighting = -1.0
            elif new_weighting > 1.0:
                new_weighting = 1.0

            edge.set_weighting(new_weighting)
        return None

    def estimate_neighbour_opinions(self, agent: Agent) -> dict[str, float]:
        """
        Return the individual opinion climate values perceived by the Agent for each other Agent within this social hierarchy.

        :param agent: The agent which is estimating its neighbours' opinions.
        :type agent: Agent
        :return: A mapping of <neighbour id : estimated opinion value> for each opinion perceived by the agent in the hierarchy.
        :rtype: dict[str, float]
        """
        observed_opinions: dict[str, float] = {}

        direct_neighbours: list[GraphNode] | None = self.get_neighbours(agent)
        if direct_neighbours is not None:
            for direct_neighbour in direct_neighbours:
                observed_opinion: float = direct_neighbour.agent.opinion
                observed_opinions[direct_neighbour.agent.id] = observed_opinion
        else:
            # Set it to an empty list to simplify the `node in direct_neighbours` check below
            direct_neighbours = []

        for node in self.graph.nodes():
            if node.agent.id == agent.id or node in direct_neighbours:
                # Only look at indirect neighbours
                continue

            raw_observed_opinion: float = node.agent.opinion
            attenuated_opinion: float = beta_value_attenuation(raw_observed_opinion)

            if -0.5 > attenuated_opinion > 0.5:
                observed_opinions[node.agent.id] = raw_observed_opinion

        return observed_opinions

    def estimate_opinion_climate(self, agent: Agent) -> float:
        """
        Return the unique opinion climate perceived by the Agent within this social hierarchy.

        :param agent: The agent which is estimating the opinion climate.
        :type agent: Agent
        :return: The agent's perceived 'aggregated opinion' of this whole social hierarchy.
        :rtype: float
        """
        observed_opinions: list[
            float
        ] = []  # The observed opinions of the agent's direct neighbours and the relevant observed opinions of indirect neighbours

        direct_neighbours: list[GraphNode] | None = self.get_neighbours(agent)
        if direct_neighbours is not None:
            for direct_neighbour in direct_neighbours:
                observed_opinion: float = direct_neighbour.agent.opinion
                observed_opinions.append(observed_opinion)
        else:
            # Set it to an empty list to simplify the `node in direct_neighbours` check below
            direct_neighbours = []

        for node in self.graph.nodes():
            if node.agent.id == agent.id or node in direct_neighbours:
                # Only look at indirect neighbours
                continue

            raw_observed_opinion: float = node.agent.opinion
            attenuated_opinion: float = beta_value_attenuation(raw_observed_opinion)

            if (
                -0.5 > attenuated_opinion > 0.5
            ):  # Only take values which are still relevant after attenuation (i.e. values stronger than an absolute 0.5 after attenuation)
                observed_opinions.append(
                    raw_observed_opinion
                )  # NOT the attenuated opinion, as that would funamentally alter the nature of the opinion climate

        summed_opinions: float = sum(
            observed_opinions
        )  # Sum all of the observed opinions

        if len(observed_opinions) >= 1:
            opinion_climate: float = summed_opinions / float(
                len(observed_opinions)
            )  # Find the average of the aggregated, relevant opinions
            return opinion_climate
        else:
            return 0.0

    def calculate_polarisation(self) -> float:
        r"""
        Calculates the level of opinion polarisation in this :class:`Graph` based on the equation:

        .. math::

            \pi(k) = \frac{1}{|K|(|K| - 1)}\sum_{i \neq j}^{i \in K, j \in K}(d_{ij} - y)^{2}

        where :math:`K` is the set of agents within this Graph, :math:`d_{ij}` is the distance between
        the opinions of agents :math:`i` and :math:`j`, and :math:`y` is the mean opinion distance among
        all agents in this Graph.

        :return: The measure of opinion radicalisation in this social hierarchy.
        :rtype: float
        """
        K: int = self.node_count
        opinion_distances: dict[str, float] = {}

        for i in self.graph.nodes():
            for j in self.graph.nodes():
                if i.agent.id == j.agent.id:
                    continue
                opinion_distance: float = abs(i.agent.opinion - j.agent.opinion)
                opinion_distances[f"{i.index},{j.index}"] = opinion_distance

        y: float = sum(opinion_distances.values()) / len(opinion_distances.values())

        summation: float = 0.0
        for distance in opinion_distances.values():
            square_distance: float = (distance - y) ** 2
            summation += square_distance

        radicalisation_measure: float = summation / (K * (K - 1))
        return radicalisation_measure

    def __in__(self, iterable: Iterable[Graph]) -> bool:
        """
        Determine if the Graph is contained within the Iterable of Graphs.

        :param iterable: The collection of graph objects in which membership is being determined.
        :type iterable: Iterable[Graph]
        :return: A flag indicating if this Graph is contained within the iterable.
        :rtype: bool
        """
        for graph in iterable:
            if self.name == graph.name:
                return True
        return False

    @override
    def __str__(self) -> str:
        """
        An override of the Graph string representation when calling print().

        :return: A printable representation outlining the name and graph properties of the specific social hierarchy.
        :rtype: str
        """
        return f"Graph representing the {self.name} social hierarchy with {self.node_count} nodes and {self.edge_count} edges"


class GraphSet:
    """
    A class that will collect all of the different social hierarchy graphs in the same structure
    and provide utilities using this collection.

    :param graphs: Existing Graph objects that should be added to the GraphSet.
    :type graphs: list[Graph], optional
    """

    def __init__(self, graphs: list[Graph] | None = None) -> None:
        self.graphs: list[Graph] = []
        if graphs:
            self.graphs = graphs

    def save_graphset(self, directory_path: str) -> None:
        """
        Save all of the graphs contained within this graphset into a compressed subdirectory representing
        the saved GraphSet.

        :param directory_path: The path to the directory where the graphset subdirectory should be created.
        :type directory_path: str
        """
        # Assume that the passed directory path is to the base save path, not directly to the graphset subdirectory
        subdirectory_path: str = f"{directory_path}/_graphset"

        if os.path.isdir(subdirectory_path):
            # Remove the existing directory to allow for a new overwrite
            rmtree(subdirectory_path)

        # Create the _graphset subdirectory
        os.mkdir(subdirectory_path)

        graph_save_paths: list[str] = []
        node_save_paths: list[str] = []
        edge_save_paths: list[str] = []

        for graph in self.graphs:
            # Create a graph subdirectory
            os.mkdir(f"{subdirectory_path}/{graph.name}")

            # Save path for the specific hierarchy graph
            graph_save_path: str = (
                f"{subdirectory_path}/{graph.name}/graph_{graph.name}.graphml"
            )
            graph.save_graph(graph_save_path)
            graph_save_paths.append(graph_save_path)

            # Create a nodes subdirectory
            os.mkdir(f"{subdirectory_path}/{graph.name}/nodes")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                saved_node_paths = {executor.submit(self.write_node_pickle, node, subdirectory_path, graph.name, node.index): node.index for node in graph.graph.nodes()}
                for future in concurrent.futures.as_completed(saved_node_paths):
                    node_index = saved_node_paths[future]
                    try:
                        node_save_path = future.result()
                    except Exception as exc:
                        print(f"Failed to write a pickle for node {node_index} in graph {graph.name} with exception: {exc}")
                    else:
                        node_save_paths.append(node_save_path)

            # Create an edges subdirectory
            os.mkdir(f"{subdirectory_path}/{graph.name}/edges")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                saved_edge_paths = {executor.submit(self.write_edge_pickle, edge, subdirectory_path, graph.name, edge.index): edge.index for edge in graph.graph.edges()}
                for future in concurrent.futures.as_completed(saved_edge_paths):
                    edge_index = saved_edge_paths[future]
                    try:
                        edge_save_path = future.result()
                    except Exception as exc:
                        print(f"Failed to write a pickle for edge {edge_index} in graph {graph.name} with exception: {exc}")
                    else:
                        edge_save_paths.append(edge_save_path)

        zip_path: str = f"{subdirectory_path}.zip"

        if os.path.exists(zip_path):
            # Remove the existing zip file to allow for a new overwrite
            os.remove(zip_path)

        # Compress the subdirectory to minimise storage, and encapsulate all graphs into a single object
        with zipfile.ZipFile(
            zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=4
        ) as subdir_zip:
            # Zip the graphml files
            for graph_path in graph_save_paths:
                graph_path_components: list[str] = graph_path.split("/")
                subdir_zip.write(graph_path, arcname=f"{graph_path_components[-2]}/{graph_path_components[-1]}")

            # Zip the node pickles
            for node_path in node_save_paths:
                node_path_components: list[str] = node_path.split("/")
                subdir_zip.write(node_path, arcname=f"{node_path_components[-3]}/{node_path_components[-2]}/{node_path_components[-1]}")

            # Zip the edge pickles
            for edge_path in edge_save_paths:
                edge_path_components: list[str] = edge_path.split("/")
                subdir_zip.write(edge_path, arcname=f"{edge_path_components[-3]}/{edge_path_components[-2]}/{edge_path_components[-1]}")

        # Remove the uncompressed subdirectory if compression was successful
        if os.path.exists(zip_path):
            rmtree(subdirectory_path)

        return None

    def write_node_pickle(self, graph_node: GraphNode, subdirectory_path: str, graph_name: str, idx: int) -> str:
        """
        A helper function that allows for multithreading of :meth:`~gatoh.graphs.graphs.GraphSet.save_graphset`.

        :param graph_node: The graph node which is being pickled.
        :type graph_node: GraphNode
        :param subdirectory_path: The subdirectory to which the pickled graph node is being written to.
        :type subdirectory_path: str
        :param graph_name: The name of the graph that the node belongs to.
        :type graph_name: str
        :param idx: The index of the node within its parent graph.
        :type idx: int
        :return: The path to which the node pickle was saved to.
        :rtype: str
        """
        node_save_path: str = f"{subdirectory_path}/{graph_name}/nodes/node_{idx}.pkl"
        with open(node_save_path, "wb") as node_pickle:
            pickle.dump(graph_node, node_pickle)
        return node_save_path

    def write_edge_pickle(self, graph_edge: GraphEdge, subdirectory_path: str, graph_name: str, idx: int) -> str:
        """
        A helper function that allows for multithreading of :meth:`~gatoh.graphs.graphs.GraphSet.save_graphset`.

        :param graph_edge: The graph edge which is being pickled.
        :type graph_edge: GraphEdge
        :param subdirectory_path: The subdirectory to which the pickled graph edge is being written to.
        :type subdirectory_path: str
        :param graph_name: The name of the graph that the edge belongs to.
        :type graph_name: str
        :param idx: The index of the edge within its parent graph.
        :type idx: int
        :return: The path to which the edge pickle was saved to.
        :rtype: str
        """
        edge_save_path: str = f"{subdirectory_path}/{graph_name}/edges/edge_{idx}.pkl"
        with open(edge_save_path, "wb") as edge_pickle:
            pickle.dump(graph_edge, edge_pickle)
        return edge_save_path

    def load_graphset(
        self, load_path: str, rw_params: dict[str, tuple[float, float]], worker_pool: Pool | None = None,
    ) -> None:
        """
        Loads a GraphSet that has been saved following the same process as in the save_graphset() function.

        :param load_path: The path to the model's overall save directory.
        :type load_path: str
        :param rw_params: A <name : rw_params> mapping containing the relevant external information for each graph.
        :type rw_params: dict[str, tuple[float, float]]
        :param worker_pool: A pool of workers that can distribute the processing of the graphset loading amongst themselves.
        :type worker_pool: :class:`~multiprocessing.pool.Pool`, optional
        """
        zip_load_path: str = f"{load_path}/_graphset.zip"

        if not os.path.exists(zip_load_path):
            raise FileNotFoundError(
                f"No saved GraphSet was found at the path: {zip_load_path}"
            )

        # The path to the uncompressed subdirectory
        subdirectory_path: str = f"{load_path}/_graphset"

        # Remove any existing subdirectory with the same name to replace it with the newly loaded one
        if os.path.isdir(subdirectory_path):
            rmtree(subdirectory_path)

        # Create the uncompressed subdirectory
        os.mkdir(subdirectory_path)

        # Extract all the graphml files to the uncompressed subdirectory
        with zipfile.ZipFile(
            zip_load_path, mode="r", compression=zipfile.ZIP_DEFLATED, compresslevel=4
        ) as subdir_zip:
            subdir_zip.extractall(path=subdirectory_path)

        save_dirs: list[str] = list(os.walk(subdirectory_path))[0][1]

        if worker_pool is not None:
            loaded_graphs = worker_pool.starmap(self.load_graphset_multi, [(save_dir, subdirectory_path, rw_params[os.path.basename(save_dir)]) for save_dir in save_dirs])
            for loaded_graph in loaded_graphs:
                self.add_graph(loaded_graph)

            # Manual garbage collection
            del loaded_graphs
            _ = gc.collect()
        else:
            for save_dir in save_dirs:
                loaded_graph: Graph = self.load_graphset_multi(save_dir, subdirectory_path, rw_params[os.path.basename(save_dir)])
                self.add_graph(loaded_graph)

                # Manual garbage collection
                del loaded_graph
                _ = gc.collect()
        return None

    def load_graphset_multi(self, save_dir: str, subdirectory_path: str, rw_params: tuple[float, float]) -> Graph:
        """
        A helper function that allows for parallel processing of Graph loading for :meth:`~gatoh.graphs.graphs.GraphSet.load_graphset`.

        :param save_dir: The path of the directory to which all of a graph's files have been saved to.
        :type save_dir: str
        :param subdirectory_path: The path to the subdirectory in which the graph's save directory is located.
        :type subdirectory_path: str
        :param rw_params: The (mean, variance) random-walk parameters for the graph that is being loaded.
        :type rw_params: tuple[float, float]
        :return: A loaded graph with all included nodes and edges.
        :rtype: Graph
        """
        graph_name: str = os.path.basename(save_dir)

        graphml_path: str = f"{subdirectory_path}/{graph_name}/graph_{graph_name}.graphml"

        new_graph: Graph = Graph("", (0.0, 0.0))
        new_graph.load_graph(graphml_path, graph_name, rw_params)

        node_dir: str = f"{subdirectory_path}/{graph_name}/nodes"
        edge_dir: str = f"{subdirectory_path}/{graph_name}/edges"

        node_files: list[str] = list(os.walk(node_dir))[0][2]
        edge_files: list[str] = list(os.walk(edge_dir))[0][2]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            loaded_nodes = {executor.submit(self.load_node, node_dir, node_file): node_file for node_file in node_files}
            for future in concurrent.futures.as_completed(loaded_nodes):
                node_file = loaded_nodes[future]
                try:
                    node_info = future.result()
                except Exception as exc:
                    print(f"Failed to extract the pickled node at file {node_file} for graph {graph_name} with exception: {exc}")
                else:
                    new_graph.graph[node_info[1]] = node_info[0]

            loaded_edges = {executor.submit(self.load_edge, edge_dir, edge_file): edge_file for edge_file in edge_files}
            for future in concurrent.futures.as_completed(loaded_edges):
                edge_file = loaded_edges[future]
                try:
                    edge_info = future.result()
                except Exception as exc:
                    print(f"Failed to extract the pickled edge at file {edge_file} for graph {graph_name} with exception: {exc}")
                else:
                    new_graph.graph.update_edge_by_index(edge_info[1], edge_info[0])
        return new_graph

    def load_node(self, node_dir: str, node_file: str) -> tuple[GraphNode, int]:
        """
        A helper function that allows for multithreading within :meth:`~gatoh.graphs.graphs.GraphSet.load_graphset_multi`.

        :param node_dir: The root directory where all graph nodes have been saved.
        :type node_dir: str
        :param node_file: The name of the pickle file that is being loaded.
        :type node_file: str
        :return: The unpickled graph node object and its corresponding index in the graph.
        :rtype: tuple[GraphNode, int]
        """
        node_index: int = int((os.path.basename(node_file).split("_")[-1]).split(".")[0])
        with open(f"{node_dir}/{node_file}", "rb") as node_pickle:
            node_object: GraphNode = pickle.load(node_pickle)
        return (node_object, node_index)

    def load_edge(self, edge_dir: str, edge_file: str) -> tuple[GraphEdge, int]:
        """
        A helper function that allows for multithreading within :meth:`~gatoh.graphs.graphs.GraphSet.load_graphset_multi`.

        :param edge_dir: The root directory where all graph edges have been saved.
        :type edge_dir: str
        :param edge_file: The name of the pickle file that is being loaded.
        :type edge_file: str
        :return: The unpickled graph edge object and its corresponding index in the graph.
        :rtype: tuple[GraphEdge, int]
        """
        edge_index: int = int((os.path.basename(edge_file).split("_")[-1]).split(".")[0])
        with open(f"{edge_dir}/{edge_file}", "rb") as edge_pickle:
            edge_object: GraphEdge = pickle.load(edge_pickle)
        return (edge_object, edge_index)

    def add_graph(self, graph: Graph) -> None:
        """
        A setter function to add a new Graph object to the GraphSet.

        :param graph: The graph to add to the GraphSet.
        :type graph: Graph
        """
        self.graphs.append(graph)
        return None

    def graph_at_index(self, graph_index: int) -> Graph | None:
        """
        A getter function to return a Graph object stored at the given index in the GraphSet.

        :param graph_index: The index of the graph to return.
        :type graph_index: int
        :return: The graph object to return if it exists, or None otherwise.
        :rtype: Graph | None
        """
        try:
            return self.graphs[graph_index]
        except IndexError:
            print(
                f"Index {graph_index} is out of bounds for the GraphSet. Only {len(self.graphs)} social hierarchies have been created."
            )
            return None

    def get_hierarchy(self, hierarchy: str) -> Graph | None:
        """
        A getter function to return a Graph object with the given hierarchy name.

        :param hierarchy: The name of the social hierarchy represented by the Graph to return.
        :type hierarchy: str
        :return: The Graph object of the specified hierarchy, or None if no matching hierarchy was found.
        :rtype: Graph | None
        """
        for graph in self.graphs:
            if graph.name == hierarchy:
                return graph

        print(f"No graph representing the social hierarchy '{hierarchy}' was found...")
        return None

    def get_index(self, hierarchy: str) -> int:
        """
        A getter function that returns the index of a given hierarchy within the GraphSet.

        :param hierarchy: The name of the hierarchy that is being searched for.
        :type hierarchy: str
        :raises KeyError: If the input hierarchy does not exist within the GraphSet.
        :return: The index of the hierarchy within the GraphSet.
        :rtype: int
        """
        for idx, graph in enumerate(self.graphs):
            if graph.name == hierarchy:
                return idx

        raise KeyError(
            f"The social hierarchy '{hierarchy}' does not exist in the GraphSet -- cannot return an index."
        )

    def list_hierarchies(self, print_out: bool = False) -> list[str]:
        """
        A utility function that iterates over the GraphSet and prints out the names of all the social hierarchies that are present.

        :param print_out: A flag indicating if the listed hierarchies should be printed to the terminal.
        :type print_out: bool, optional
        :return: The names of all social hierarchies present in the GraphSet.
        :rtype: list[str]
        """
        social_hierarchies: list[str] = []
        for graph in self.graphs:
            social_hierarchies.append(graph.name)

        if print_out:
            print(
                f"\nSocial hierarchies present in the GraphSet:\n\t{social_hierarchies}\n\n"
            )

        return social_hierarchies

    def get_agent_hierarchies(self, agent: Agent) -> list[str]:
        """
        A helper function that determines which social hierarchies an Agent is contained in.

        :param agent: The agent for which hierarchy memberships are being determined.
        :type agent: Agent
        :return: The names of all the social hierarchies to which the input Agent belongs in.
        :rtype: list[str]
        """
        member_of: list[str] = []
        for hierarchy in self.graphs:
            if hierarchy.agent_in_graph(agent):
                member_of.append(hierarchy.name)
        return member_of

    def calculate_polarisation(self, hierarchy: str) -> float:
        """
        A wrapper that calls a specific hierarchy graph's calculate_polarisation function and returns its value.

        :param hierarchy: The name of the hierarchy for which polarisation is being calculated.
        :type hierarchy: str
        :raises ValueError: If the input hierarchy does not exist or otherwise cannot be retrieved.
        :return: The hierarchy polarisation value.
        :rtype: float
        """
        hierarchy_graph: Graph | None = self.get_hierarchy(hierarchy)
        if hierarchy_graph is not None:
            return hierarchy_graph.calculate_polarisation()
        else:
            raise ValueError(f"Tried to calculate polarisation for an invalid hierarchy: {hierarchy}")

    def agent_opinion_threshold(
        self, agent: Agent, threshold: float = 0.9
    ) -> Iterable[str]:
        """
        A utility function that iterates over the GraphSet and records for which social hierarchies a specific Agent's weighting
        of those hierarchies is above a certain threshold value.

        :param agent: The agent for which to check the AgentSet for.
        :type agent: Agent
        :param threshold: The absolute threshold value over which the Agent's opinion is considered significant.
        :type threshold: float, optional
        :return: The names of the hierarchies for which the agent's weighting is above the threshold.
        :rtype: Iterable[str]
        """
        significant_hierarchies: Iterable[str] = []
        for hierarchy in self.graphs:
            if hierarchy.agent_in_graph(agent):
                social_weighting: float = agent.social_weightings[hierarchy.name]
                if abs(social_weighting) > threshold:
                    significant_hierarchies.append(hierarchy.name)
        return significant_hierarchies

    def __in__(self, graph: Graph) -> bool:
        """
        A method defining how a GraphSet checks for Graph membership.

        :param graph: The graph whose membership is being checked for.
        :type graph: Graph
        :return: A flag indicating if the Graph object is contained in self.graphs.
        :rtype: bool
        """
        return graph in self.graphs

    def __contains__(self, graph: Graph) -> bool:
        """
        A secondary method defining how a GraphSet checks for Graph membership.

        :param graph: The graph whose membership is being checked for.
        :type graph: Graph
        :return: A flag indicating if the Graph object is contained in self.graphs.
        :rtype: bool
        """
        return self.graphs.__contains__(graph)

    def __len__(self) -> int:
        """
        A method defining how a GraphSet checks its length.

        :return: The number of Graph objects contained within the GraphSet.
        :rtype: int
        """
        return len(self.graphs)

    def __iter__(self) -> Iterator[Graph]:
        """
        A method defining how a GraphSet iterates over the Graphs contained within it.

        :return: The Graphs in the GraphSet.
        :rtype: Iterator[Graph]
        """
        return self.graphs.__iter__()

    @override
    def __str__(self) -> str:
        """
        An override of what calling `print()` on this object will output.

        :return: A printable representation listing the names of the hierarchies which are contained in the GraphSet.
        :rtype: str
        """
        return f"GraphSet containing the graphs of the following social hierarchies:\n\n{self.list_hierarchies()}"
