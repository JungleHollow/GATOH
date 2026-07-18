"""
This utils file should cover any miscellaneous functions that facilitate running the package across modules
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TypeVar, cast

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gatoh.model import ConfigData
    from yaml import SequenceNode

import matplotlib.pyplot as plt
import numpy as np
import rustworkx as rx
import yaml
from multimethod import multimethod
from scipy.stats import beta, gamma, levy, norm, truncnorm, uniform


T = TypeVar("T", bool, str, int, float)

# ========== Graph utils ========== #


@dataclass
class NodeChanges:
    """
    Dataclass that stores all the relevant information needed to update GraphNodes in the model's base graph.

    :param opinion: The newest opinion value that the agent in the node should have.
    :type opinion: float
    :param previous_opinion: The newest previous_opinion value that the agent in the node should have.
    :type previous_opinion: float
    :param radicalised: The newest radicalisation status that the agent in the node should have.
    :type radicalised: bool
    :param social_weightings: The newest social weightings that the agent in the node should have.
    :type social_weightings: dict[str, float]
    """

    opinion: float
    previous_opinion: float
    radicalised: bool
    social_weightings: dict[str, float]

    def __init__(
        self,
        opinion: float,
        previous_opinion: float,
        radicalised: bool,
        social_weightings: dict[str, float],
    ) -> None:
        self.opinion = opinion
        self.previous_opinion = previous_opinion
        self.radicalised = radicalised
        self.social_weightings = social_weightings


@dataclass
class EdgeChanges:
    """
    Dataclass that stores all the relevant information needed to update GraphEdges in the model's base graph.

    :param hierarchy: The name of the hierarchy to which the relationships belongs.
    :type hierarchy: str
    :param weighting: The newest weighting value that the relationship should have.
    :type weighting: float
    """

    hierarchy: str
    weighting: float

    def __init__(self, hierarchy: str, weighting: float) -> None:
        self.hierarchy = hierarchy
        self.weighting = weighting


def pygraph_to_pydigraph(input_graph: rx.PyGraph) -> rx.PyDiGraph:
    """
    Transform an arbitrary :py:class:`rustworxk.PyGraph` to a :py:class:`rustworkx.PyDiGraph` object where each unidirectional edge in the :py:class:`PyGraph` becomes two opposing monodirectional edges with weight equal to the original.
    Mainly used exclusively to transform undirected :py:func:`watts_strogatz_graph` returns to directed ones for use in the model.

    :param input_graph: The undirected input graph to be transformed into a directed version.
    :type input_graph: :py:class:`rustworkx.PyGraph`
    :return: A directed version of the input graph where each edge has been transformed into two edges pointing in opposing directions.
    :rtype: :py:class:`rustworkx.PyDiGraph`
    """
    new_graph: rx.PyDiGraph = rx.PyDiGraph()
    for node in input_graph.nodes():
        _ = new_graph.add_node(node)

    for edge in input_graph.weighted_edge_list():
        # 'edge' is a (node_a_index, node_b_index, weight) tuple
        # Given that multiple edges were not allowed at creation, each combination of a and b should be unique
        _ = new_graph.add_edge(edge[0], edge[1], edge[2])  # Edge going from a -> b
        _ = new_graph.add_edge(edge[1], edge[0], edge[2])  # Edge going from b -> a
    return new_graph


@multimethod
def watts_strogatz_graph(n: int, k: int, p: float, seed: int) -> rx.PyGraph:
    """
    Returns an undirected Watts-Strogatz small-world graph generated using :py:mod:`rustworkx`.
    An adapted version of :py:func:`~networkx.generators.random_graphs.watts_strogatz_graph` from the :py:mod:`NetworkX` library.

    :param n: The number of nodes in the graph.
    :type n: int
    :param k: The number of nearest neighbours that each node is joined to initially.
    :type k: int
    :param p: The probability of rewiring each edge of the original ring lattice.
    :type p: float
    :param seed: The random seed to use for random generation.
    :type seed: int
    :return: An undirected Watts-Strogatz small-world graph that has been generated.
    :rtype: :py:class:`rustworkx.PyGraph`
    """
    random_gen: random.Random = random.Random(seed)
    created_graph: rx.PyGraph = watts_strogatz_creator(n, k, p, random_gen)
    return created_graph


@multimethod
def watts_strogatz_graph(
    n: int, k: int, p: float, seed: np.random.RandomState
) -> rx.PyGraph:
    """
    Returns an undirected Watts-Strogatz small-world graph generated using :py:mod:`rustworkx`.
    An adapted version of :py:func:`~networkx.generators.random_graphs.watts_strogatz_graph` from the :py:mod:`NetworkX` library.

    :param n: The number of nodes in the graph.
    :type n: int
    :param k: The number of nearest neighbours that each node is joined to initially.
    :type k: int
    :param p: The probability of rewiring each edge of the original ring lattice.
    :type p: float
    :param seed: The random seed to use for random generation.
    :type seed: :py:class:`~numpy.random.RandomState`
    :return: The created graph.
    :rtype: :py:class:`rustworkx.PyGraph`
    """
    random_gen: np.random.RandomState = seed
    created_graph: rx.PyGraph = watts_strogatz_creator(n, k, p, random_gen)
    return created_graph


@multimethod
def watts_strogatz_graph(n: int, k: int, p: float, seed: None) -> rx.PyGraph:
    """
    Returns an undirected Watts-Strogatz small-world graph generated using :py:mod:`rustworkx`.
    An adapted version of :py:func:`~networkx.generators.random_graphs.watts_strogatz_graph` from the :py:mod:`NetworkX` library.

    :param n: The number of nodes in the graph.
    :type n: int
    :param k: The number of nearest neighbours that each node is joined to initially.
    :type k: int
    :param p: The probability of rewiring each edge of the original ring lattice.
    :type p: float
    :return: The created graph.
    :rtype: :py:class:`rustworkx.PyGraph`
    """
    random_gen: random.Random = random.Random()
    created_graph: rx.PyGraph = watts_strogatz_creator(n, k, p, random_gen)
    return created_graph


def watts_strogatz_creator(
    n: int,
    k: int,
    p: float,
    random_gen: random.Random | np.random.RandomState,
) -> rx.PyGraph:
    """
    A helper function that performs the actual Graph object creation for the multidispatched functions above.

    :param n: The number of nodes in the graph.
    :type n: int
    :param k: The number of nearest neighbours that each node is joined to initially.
    :type k: int
    :param p: The probability of rewiring each edge of the original ring lattice.
    :type p: float
    :param random_gen: The random generator to use for Graph creation.
    :type random_gen: :py:class:`random.Random` | :py:class:`numpy.random.RandomState`
    :raises ValueError: If k is equal to or greater than n.
    :return: The created graph.
    :rtype: :py:class:`rustworkx.PyGraph`
    """
    if k >= n:
        # >= instead of == as this utility function does not care about accounting for complete graphs...
        raise ValueError(
            "k is larger than or equal to n; choose a smaller k or larger n."
        )

    G: rx.PyGraph = rx.PyGraph()
    nodes: list[int] = list(range(n))  # nodes labeled 0 to n-1

    _ = G.add_nodes_from(
        nodes
    )  # Add the index-labelled nodes for now (GraphNode objects will replace these in the calling module)

    # Connect each node to k/2 neighbours
    for j in range(1, k // 2 + 1):
        targets: list[int] = (
            nodes[j:] + nodes[0:j]
        )  # first j nodes become last in the list

        weightings: list[float] = [0.0 for _ in range(len(nodes))]
        edges_info: list[tuple[int, int, float]] = list(zip(nodes, targets, weightings))
        _ = G.add_edges_from(edges_info)

        # Manual garbage collection
        del weightings, edges_info

    # Rewire edges from each node
    # Loop over all nodes in order (label) and neighbours in order (distance)
    # No self loops or multiple edges allowed
    for j in range(1, k // 2 + 1):  # Outer loop is neighbours
        targets = nodes[j:] + nodes[0:j]
        # Inner loop in noder order
        for u, v in zip(nodes, targets):
            if random_gen.random() < p:
                w = random_gen.choice(nodes)
                # Enforce no self loops or multiple edges
                while w == u or G.has_edge(u, w):
                    w = random_gen.choice(nodes)
                    if G.degree(u) >= n - 1:
                        break  # Skip this rewiring
                else:
                    G.remove_edge(u, v)
                    _ = G.add_edge(u, w, 0.0)
    return G


def connected_watts_strogatz_graph(
    n: int,
    k: int,
    p: float,
    tries: int = 100,
    seed: int | np.random.RandomState | None = None,
) -> rx.PyDiGraph:
    """
    Returns a connected, directed Watts-Strogatz small-world graph.
    An adapted version of :py:func:`~networkx.generators.random_graph.connected_watts_strogatz_graph` from the :py:mod:`NetworkX` library.

    :param n: The number of nodes in the graph.
    :type n: int
    :param k: The number of nearest neighbours that each node is joined to initially.
    :type k: int
    :param p: The probability of rewiring each edge of the original ring lattice.
    :type p: float
    :param tries: The number of times to try producing a connected graph after rewiring, before raising an exception.
    :type tries: int, optional
    :param seed: The random seed to use for random generation.
    :type seed: int | :py:class:`numpy.random.RandomState`, optional
    :raises RuntimeError: If the function reaches the maximum number of attempts without producing a connected graph.
    :return: The created graph, which is assured to be connected.
    :rtype: :py:class:`rustworkx.PyDiGraph`
    """
    for _ in range(tries):
        graph: rx.PyGraph
        if seed is not None:
            graph = watts_strogatz_graph(n, k, p, seed)
        else:
            graph = watts_strogatz_graph(n, k, p)
        if rx.is_connected(graph):
            directed_graph: rx.PyDiGraph = pygraph_to_pydigraph(graph)
            return directed_graph
    raise RuntimeError(
        "Exceeded maximum number of tries generating a connected Watts-Strogatz small-world graph..."
    )


# ========== Math utils ========== #


def beta_value_attenuation(input_value: float, a: float = 0.9, b: float = 0.9) -> float:
    """
    Takes an input value and rescales it using a beta distribution. This function is intended to be used exclusively
    for the attenuation of indirect neighbouring opinions when an agent is estimating an opinion climate in a hierarchy.

    Importantly, the output values are not used for constructing the opinion climate, instead they serve as a threshold
    used to decide if the original opinion values are included or not.

    a and b should generally be equal to each other and less than 1.0 to approximate a binomial distribution whose PDF still has
    non-zero values in the range (0, 1). This is to enable modeling of the tendency for opinions to become polarised over time;
    with this distribution meaning that agents will place much higher importance on extreme observed opinions and less importance
    on moderate opinions.

    :param input_value: The value to be attenuated. Should always be in the range [-1, 1].
    :type input_value: float
    :param a: The alpha parameter for the beta distribution.
    :type a: float, optional
    :param b: The beta parameter for the beta distribution.
    :type b: float, optional
    :return: The attenuated input value.
    :rtype: float
    """
    original_opinion: float = input_value

    # Shift the range of input_value from [-1, 1] to [0, 1]
    input_value = (input_value / 2.0) + 0.5

    # Constrain to (0, 1) to prevent the beta pdf from reaching infinity
    if input_value == 0.0:
        input_value = 0.001
    elif input_value == 1.0:
        input_value = 0.999

    # Define the beta function and find the upper bound of its PDF
    beta_func = beta(a, b)
    upper_bound: float = beta_func.pdf(0.001)

    # Calculate the beta PDF of the input value
    beta_value: float = beta_func.pdf(input_value)

    # Normalise the beta value using the upper bound of the PDF
    attenuation_factor: float = beta_value / upper_bound

    # Rescale the original value by the attenuation factor
    attenuated_opinion: float = original_opinion * attenuation_factor

    # Check if range has to be constrained (float operations)
    if attenuated_opinion < -1.0:
        attenuated_opinion = -1.0
    elif attenuated_opinion > 1.0:
        attenuated_opinion = 1.0

    return attenuated_opinion


# ========== Random Generation Utils ==========


def draw_random_value(
    distribution: str, parameters: dict[str, float] | None = None
) -> float:
    """
    Utility function that handles random value generation from multiple distributions in the same function.

    All values generated by this function will be in the range [0, 1], with any necessary scaling ocurring in
    the calling functions.

    Note, "scale" and "loc" parameters should still be included as their default values (0 for loc, and 1 for scale)
    when calling with parameters to prevent dictionary key errors.

    :param distribution: The name of the distribution to draw from.
    :type distribution: str
    :param parameters: A <parameter, value> mapping that contains any relevant parameters to be specified for a given distribution.
    :type parameters: dict[str, float]
    :raises ValueError: If the input distribution is not valid or unsupported.
    :return: A value drawn from the random distribution.
    :rtype: float
    """
    drawn_value: float = 0.0

    match distribution:
        case "gaussian":
            if parameters:
                drawn_value = truncnorm.rvs(
                    parameters["a"],
                    parameters["b"],
                    loc=parameters["loc"],
                    scale=parameters["scale"],
                )
            else:
                drawn_value = truncnorm.rvs(0.0, 1.0)
        case "beta":
            if parameters:
                drawn_value = beta.rvs(
                    parameters["a"],
                    parameters["b"],
                    loc=parameters["loc"],
                    scale=parameters["scale"],
                )
            else:
                drawn_value = beta.rvs(1.0, 1.0)
        case "levy":
            if parameters:
                drawn_value = float(levy.rvs(loc=parameters["loc"], scale=parameters["scale"]))
            else:
                drawn_value = float(levy.rvs())
        case "uniform":
            if parameters:
                drawn_value = float(uniform.rvs(
                    loc=parameters["loc"], scale=parameters["scale"]
                ))
            else:
                drawn_value = float(uniform.rvs())
        case "gamma":
            if parameters:
                drawn_value = gamma.rvs(
                    parameters["a"], loc=parameters["loc"], scale=parameters["scale"]
                )
            else:
                drawn_value = gamma.rvs(1.0)
        case _:
            raise ValueError(
                f"The given distribution ({distribution}) does not match any valid implemented types."
            )
    return drawn_value


def random_coinflip(return_type: str) -> T:
    """
    Simulates a random coinflip, returning the result as either a boolean, an integer, a float, or a string.

    The exact return type strings supported are:
        - "bool"
        - "int"
        - "float"
        - "string"

    :param return_type: The data type of the returned coinflip result.
    :type return_type: str
    :return: The outcome of the random coinflip.
    :rtype: bool | int | float | str
    """
    coinflip_result: bool = random.choices([True, False], k=1)[0]

    match return_type:
        case "bool":
            return cast(T, coinflip_result)
        case "int":
            if coinflip_result:
                return cast(T, 1)
            return cast(T, 0)
        case "float":
            if coinflip_result:
                return cast(T, 1.0)
            return cast(T, 0.0)
        case "string":
            if coinflip_result:
                return cast(T, "yes")
            return cast(T, "no")
        case _:
            return cast(T, coinflip_result)  # Defaults to boolean if no valid input type was passed.


# ========== Random Walk Utils ==========


def value_rw_delta(input_value: float, mean: float, variance: float) -> float:
    """
    Draws a random walk delta from a normal distribution with the specified parameters, and then adds this delta
    to the input value before returning.

    :param input_value: The value from which the random walk begins.
    :type input_value: float
    :param mean: The mean of the normal distribution from which the delta is drawn.
    :type mean: float
    :param variance: The variance of the normal distribution from which the delta is drawn.
    :type variance: float
    :return: The result of the random walk.
    :rtype: float
    """
    rw_delta: float = float(norm.rvs(loc=mean, scale=variance))
    rw_result: float = input_value + rw_delta
    return rw_result


# ========== Data Persistence Utils ==========


class YamlLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node: SequenceNode):
        """
        Adds the ability to load Python tuples whilst maintaining safe_load functionality.
        """
        return tuple(self.construct_sequence(node))


YamlLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", YamlLoader.construct_python_tuple
)


def create_config_file(save_path: str, config_data: ConfigData) -> None:
    """
    Creates a structured config file from the input config data, and then saves it to the specified path.

    :param save_path: The path in which to save the config file.
    :type save_path: str
    :param config_data: A <name : value> mapping specifying the values of specific parameters to be stored in the config file.
    :type config_data: dict
    """
    with open(save_path, "w") as config_file:
        yaml.dump(config_data, config_file)

    return None


# ========== Visualisation and Graphing Utils ==========


def plot_graph(
    x_vals: dict[str, list[int | float]],
    y_vals: dict[str, list[int | float]],
    plot_type: str = "line",
    show_fig: bool = False,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    save_path: str | None = None,
    vertical_x: int | None = None,
    vertical_name: str | None = None,
    horizontal_y: int | None = None,
    horizontal_name: str | None = None,
) -> None:
    """
    A helper function that handles 2D plotting of data point, with possible separation by categories.

    :param x_vals: A <category : values> mapping that defines the values for the x-axis.
    :type x_vals: dict[str, list]
    :param y_vals: A <category : values> mapping that defines the values for the y-axis.
    :type y_vals: dict[str, list]
    :param plot_type: The type of graph to plot.
    :type plot_type: str, optional
    :param show_fig: A flag indicating if the graph should be displayed in the script output.
    :type show_fig: bool, optional
    :param x_label: The label to give to the x-axis.
    :type x_label: str, optional
    :param y_label: The label to give to the y-axis.
    :type y_label: str, optional
    :param title: The title to give to the graph.
    :type title: str, optional
    :param save_path: The path to save the graph image to.
    :type save_path: str, optional
    :param vertical_x: An X-axis value specifying where a vertical line should be added to the graph.
    :type vertical_x: int, optional
    :param vertical_name: A label for the graph's vertical line.
    :type vertical_name: str, optional
    :param horizontal_y: A Y-axis value specifying where a horizontal line should be added to the graph.
    :type horizontal_y: int, optional
    :param horizontal_name: A label for the graph's horizontal line.
    :type horizontal_name: str, optional
    :raises NotImplementedError: If the input plot_type is invalid or not currently supported.
    """
    fig, ax = plt.subplots()

    for key in x_vals.keys():
        current_x: list[int | float] = x_vals[key]
        current_y: list[int | float] = y_vals[key]

        match plot_type:
            case "line":
                # If "Average" is a key for a line graph, plot it using a different style and distinct colour
                if key == "Average":
                    _ = ax.plot(current_x, current_y, "--k", linewidth=0.8, label=key)
                else:
                    _ = ax.plot(current_x, current_y, linewidth=0.8, label=key)
            case "scatter":
                _ = ax.scatter(current_x, current_y, label=key)
            case "bar":
                _ = ax.bar(current_x, current_y, label=key)
            case _:
                raise NotImplementedError(
                    "Currently only 'line', 'scatter', and 'bar' graphs are supported by this function."
                )

    if vertical_x:
        if vertical_name:
            _ = ax.axvline(
                x=vertical_x, color="r", ls="--", linewidth=0.8, label=vertical_name
            )
        else:
            _ = ax.axvline(x=vertical_x, color="r", ls="--", linewidth=0.8)
    if horizontal_y:
        if horizontal_name:
            _ = ax.axhline(
                y=horizontal_y, color="r", ls="--", linewidth=0.8, label=horizontal_name
            )
        else:
            _ = ax.axhline(y=horizontal_y, color="r", ls="--", linewidth=0.8)

    ax.legend()

    if x_label:
        _ = ax.set_xlabel(x_label)
    if y_label:
        _ = ax.set_ylabel(y_label)
    if title:
        _ = ax.set_title(title)
    if save_path:
        plt.savefig(save_path, dpi=300.0)
        print(f"Plotted graph successfully saved to path: {save_path}")
    if show_fig:
        plt.show()

    return None
