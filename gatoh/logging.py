from __future__ import annotations

import csv
import tracemalloc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoggerDevStats:
    """
    Dataclass that defines the relevant developer statistics that are tracked and stored by the Logger.

    :param max_iterations: The maximum number of iterations that the model will run its simulation for.
    :type max_iterations: int
    """

    # The maximum iterations that the simulation will run for
    max_iterations: int
    # Model memory usage at the start of each iteration
    current_memory_usage: list[int] = field(default_factory=list)
    # The highest reported memory usage during model runtime
    peak_memory_usage: int = 0
    # Total number of function calls throughout the entire runtime
    all_function_calls: int = 0
    # Total number of calls to functions throughout the runtime
    function_calls: dict[str, int] = field(default_factory=dict)
    # Total computational runtime of each function per iteration
    functions_runtime: dict[int, dict[str, float]] = field(default_factory=dict)
    # Total computational runtime of each iteration
    iteration_runtime: list[float] = field(default_factory=list)
    # The current iteration that the simulation is at
    current_iteration: int = 0
    # The total number of I/O writes performed during runtime
    write_operations: int = 0
    # The total number of I/O reads performed during runtime
    read_operations: int = 0
    # The total runtime, calculated by summing the individual iteration runtimes
    total_runtime: float = 0.0

    def __init__(self, max_iterations: int) -> None:
        """
        Store the number of max iterations and initialise all lists and dictionaries with the approrpiate
        sizes to match the number of iterations.
        """
        # Start tracing memory usage
        tracemalloc.start()

        self.max_iterations  = max_iterations
        self.current_iteration  = 0

        self.peak_memory_usage = 0
        self.current_memory_usage = [0 for _ in range(self.max_iterations)]

        self.all_function_calls = 0
        self.function_calls = {}

        self.iteration_runtime = [0 for _ in range(self.max_iterations)]
        self.functions_runtime = {}
        self.total_runtime = 0.0

        for iteration in range(self.max_iterations):
            self.functions_runtime[iteration] = {}

    def record_peak_mem_usage(self, memory_usage: int) -> None:
        """
        A setter function that checks if the input memory usage is higher than the current recorded peak,
        and updates the stored value as needed.

        :param memory_usage: The memory usage value that is being checked (in bytes).
        :type memory_usage: int
        """
        self.peak_memory_usage = max(memory_usage, self.peak_memory_usage)
        self.log_function_call("LoggerDevStats.record_peak_mem_usage")
        return None

    def log_function_call(self, function_name: str) -> None:
        """
        A setter function that increments the count of calls to a specific function.

        :param function_name: The name of the function that was called.
        :type function_name: str
        """
        if function_name not in self.function_calls:
            self.function_calls[function_name] = 1
        else:
            self.function_calls[function_name] += 1
        # Also update the overall number of model function calls
        self.increment_function_calls()
        return None

    def log_function_runtime(self, function_name: str, runtime: float, iteration: int) -> None:
        """
        A setter function that updates the total runtime for a specific function.

        :param function_name: The name of the function that finished running.
        :type function_name: str
        :param runtime: The time in seconds that the function ran for.
        :type runtime: float
        :param iteration: The iteration that the runtime is being logged for.
        :type iteration: int
        """
        # Ensure that the function name exists in functions_runtime for this iteration
        _ = self.functions_runtime[iteration].setdefault(function_name, 0.0)
        self.functions_runtime[iteration][function_name] += runtime
        self.log_function_call("LoggerDevStats.log_function_runtime")
        return None

    def increment_function_calls(self) -> None:
        """
        A setter function that increments the count of overall model function calls.
        """
        self.all_function_calls += 1
        return None

    def increment_writes(self) -> None:
        """
        A setter function that increments the count of the total write operations.
        """
        self.write_operations += 1
        self.log_function_call("LoggerDevStats.increment_writes")
        return None

    def increment_reads(self) -> None:
        """
        A setter function that increments the count of total write operations.
        """
        self.read_operations += 1
        self.log_function_call("LoggerDevStats.increment_reads")
        return None

    def set_memory_usage(self, memory_usage: int, iteration: int) -> None:
        """
        A setter function that records the total model memory usage at the start of a specific iteration.

        :param memory_usage: The total memory usage of the model at the start of an iteration (in bytes).
        :type memory_usage: int
        :param iteration: The iteration that the memory usage is being recorded for.
        :type iteration: int
        """
        self.current_memory_usage[iteration] = memory_usage

        # Check if this is a new peak usage value and handle accordingly
        self.record_peak_mem_usage(memory_usage)
        self.log_function_call("LoggerDevStats.set_memory_usage")
        return None

    def set_iteration_runtime(self, runtime: float, iteration: int) -> None:
        """
        A setter function that records the total runtime of a specific iteration.

        :param runtime: The total runtime in seconds of an iteration.
        :type runtime: float
        :param iteration: The iteration that the runtime is being recorded for.
        :type iteration: int
        """
        self.iteration_runtime[iteration] = runtime
        self.log_function_call("LoggerDevStats.set_iteration_runtime")
        return None

    def new_iteration(self, init: bool = False) -> None:
        """
        Increment the current_iteration counter and update the total runtime.
        """
        self.current_iteration += 1

        if init or self.current_iteration < 1:
            self.log_function_call("LoggerDevStats.new_iteration")
            return None

        self.total_runtime += self.iteration_runtime[self.current_iteration - 1]
        self.log_function_call("LoggerDevStats.new_iteration")
        return None

    def current_iteration_repr(self) -> str:
        """
        Extract all the relevant information for the current iteration and format it into a string to be printed to the terminal.

        :return: A formatted text representation containing all the developer information for the current model iteration.
        :rtype: str
        """
        formatted_string: str = (
            f"""\n\n==== DEBUG -- GATOH model runtime statistics at iteration {self.current_iteration}/{self.max_iterations}====
                \n\nMemory usage at the start of the iteration: {self.current_memory_usage[self.current_iteration]}
                \nRecorded peak memory usage: {self.peak_memory_usage}
                \nTotal number of model function calls: {self.all_function_calls}
                \nCurrent model runtime: {self.total_runtime}
            """
        )
        self.log_function_call("LoggerDevStats.current_iteration_repr")
        return formatted_string


@dataclass
class LoggerVariables:
    """
    Dataclass that defines the simulation variables that are tracked and stored by the Logger.

    :param max_iterations: The maximum number of iterations that the model will run its simulation for.
    :type max_iterations: int
    :param hierarchies: The names of all social hierarchies present in the model.
    :type hierarchies: list[str]
    """

    # The maximum number of iterations that the simulation will run for
    max_iterations: int
    # The aggregated community opinion climate at each timestep
    aggregate_opinions: list[float] = field(default_factory=list)
    # The number of radicalised agents that exist in the model at each timestep
    radicalised_agents: list[int] = field(default_factory=list)
    # The number of deradicalisation events that occur at each timestep
    deradicalised_agents: list[int] = field(default_factory=list)
    # The total count of opinion silencing effects that have ocurred in the simulation over time
    silenced_agents: list[int] = field(default_factory=list)
    # The total count of opinion negation effects that have ocurred in the simulation over time
    negated_agents: list[int] = field(default_factory=list)
    # The calculated layer interdependence of each hierarchy at each timestep
    layer_interdependences: dict[str, list[float]] = field(default_factory=dict)
    # The calculated layer polarisation of each hierarchy at each timestep
    layer_polarisations: dict[str, list[float]] = field(default_factory=dict)
    # The log odds of radicalisation in the model at each timestep
    radicalisation_logodds: list[float] = field(default_factory=list)
    # The current iteration that the simulation is at
    current_iteration: int = 0
    # Space for any dynamically tracked model parameters to be stored
    model_parameters: dict[str, list[Any]] = field(default_factory=dict)

    def __init__(self, max_iterations: int, hierarchies: list[str]) -> None:
        """
        Store the number of max iterations and initialise all lists and dictionaries with the appropriate hierarchy names
        and sizes to match the number of iterations.
        """
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.aggregate_opinions = [0.0 for _ in range(self.max_iterations)]
        self.radicalised_agents = [0 for _ in range(self.max_iterations)]
        self.deradicalised_agents = [0 for _ in range(self.max_iterations)]
        self.silenced_agents = [0 for _ in range(self.max_iterations)]
        self.negated_agents = [0 for _ in range(self.max_iterations)]
        self.radicalisation_logodds = [0.0 for _ in range(self.max_iterations)]
        self.model_parameters = {}

        self.layer_interdependences = {}
        self.layer_polarisations = {}

        for hierarchy in hierarchies:
            self.layer_interdependences[hierarchy] = [
                0.0 for _ in range(self.max_iterations)
            ]
            self.layer_polarisations[hierarchy] = [
                0.0 for _ in range(self.max_iterations)
            ]

    def increment_radicalised(self, flag: bool) -> None:
        """
        A simple setter function that checks the input flag and updates the radicalisation count accordingly.

        :param flag: A flag indicating if radicalisation ocurred.
        :type flag: bool
        """
        if flag:
            self.radicalised_agents[self.current_iteration - 1] += 1
        return None

    def increment_deradicalised(self, flag: bool) -> None:
        """
        A simple setter function that checks the input flag and updates the deradicalisation count accordingly.

        :param flag: A flag indicating if deradicalisation occurred.
        :type flag: bool
        """
        if flag:
            self.deradicalised_agents[self.current_iteration - 1] += 1
        return None

    def increment_silenced(self, flag: bool) -> None:
        """
        A simple setter function that checks the input flag and updates the opinion silencing events count accordingly.

        :param flag: A flag indicating if opinion silencing ocurred.
        :type flag: bool
        """
        if flag:
            self.silenced_agents[self.current_iteration - 1] += 1
        return None

    def increment_negated(self, flag: bool) -> None:
        """
        A simple setter function that checks the input flag and updates the opinion negation events count accordingly.

        :param flag: A flag indicating if opinion negation ocurred.
        :type flag: bool
        """
        if flag:
            self.negated_agents[self.current_iteration - 1] += 1
        return None

    def store_aggregate_opinion(self, agg_opp: float) -> None:
        """
        A setter function that simplifies the storing of aggregate opinion values at each iteration.

        :param agg_opp: The aggregate opinion value to store for the current iteration.
        :type agg_opp: float
        """
        self.aggregate_opinions[self.current_iteration - 1] = agg_opp
        return None

    def store_radicalisation_logodds(self, r_logodds: float) -> None:
        """
        A setter function that simplifies the storing of the model's radicalisation log odds at each iteration.

        :param r_logodds: The model's radicalisation log odds value to store for the current iteration.
        :type r_logodds: float
        """
        self.radicalisation_logodds[self.current_iteration - 1] = r_logodds
        return None

    def store_layer_interdependences(self, layer_interdeps: dict[str, float]) -> None:
        """
        A setter function that simplifies the storing of the model's layer interdependences at each iteration.

        :param layer_interdeps: A <hierarchy : interdependence value> mapping that tracks the layer interdependences to be stored for this iteration.
        :type layer_interdeps: dict[str, float]
        """
        for hierarchy, interdependence in layer_interdeps.items():
            self.layer_interdependences[hierarchy][self.current_iteration - 1] = (
                interdependence
            )
        return None

    def store_layer_polarisations(self, layer_polars: dict[str, float]) -> None:
        """
        A setter function that simplifies the storing of the model's layer polarisations at each iteration.

        :param layer_polars: A <hierarchy : polarisation value> mapping that tracks the layer polarisations to be stored for this iteration.
        :type layer_polars: dict[str, float]
        """
        for hierarchy, polarisation in layer_polars.items():
            self.layer_polarisations[hierarchy][self.current_iteration - 1] = (
                polarisation
            )
        return None

    def store_model_parameters(self, model_parameters: dict[str, Any] | None = None) -> None:
        """
        A setter function that simplifies the storing of the model's tracked parameters at each iteration.

        :param model_parameters: A <parameter : value> mapping for the tracked model parameters to be logged.
        :type model_parameters: dict[str, Any], optional
        """
        # It is assumed that the model parameters will always exist in self.model_parameters by the time this function is being called...
        if model_parameters is not None:
            for model_parameter, value in model_parameters.items():
                self.model_parameters[model_parameter][self.current_iteration - 1] = value
        return None

    def new_iteration(self, init: bool = False) -> None:
        """
        Increment the current_iteration counter and then copy all the values from the previous iteration to their respective list indexes for the new iteration.

        :param init: A flag indicating if the call is being made during the first model iteration (no previous values to copy)
        :type init: bool, optional
        """
        self.current_iteration += 1

        if init or self.current_iteration <= 1:
            return None

        # Variables defined to reduce repetition below
        t_now: int = self.current_iteration - 1
        t_last: int = self.current_iteration - 2
        # -1 and -2 indexes due to indexing logic for lists...

        # Only these 3 variables must be carried over, all others are calculated at the end of the timestep independently
        self.radicalised_agents[t_now] = self.radicalised_agents[t_last]
        self.deradicalised_agents[t_now] = self.deradicalised_agents[t_last]
        self.silenced_agents[t_now] = self.silenced_agents[t_last]
        self.negated_agents[t_now] = self.negated_agents[t_last]
        return None

    def current_layers_repr(self) -> str:
        """
        Extract all the per-hierarchy variables for the current iteration and format it into a substring to be appended to the main iteration output.

        :return: A formatted text representation containing all the per-hierarchy variables for the current model iteration.
        :rtype: str
        """
        output_string: str = (
            "Hierarchy Name\tLayer Interdependence\tLayer Polarisation\n"
        )
        for hierarchy in self.layer_interdependences:
            interdepence: float = self.layer_interdependences[hierarchy][
                self.current_iteration - 1
            ]
            polarisation: float = self.layer_polarisations[hierarchy][
                self.current_iteration - 1
            ]
            hierarchy_string: str = f"{hierarchy}\t{interdepence}\t{polarisation}\n"
            output_string += hierarchy_string
        return output_string

    def current_model_params_repr(self) -> str:
        """
        Extract all the explicitly tracked model parameters for the current iteration and format them into a substring to be appended to the main iteration output.

        :return: A formatted text representation containing all the tracked model parameters for the current model iteration.
        :rtype: str
        """
        output_string: str = ""
        if len(self.model_parameters) > 0:
            output_string += "Model Parameter\tValue\n"
            for parameter, value in self.model_parameters.items():
                output_string += f"{parameter}\t{value}\n"
        return output_string

    def current_iteration_repr(self) -> str:
        """
        Extract all variable information for the current iteration and format it into a string to be printed to the terminal.

        :return: A formatted text representation containing all the variables for the current model iteration.
        :rtype: str
        """
        formatted_string: str = (
            f"""\n\n==== GATOH model variables at iteration {self.current_iteration}/{self.max_iterations}====
                \n\nAggregate community opinion: {self.aggregate_opinions[self.current_iteration - 1]}
                \nNumber of radicalisation events in the community: {self.radicalised_agents[self.current_iteration - 1]}
                \nNumber of deradicalisation events in the community: {self.deradicalised_agents[self.current_iteration - 1]}
                \nLog odds of radicalisation ocurring: {self.radicalisation_logodds[self.current_iteration - 1]}
                \nNumber of opinion silencing events: {self.silenced_agents[self.current_iteration - 1]}
                \nNumber of opinion negation events: {self.negated_agents[self.current_iteration - 1]}
                \n\n**** Layer statistics ****\n\n"""
            + self.current_layers_repr()
        )
        if len(self.model_parameters) > 0:
            formatted_string += f"\n\n**** Model parameters ****\n\n{self.current_model_params_repr()}"
        return formatted_string

    def get_fieldnames(self) -> list[str]:
        """
        A helper function that provides a place to collect and return all of the dataclass' attribute names for use
        as CSV column names.

        :return: The names of all the attributes in this dataclass.
        :rtype: list[str]
        """
        attribute_names: list[str] = [
            "iterations",
            "aggregate_opinions",
            "radicalised_agents",
            "deradicalised_agents",
            "silenced_agents",
            "negated_agents",
            "radicalisation_logodds",
        ]

        for key in self.layer_interdependences:
            attribute_names.append(f"layer_interdependences_{key}")

        for key in self.layer_polarisations:
            attribute_names.append(f"layer_polarisations_{key}")

        for key in self.model_parameters:
            attribute_names.append(f"{key}")

        return attribute_names


@dataclass
class LoggedAgents:
    """
    Dataclass that provides the framework used to track specific attribute of certain agents throughout model runtimes.

    :param max_iterations: The maximum number of iterations that the model will run its simulation for.
    :type max_iterations: int
    """

    # The maximum iterations that the simulation will run for
    max_iterations: int
    # The current iteration that the simulation is at
    current_iteration: int = 0
    # The dictionary that will be used to store agent opinions
    opinions: dict[str, list[float]] = field(default_factory=dict)
    # The dictionary that will be used to store agent previous opinions
    previous_opinions: dict[str, list[float]] = field(default_factory=dict)
    # The dictionary that will be used to store agent radicalisation statuses
    radicalisations: dict[str, list[bool]] = field(default_factory=dict)
    # The dictionary that will be used to store agent social weightings
    social_weightings: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    # The dictionary that will be used to store agent silencing statuses
    silencings: dict[str, dict[str, list[bool]]] = field(default_factory=dict)
    # The dictionary that will be used to store any custom attributes
    custom_attributes: dict[str, dict[str, list[Any]]] = field(default_factory=dict)

    def __init__(self, max_iterations: int) -> None:
        """
        Store the number of max iterations.
        """
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.opinions = {}
        self.previous_opinions = {}
        self.radicalisations = {}
        self.social_weightings = {}
        self.silencings = {}
        self.custom_attributes = {}

    def track_attribute(self, agent_id: str, attribute_name: str, agent_hierarchies: list[str] | None = None) -> None:
        """
        Initialise the list to track the specified attribute for an agent.

        :param agent_id: The unique ID of the agent for which the attribute is being tracked.
        :type agent_id: str
        :param attribute_name: The name of the parameter to be tracked.
        :type attribute_name: str
        :param agent_hierarchies: The names of the hierarchies that the agent belongs to.
        :type agent_hierarchies: list[str], optional
        :raises ValueError: If the tracked attribute is social_weightings or is_silenced and no agent_hierarchies are being passed.
        """
        match attribute_name:
            case "opinion":
                self.opinions[agent_id] = [0.0 for _ in range(self.max_iterations)]
            case "previous_opinion":
                self.previous_opinions[agent_id] = [0.0 for _ in range(self.max_iterations)]
            case "radicalised":
                self.radicalisations[agent_id] = [False for _ in range(self.max_iterations)]
            case "social_weightings":
                if agent_hierarchies is None:
                    raise ValueError("Wanting to track social_weightings for an agent but no agent hierarchies were passed to the logger")
                self.social_weightings[agent_id] = {hierarchy : [0.0 for _ in range(self.max_iterations)] for hierarchy in agent_hierarchies}
            case "is_silenced":
                if agent_hierarchies is None:
                    raise ValueError("Wanting to track is_silenced for an agent but no agent hierarchies were passed to the logger")
                self.silencings[agent_id] = {hierarchy: [False for _ in range(self.max_iterations)] for hierarchy in agent_hierarchies}
            case _:
                self.custom_attributes.setdefault(agent_id, {})[attribute_name] = [None for _ in range(self.max_iterations)]
        return None

class GATOHLogger:
    """
    The logging module will contain all functions related to logging and/or printing model progress and information
    both during and after simulation.

    :param max_iterations: The maximum number of iterations that the parent model is running its simulation for.
    :type max_iterations: int
    :param hierarchies: The names of the social hierarchies present in the parent model.
    :type hierarchies: list[str]
    :param verbose: A flag indicating if extended information should be printed during logging.
    :type verbose: bool, optional
    :param print_interval: The number of model iterations to run in between each printed logging output.
    :type print_interval: int, optional
    :param print_outside_interval: A flag indicating if a simple string indicating just the iteration number should be printed outside the print_interval.
    :type print_outside_interval: bool, optional
    :param write_file: A flag indicating if a log file should be written to disk at the end of logging.
    :type write_file: bool, optional
    :param debug: A flag indicating if additional developer statistics should be logged and reported.
    :type debug: bool, optional
    """
    def __init__(
        self,
        max_iterations: int,
        hierarchies: list[str],
        verbose: bool = False,
        print_interval: int = 10,
        print_outside_interval: bool = True,
        write_file: bool = True,
        debug: bool = False,
    ) -> None:
        self.verbose: bool = verbose
        self.print_interval: int = print_interval
        self.print_outside_interval: bool = print_outside_interval
        self.write_file: bool = write_file
        self.debug: bool = debug
        self.variables: LoggerVariables = LoggerVariables(max_iterations, hierarchies)
        self.agents: LoggedAgents = LoggedAgents(max_iterations)
        if self.debug:
            self.dev_stats: LoggerDevStats = LoggerDevStats(max_iterations)

    def format_non_interval_print(self) -> str:
        """
        Returns a formatted string to be printed out on iterations which fall outside the print interval (to still provide some feedback on iteration progress)

        Defined as its own function to allow for easy modification in the future.

        :return: The formatted text to print outside of the print interval.
        :rtype: str
        """
        non_interval_string: str = f"\n\n========== Iteration {self.variables.current_iteration}/{self.variables.max_iterations} ==========\n\n"
        if self.debug:
            self.log_function_call("GATOHLogger.format_non_interval_print")
        return non_interval_string

    def new_iteration(self, init: bool = False) -> None:
        """
        A wrapper that calls LoggerVariables new_iteration().

        :param init: A flag indicating if this function is being called from the first iteration of the model.
        :type init: bool, optional
        """
        self.variables.new_iteration(init=init)
        if self.debug:
            self.dev_stats.new_iteration(init=init)
            self.log_function_call("GATOHLogger.new_iteration")
        return None

    def track_model_parameters(self, parameters: list[str]) -> None:
        """
        A utility function that adds new model parameters to be tracked to the logger variables.

        :param parameters: The names of the model parameters to be tracked.
        :type parameters: list[str]
        """
        for parameter in parameters:
            self.variables.model_parameters.setdefault(parameter, [0 for _ in  range(self.variables.max_iterations)])
        if self.debug:
            self.log_function_call("GATOHLogger.track_model_parameters")
        return None

    def iteration(
        self,
        aggregate_opinion: float,
        radicalisation_logodds: float,
        layer_interdependences: dict[str, float],
        layer_polarisations: dict[str, float],
        model_parameters: dict[str, Any] | None = None,
    ) -> None:
        """
        Store all relevant model variables and states based on the level of logging that has been specified.

        :param aggregate_opinion: The aggregate network opinion that has been observed in the model at the end of this iteration.
        :type aggregate_opinion: float
        :param radicalisation_logodds: The log odds of an agent being radicalised in the model at the end of this iteration.
        :type radicalisation_logodds: float
        :param layer_interdependences: A <hierarchy : value> mapping containing the calculated layer interdependency for each hierarchy in the model at the end of this iteration.
        :type layer_interdependences: dict[str, float]
        :param layers_polarisation: A <hierarchy : value> mapping containing the calculated polarisation for each hierarchy in the model at the end of this iteration.
        :type layers_polarisation: dict[str, float]
        :param model_parameters: A <parameter : value> mapping containing a tracked model parameter's value at the end of this iteration.
        :type model_parameters: dict[str, Any], optional
        """
        self.variables.store_aggregate_opinion(aggregate_opinion)
        self.variables.store_radicalisation_logodds(radicalisation_logodds)
        self.variables.store_layer_interdependences(layer_interdependences)
        self.variables.store_layer_polarisations(layer_polarisations)
        self.variables.store_model_parameters(model_parameters)

        if self.debug:
            self.log_function_call("LoggerVariables.store_aggregate_opinion")
            self.log_function_call("LoggerVariables.store_radicalisation_logodds")
            self.log_function_call("LoggerVariables.store_layer_interdependences")
            self.log_function_call("LoggerVariables.store_layer_polarisations")
            self.log_function_call("GATOHLogger.iteration")
        return None

    def debug_iteration(self) -> None:
        """
        Store all relevant developer variables based on the level of logging that has been specified.
        """
        current, peak = tracemalloc.get_traced_memory()

        self.dev_stats.set_memory_usage(current, self.dev_stats.current_iteration)
        self.dev_stats.record_peak_mem_usage(peak)

        self.log_function_call("GATOHLogger.debug_iteration")
        return None

    def debug_iteration_print(self) -> str:
        """
        A method which formats the debug statistics for printing.

        :return: An appropriately formatted textual representation of debug statistics to be printed out by the model for this iteration.
        :rtype: str
        """
        print_string: str
        if self.debug:
            print_string = self.dev_stats.current_iteration_repr()
            self.log_function_call("LoggerDevStats.debug_iteration_print")
            return print_string
        else:
            return ""

    def iteration_print(self) -> str:
        """
        A method which formats the model statistics at the appropriate print_interval.

        :return: An appropriately formatted textual representation of variables to be printed out by the model for this iteration.
        :rtype: str
        """
        print_string: str
        if self.variables.current_iteration % self.print_interval == 0:
            print_string = self.variables.current_iteration_repr()
            if self.debug:
                self.log_function_call("LoggerVariables.current_iteration_repr")
                self.log_function_call("GATOHLogger.iteration_print")
            return print_string
        else:
            print_string = self.format_non_interval_print()
            if self.debug:
                self.log_function_call("GATOHLogger.iteration_print")
            return print_string

    def log_function_call(self, function_name: str) -> None:
        """
        A wrapper to the DevStats log_function_call.

        :param function_name: The name of the function that was called.
        :type function_name: str
        """
        self.dev_stats.log_function_call(function_name)
        self.dev_stats.log_function_call("GATOHLogger.log_function_call")
        return None

    def save_data(self, save_path: str) -> bool:
        """
        Saves all logged data to the specified path.

        :param save_path: The path to save the logger's data to.
        :type save_path: str
        :return: A flag indicating if data was successfully saved or not.
        :rtype: bool
        """
        # An empty string means that no explicit save path was given to the model, and it is assumed that no saving is desired.
        if save_path == "":
            return False

        with open(save_path, "w", newline="") as csvfile:
            field_names: list[str] = self.variables.get_fieldnames()

            csv_writer: csv.DictWriter[str] = csv.DictWriter(csvfile, fieldnames=field_names)
            csv_writer.writeheader()

            for i in range(self.variables.max_iterations):
                row_dict: dict[str, str] = {
                    "iterations": f"{i + 1}",
                    "aggregate_opinions": f"{self.variables.aggregate_opinions[i]}",
                    "radicalised_agents": f"{self.variables.radicalised_agents[i]}",
                    "deradicalised_agents": f"{self.variables.deradicalised_agents[i]}",
                    "silenced_agents": f"{self.variables.silenced_agents[i]}",
                    "negated_agents": f"{self.variables.negated_agents[i]}",
                    "radicalisation_logodds": f"{self.variables.radicalisation_logodds[i]}",
                }

                # Append interdependences and polarisations in the same loop as the headers should handle ordering automatically
                for hierarchy in self.variables.layer_interdependences:
                    row_dict[f"layer_interdependences_{hierarchy}"] = (
                        f"{self.variables.layer_interdependences[hierarchy][i]}"
                    )
                    row_dict[f"layer_polarisations_{hierarchy}"] = (
                        f"{self.variables.layer_polarisations[hierarchy][i]}"
                    )

                # Append the explicitly tracked model parameters in the same loop as the headers should handle ordering automatically
                for model_parameter in self.variables.model_parameters:
                    row_dict[model_parameter] = f"{self.variables.model_parameters[model_parameter][i]}"

                csv_writer.writerow(row_dict)

        if self.debug:
            self.log_function_call("GATOHLogger.save_data")

        return True
