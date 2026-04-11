from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoggerVariables:
    """
    Dataclass that defines the simulation variables that are stored and tracked by the Logger.
    """

    # The maximum number of iterations that the simulation will run for
    max_iterations: int
    # The aggregated community opinion climate at each timestep
    aggregate_opinions: list[float] = field(default_factory=list)
    # The number of radicalised agents that exist in the model at each timestep
    radicalised_agents: list[int] = field(default_factory=list)
    # The total count of opinion silencing effects that have ocurred in the simulation over time
    silenced_agents: list[int] = field(default_factory=list)
    # The total count of opinion negation effects that have ocurred in the simulation over time
    negated_agents: list[int] = field(default_factory=list)
    # The calculated layer interdependence of each hierarchy at each timestep
    layer_interdependences: dict[str, list[float]] = field(default_factory=dict)
    # The calculated layer navigabilities of each hierarchy at each timestep
    layer_navigabilities: dict[str, list[float]] = field(default_factory=dict)
    # The log odds of radicalisation in the model at each timestep
    radicalisation_logodds: list[float] = field(default_factory=list)
    # The current iteration that the simulation is at
    current_iteration: int = 0

    def __init__(self, max_iterations: int, hierarchies: list[str]) -> None:
        """
        Store the number of max iterations and initialise all lists and dictionaries with the appropriate hierarchy names
        and sizes to match the number of iterations.

        :param max_iterations: The maximum number of iterations that the model will run its simulation for.
        :param hierarchies: A list containing the names of all social hierarchies present in the model.
        """
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.aggregate_opinions = [0.0 for _ in range(self.max_iterations)]
        self.radicalised_agents = [0 for _ in range(self.max_iterations)]
        self.silenced_agents = [0 for _ in range(self.max_iterations)]
        self.negated_agents = [0 for _ in range(self.max_iterations)]
        self.radicalisation_logodds = [0.0 for _ in range(self.max_iterations)]

        for hierarchy in hierarchies:
            self.layer_interdependences[hierarchy] = [
                0.0 for _ in range(self.max_iterations)
            ]
            self.layer_navigabilities[hierarchy] = [
                0.0 for _ in range(self.max_iterations)
            ]

    def current_layers_repr(self) -> str:
        """
        Extract all the per-hierarchy variables for the current iteration and format it into a substring to be appended to the main iteration output.

        :return: A formatted substring containing all the per-hierarchy variables for the current model iteration.
        """
        output_string: str = (
            "\tHierarchy Name\tLayer Interdependence\tLayer Navigability\n"
        )
        for hierarchy in self.layer_interdependences.keys():
            interdepence: float = self.layer_interdependences[hierarchy][
                self.current_iteration
            ]
            navigability: float = self.layer_navigabilities[hierarchy][
                self.current_iteration
            ]
            hierarchy_string: str = f"\t{hierarchy}\t{interdepence}\t{navigability}\n"
            output_string += hierarchy_string
        return output_string

    def current_iteration_repr(self) -> str:
        """
        Extract all variable information for the current iteration and format it into a string to be printed to the terminal.

        :return: A formatted string containing all the variables for the current model iteration.
        """
        formatted_string: str = f"""\n\n==== GATOH model variables at iteration {self.current_iteration}/{self.max_iterations} ====\n\n
            Aggregate community opinion: {self.aggregate_opinions[self.current_iteration]}\n
            Number of radicalised agents in the community: {self.radicalised_agents[self.current_iteration]}\n
            Log odds of radicalisation ocurring: {self.radicalisation_logodds[self.current_iteration]}\n
            Number of opinion silencing events: {self.silenced_agents[self.current_iteration]}\n
            Number of opinion negation events: {self.negated_agents[self.current_iteration]}\n\n
            **** Layer statistics ****\n\n
            """ + self.current_layers_repr()
        return formatted_string


class GATOHLogger:
    """
    The logging module will contain all functions related to logging and/or printing model progress and information
    both during and after simulation.
    """

    # TODO: Expand on logging capabilities

    def __init__(
        self,
        model: Any,
        max_iterations: int,
        hierarchies: list[str],
        verbose: bool = False,
        print_interval: int = 10,
        write_file: bool = True,
    ) -> None:
        """
        :param model: The parent ABModel object that the GATOHLogger object is being attached to.
        :param max_iterations: The maximum number of iterations that the parent model is running its simulation for.
        :param hierarchies: A list containing the names of the social hierarchies present in the parent model.
        :param verbose: a flag to indicate if extended information should be printed during logging.
        :param print_interval: the number of model iterations to run in between each printed logging output.
        :param write_file: a flag to indicate if a log file should be written to disk at the end of logging.
        """
        self.parent_model: Any = model
        self.verbose: bool = verbose
        self.print_interval: int = print_interval
        self.write_file: bool = write_file
        self.variables: LoggerVariables = LoggerVariables(max_iterations, hierarchies)

    def iteration(self) -> None:
        """
        Inspect and store all relevant model variables and states based on the level of logging that has been specified.
        """
        # TODO: Implement this function
        pass

    def iteration_print(self, current_iteration: int) -> None:
        """
        A method which prints out informative model statistics at the appropriate print_interval.

        :param current_iteration: The current iteration that the model is at when calling this method.
        """
        if current_iteration % self.print_interval != 0:
            return None
        else:
            # TODO: Finish this print block
            pass
        return None
