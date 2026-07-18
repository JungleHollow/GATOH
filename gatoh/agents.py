from __future__ import annotations

import gc
import os
import pickle
import random as rd
import warnings
import zipfile
from collections.abc import Iterable, Iterator
import concurrent.futures
from copy import deepcopy
from shutil import rmtree
from typing import TypeVar, override

from gatoh.utils import draw_random_value, random_coinflip, value_rw_delta

# Definition of all valid, existing Agent personality types
PERSONALITIES: list[str] = ["neutral", "rational", "erratic", "impulsive", "social"]

# A generic to be used in cases where variables may be any type.
T = TypeVar("T")
# A generic to be used for the case where a fully generic dictionary may be passed (e.g dict[S, T])
S = TypeVar("S")

def draw_personality() -> str:
    """
    An Agent utility function that randomly draws a valid Agent personality type.

    :return: The string representing the drawn personality type.
    :rtype: str
    """
    drawn_personality: str = rd.choice(PERSONALITIES)
    return drawn_personality


class Agent:
    """
    A class to define the Agent objects that will interact with each other in an agent-based model.

    Supported positional arguments:
        - <string> to set the Agent's id.
        - <dict> of {hierarchy_name : weight} for the personal value that this Agent assigns to each social hierarchy
        - <float> in the range [-1, 1] to set the Agent's initial opinion on the topic of interest
        - <bool> to define if the socially contagious belief will be of personal benefit to this Agent
        - <(string, float)> for the Agent's defined personality and their social susceptibility (range [0, 1])

    Any arbitrary keyword arguments that are passed to initialise this object will create a new attribute for
    the Agent object, and store the value of that attribute as the input data type.

    If any pre-existing attributes are passed as keyword arguments, this simply sets the attributes to the input value.

    If both a positional argument and a corresponding keyword argument are passed for the same attribute,
    the value given to the keyword argument will override whatever was passed as the positional value.

    :param id: Positional argument -- provides a unique identifier for an agent.
    :type id: str, optional
    :param social_weightings: Positional argument -- <hierarchy : weighting> mapping indicating the personal value that the agent assigns to each social hierarchy.
    :type social_weightings: dict[str, float], optional
    :param opinion: Positional argument -- the current opinion value that the agent is being initialised with.
    :type opinion: float, optional
    :param personal_benefit: Positional argument -- a flag indicating if the socially contagious belief is of personal benefit to the agent.
    :type personal_benefit: bool, optional
    :param behaviour_tuple: Positional argument -- a pair of variables detailing the agent's defined personality and social susceptibility (range [0, 1]), respectively.
    :type behaviour_tuple: tuple[str, float], optional
    :param index: Keyword argument -- the agent's index within an AgentSet.
    :type index: int, optional
    :param is_silenced: Keyword argument -- <hierarchy : flag> mapping indicating in which social hierarchies the agent is silenced.
    :type is_silenced: dict[str, bool], optional
    :param previous_opinion: Keyword argument -- the opinion value held by the agent at the previous model iteration.
    :type previous_opinion: float, optional
    :param radicalised: Keyword argument -- a flag indicating if the agent is radicalised.
    :type radicalised: bool, optional
    :param rw_distributions: Keyword argument -- <hierarchy : random walk params> mapping for agent-specific dynamic hierarchy weightings.
    :type rw_distributions: dict[str, tuple[float, float]], optional
    :param opinion_rw: Keyword argument -- (mean, variance) parameters for agent-specific stochastic opinion shifts.
    :type opinion_rw: tuple[float, float], optional
    """

    def __init__(self, *args: T, **kwargs: T) -> None:
        # Attributes declared but without initialisation will be defined by self.generate_agent() in a subsequent call if no args are passed
        self.id: str  # Can be any arbitrary string, but likely will follow the form XXXX0000 allowing for up to 9999 agents per community
        self.index: int  # The Agent's index within the AgentSet it belongs in

        self.social_weightings: dict[str, float] = {}
        self.is_silenced: dict[str, bool] = {}

        self.opinion: float  # Range always [-1, 1]
        self.previous_opinion: float = (
            0.0  # Used to handle updating during model iterations
        )

        self.personal_benefit: bool  # Whether the Agent is personally benefitted by the adoption of the 'social virus' that is being spread

        self.social_susceptibility: float  # Range always [0, 1]
        self.personality: str = "neutral"
        self.radicalised: bool = False

        # To assign per-agent random-walk parameters for the dynamic hierarchy weightings
        self.rw_distributions: dict[str, tuple[float, float]] | None = None

        # To assign per-agent random-walk parameters for the stochastic opinion shifts
        self.opinion_rw: tuple[float, float] | None = None

        # If no args have been passed, it is assumed that self.generate_agent() will be subsequently called
        if args:
            for arg in args:
                if isinstance(arg, dict):
                    self.add_attribute("social_weightings", value=arg)
                    for hierarchy in self.social_weightings.keys():
                        self.is_silenced[hierarchy] = False
                elif isinstance(arg, float):
                    self.add_attribute("opinion", value=arg)
                elif isinstance(arg, tuple):
                    self.add_attribute("personality", value=arg[0])
                    self.add_attribute("social_susceptibility", value=arg[1])
                elif isinstance(arg, str):
                    self.add_attribute("id", value=arg)
                elif isinstance(arg, bool):
                    self.add_attribute("personal_benefit", value=arg)
                else:
                    pass
        if kwargs:
            for key, value in kwargs.items():
                # No checking for duplicate keys; assume that explicitly added kwargs should override any args.
                self.add_attribute(key, value=value)

    def generate_agent(
        self,
        id: str,
        index: int,
        hierarchies: list[str],
        distribution: str = "gaussian",
        explicit_rw: bool = False,
        explicit_opinion_rw: bool = False,
        personality: str | None = None,
        parameters: dict[str, float] | None = None,
        personal_benefit: bool | None = None,
    ) -> Agent:
        """
        Randomly generate an Agent object based on the input parameters.

        :param id: The id that has been assigned for this specific Agent object under the conditions of the model specifications.
        :type id: str
        :param index: The index of the Agent object within the model's AgentSet.
        :type index: int
        :param hierarchies: The names of all valid social hierarchies in the model.
        :type hierarchies: list[str]
        :param distribution: The distribution to use for relevant attribute generation (Valid distributions include: 'gaussian', 'beta')
        :type distribution: str, optional
        :param explicit_rw: A flag indicating if explicit hierarchy weighting random walk parameters should be generated for this Agent.
        :type explicit_rw: bool, optional
        :param explicit_opinion_rw: A flag indicating if an explicit opinion random walk parameter should be generated for this Agent.
        :type explicit_opinion_rw: bool, optional
        :param personality: A string defining what type of personality the agent will have (defaults to 'neutral' on Agent __init__)
        :type personality: str, optional
        :param parameters: A dictionary containing the distribution parameters used to generate random values.
        :type parameters: dict[str, float], optional
        :param personal_benefit: A boolean indicating if the Agent would be personally benefitted by the adoption of the 'social virus' being spread.
        :type personal_benefit: bool, optional
        :return: The generated Agent object.
        :rtype: Agent
        """
        # Begin by setting crucial information
        self.id = id
        self.index = index
        if personality:
            self.personality = personality

        if personal_benefit:
            self.personal_benefit = personal_benefit
        else:
            self.personal_benefit = random_coinflip("bool")

        # Generate a weighting for each hierarchy; initialise the is_silenced flag for that hierarchy
        for hierarchy in hierarchies:
            self.social_weightings[hierarchy] = draw_random_value(
                distribution, parameters=parameters
            )
            self.is_silenced[hierarchy] = False

        # Handle explicit hierarchy weighting random walk parameter generation
        if explicit_rw:
            rw_params: dict[str, tuple[float, float]] = {}
            for hierarchy in hierarchies:
                rw_mean: float = draw_random_value("gaussian", parameters=parameters)
                rw_variance: float = draw_random_value(
                    "gaussian", parameters=parameters
                )

                rw_params[hierarchy] = (rw_mean, rw_variance)
            self.rw_distributions = rw_params

        # Handle the explicit opinion random walk parameter generation
        if explicit_opinion_rw:
            rw_mean = draw_random_value("gaussian", parameters=parameters)
            rw_variance = draw_random_value("gaussian", parameters=parameters)
            self.opinion_rw = (rw_mean, rw_variance)

        # Generate the Agent's initial opinion
        self.opinion = draw_random_value(distribution, parameters=parameters)

        # If the initial opinion is very strong, the Agent is initialised as radicalised
        if -0.9 >= self.opinion >= 0.9:
            self.radicalised = True

        # Generate the Agent's susceptibility to social contagion
        self.social_susceptibility = draw_random_value(
            distribution, parameters=parameters
        )

        return self

    def add_attribute(
        self,
        name: str,
        value: T | dict[S, T] | tuple[T, ...] | list[T] | None = None,
        parameters: dict[str, float] | None = None,
        distribution: str | None = None,
        overwrite: bool = True,
    ) -> None:
        """
        Dynamically add an attribute to this Agent object. If "value" is passed, an explicit initial value is given;
        if "mean" and "var" are passed, a value is generated from a random distribution.
        Supported random distributions are:

            - "gaussian"
            - "beta"
            - "gamma"
            - "uniform"
            - "levy"

        Either "value", or parameters with "mean" and "var" included should be passed to this function, not both.

        If both an explicit value and distribution parameters are input to this function, the explicit value
        will always override the distribution parameters when setting the attribute.

        :param name: The name of the attribute to be added.
        :type name: str
        :param value: Initial value of the attribute.
        :type value: Any, optional
        :param parameters: The distribution parameters that will be used with the specified distribution for parameter generation.
        :type parameters: dict[str, float], optional
        :param distribution: String to select which random distribution will be used to generate the value.
        :type distribution: str, optional
        :param overwrite: A flag indicating if the added attribute should override any existing attributes of the same name.
        :type overwrite: bool, optional
        :raises ValueError: If no valid value or distribution parameters are input, the attribute cannot be added.
        :raises UserWarning: If overwrite is explicitly False but the attribute is existing, a warning is raised without completing the operation.
        """
        if value is None and not (distribution and parameters):
            raise ValueError(
                "Either explicit `value` or distribution and valid distribution parameters are expected when adding Agent attributes."
            )

        if not overwrite and name in self.__dict__.keys():
            # Raise a warning but do not change any attributes or crash the model if overwriting an existing attribute whilst explicitly passing overwrite = False
            warnings.warn(
                f"WARNING: Attempting to overwrite an existing Agent attribute ({name}) without meaning to.",
                category=UserWarning,
            )
        else:
            if value is not None:
                # Assume a given explicit value always overrides (mean, sdev)
                self.__dict__[name] = value
            elif distribution and parameters:
                self.__dict__[name] = draw_random_value(
                    distribution, parameters=parameters
                )
        return None

    def get_attribute(self, name: str) -> T | None:
        """
        Return any existing or dynamically added attribute held by the Agent object.

        :param name: The name of the parameter to get.
        :type name: str
        :return: The value stored for the parameter.
        :rtype: Any
        """
        return self.__dict__.get(name)

    def store_previous_opinion(self) -> None:
        """
        A setter method that stores the Agent's current opinion into the previous opinion.
        """
        self.previous_opinion = self.opinion
        return None

    def change_opinion(self, opinion_delta: float) -> None:
        """
        A setter method that changes the Agent's current opinion by a given delta value.

        :param opinion_delta: The delta value by which to shift the Agent's current opinion.
        :type opinion_delta: float
        """
        self.opinion += opinion_delta

        # Constrain the opinion back to [-1.0, 1.0] as needed
        if self.opinion < -1.0:
            self.opinion = -1.0
        elif self.opinion > 1.0:
            self.opinion = 1.0
        return None

    def change_radicalisation(self, radicalisation: bool) -> None:
        """
        A setter method that changes the Agent's radicalisation value.

        :param radicalisation: The radicalisation flag to set.
        :type radicalisation: bool
        """
        self.radicalised = radicalisation
        return None

    def change_rw_distribution(
        self, hierarchy: str, parameters: tuple[float, float]
    ) -> None:
        """
        A setter method that changes the Agent's explicit random walk parameters for a specific social hierarchy.

        :param hierarchy: The name of the hierarchy whose rw parameters are being changes.
        :type hierarchy: str
        :param parameters: The new (mean, var) for the random walk's gaussian distribution.
        :type parameters: tuple[float, float]
        """
        # Assume that if this function is being called, rw_distributions should be initialised if not already
        if not self.rw_distributions:
            self.rw_distributions = {}

        self.rw_distributions[hierarchy] = parameters
        return None

    def change_opinion_rw(self, rw_params: tuple[float, float]) -> None:
        """
        A setter method that changes the Agent's explicit opinion random walk parameters.

        :param rw_params: The new (mean, var) for the random walk's gaussian distribution.
        :type rw_params: tuple[float, float]
        """
        self.opinion_rw = rw_params
        return None

    def step(
        self,
        rw_distributions: dict[str, tuple[float, float]],
        opinion_rw: tuple[float, float],
    ) -> None:
        """
        Step the individual agent object:
            1. Handle dynamic social hierarchy weightings
            2. Handle the stochastic opinion changes experienced by the Agent

        :param rw_distributions: A mapping of <hierarchy name : (mean, variance)> defining the random walk distributions of each social hierarchy weighting in the model.
        :type rw_distribution: dict[str, tuple[float, float]]
        :param opinion_rw: A (mean, variance) tuple defining the random walk distribution at the model level for the Agent opinions.
        :type opinion_rw: tuple[float, float]
        """
        self.evolve_hierarchies(rw_distributions)
        self.stochastic_opinion(opinion_rw)
        return None

    def update(self, opinion_silenced: dict[str, bool], negation_ocurred: bool) -> None:
        """
        Updates the internal state of the agent after the model has stepped:
            1. Updates what social hierarchies the Agent's opinion is currently silenced in
            2. Inverts the Agent's current opinion if opinion negation ocurred

        :param opinion_silenced: A mapping of <hierarchy : flag> indicating which social hierarchies the Agent is silencing themselves in.
        :type opinion_silence: dict[str, bool]
        :param negation_ocurred: A flag indicating if opinion negation has ocurred in the current iteration.
        :type negation_ocurred: bool
        """
        self.is_silenced = opinion_silenced  # Update is_silenced
        if negation_ocurred:
            self.opinion *= -1.0  # Invert the Agent's current opinion
        return None

    def opinion_silencing(
        self,
        estimated_opinion_climate: float,
        silencing_threshold: float | None = None,
    ) -> tuple[bool, float]:
        """
        Determines if an agent will become silenced in a given social hierarchy based on their attributes.

        If no silencing threshold has been passed, each Agent's own social susceptibility is used as the threshold instead.

        :param estimated_opinion_climate: The opinion climate perceived by the Agent in this hierarchy (not necessarily objectively 'accurate').
        :type estimated_opinion_climate: float
        :param silencing_threshold: A hierarchy or global silencing threshold that must be surpassed for silencing to occur.
        :type silencing_threshold: float, optional
        :return: A pair of values indicating if silencing occurs, and the absolute difference between the perceived opinion climate and the Agent's own opinion, respectively.
        :rtype: tuple[bool, float]
        """
        # It is assumed that a radicalised Agent will never silence themselves regardless of the perceived opinion climate
        if self.radicalised:
            return False, 0.0

        threshold: float
        if silencing_threshold:
            threshold = silencing_threshold
        else:
            threshold = self.social_susceptibility

        absolute_difference: float = 0.0

        if self.personality in ["neutral", "rational", "erratic"]:
            # Cases where opinion silencing will be less influenced by the surrounding opinion climate.
            absolute_difference = abs(estimated_opinion_climate - self.opinion) * 0.8
        elif self.personality in ["impulsive", "social"]:
            # Cases where opinion silencing will be much more influenced by the surrounding opinion climate.
            absolute_difference = abs(estimated_opinion_climate - self.opinion)

        return absolute_difference > threshold, absolute_difference

    def opinion_negation(
        self, hierarchy: str, absolute_difference: float, threshold: float
    ) -> bool:
        """
        Checks if the Agent has experienced sufficiently 'overwhelming' social pressure in a hierarchy leading to a complete
        reversal of their opinion.

        :param hierarchy: The name of the social hierarchy where opinion negation is being checked for.
        :type hierarchy: str
        :param absolute_difference: The absolute difference between the perceived opinion climate and the Agent's own opinion.
        :type absolute_difference: float
        :param threshold: A global model threshold that has been specified for this effect to occur.
        :type threshold: float
        :return: A flag indicating if the Agent's opinion experienced a total negation.
        :rtype: bool
        """
        # It is assumed that a radicalised Agent will never experience a total opinion reversal regardless of the perceived opinion climate
        if self.radicalised:
            return False

        negation_strength: float = absolute_difference

        # Multiplication by (susceptibility * hierarchy weighting) will always decrease negation strength, whilst division will always increase it
        if self.personality in ["neutral", "rational"]:
            # Cases where opinion negation is less likely to occur
            if self.social_susceptibility * self.social_weightings[hierarchy] != 0:
                negation_strength *= (
                    self.social_susceptibility * self.social_weightings[hierarchy]
                )
        elif self.personality in ["erratic", "impulsive", "social"]:
            # Cases where opinion negation is more likely to occur
            if self.social_susceptibility * self.social_weightings[hierarchy] != 0:
                negation_strength /= (
                    self.social_susceptibility * self.social_weightings[hierarchy]
                )

        return negation_strength > threshold

    def radicalisation(
        self,
        hierarchy_changes: list[float],
        neighbour_benefits: list[bool],
        threshold: float,
    ) -> bool:
        """
        Uses the agent's own opinion as well as the neighbours' opinions to determine if
        the agent has become radicalised in their actions.

        :param hierarchy_changes: The opinion changes caused in each social hierarchy by neighbours during this iteration.
        :type hierarchy_changes: list[float]
        :param neighbour_benefits: Flags indicating the presence of personal benefit across an agent's neighbours.
        :type neighbour_benefits: list[bool]
        :param threshold: The radicalisation threshold that has been defined at the global level in the model.
        :type threshold: float
        :return: A flag indicating if the Agent has become radicalised or not.
        :rtype: bool
        """
        # If the Agent is already radicalised, always return False (as the Agent cannot become 'radicalised' again)
        if self.radicalised:
            return False

        # Absolute opinion declared here to reduce calls to abs() in the match statement
        absolute_opinion: float = abs(self.opinion)

        # Calculate "aggregate benefit" as a simple fraction of (personal benefit = True) / (length of neighbours)
        aggregate_benefit_count: float = 0.0
        for neighbour_benefit in neighbour_benefits:
            if neighbour_benefit:
                aggregate_benefit_count += 1.0

        if len(neighbour_benefits) != 0:
            aggregate_benefit: float = aggregate_benefit_count / len(neighbour_benefits)
        else:
            aggregate_benefit = aggregate_benefit_count

        match self.personality:
            case "neutral":
                # This will mean that radicalisation is exclusively determined by the strength of the Agent's opinion
                if absolute_opinion >= threshold:
                    self.radicalised = True
                    return self.radicalised
            case "rational":
                # This will likely mean that the agent is more disposed towards considering tangible benefits and their own
                # opinions when determining radicalisation, rather than external influences
                if absolute_opinion >= threshold and aggregate_benefit >= 0.5:
                    self.radicalised = True
                    return self.radicalised
                elif absolute_opinion >= threshold and not aggregate_benefit >= 0.5:
                    # In the case where the threshold is met but there is no explicit personal benefit, radicalisation is treated as a coinflip
                    self.radicalised = random_coinflip("bool")
                    return self.radicalised
            case "erratic":
                # Radicalisation is influenced by personal opinion to some extent, but is largely stochastically determined
                if absolute_opinion * 1.25 >= threshold:
                    self.radicalised = random_coinflip("bool")
                    return self.radicalised
            case "impulsive":
                # The agent places very strong consideration on tangible benefits over anything else
                if (
                    absolute_opinion >= threshold / 2
                ):  # (threshold / 2) as the Agent behaves impulsively and less is required for them to consider becoming radicalised
                    self.radicalised = self.personal_benefit
                    return self.radicalised
            case "social":
                # Radicalisation is strongly determined by the opinion climate and neighbour opinions rather than internal factors
                absolute_changes: float = 0.0

                for change in hierarchy_changes:
                    absolute_change: float = abs(change)
                    if absolute_change >= self.social_susceptibility:
                        # A strong opinion change was caused by some hierarchy
                        self.radicalised = True
                        return self.radicalised
                    else:
                        absolute_changes += absolute_change
                # If no changes were strong enough individually, check for the aggregate (with a relatively lower threshold)
                if absolute_changes >= self.social_susceptibility * (
                    len(hierarchy_changes) // 2
                ):
                    self.radicalised = True
                    return self.radicalised
            case _:
                return False
        # If this is somehow reached, an error has ocurred (but False is returned just in case)
        return False

    def evolve_hierarchies(
        self, rw_distributions: dict[str, tuple[float, float]]
    ) -> None:
        """
        Experimental function that aims to model the constantly evolving 'intrinsic value' that Agents place on
        the social hierarchies that they belong in over time.

        :param rw_distributions: A mapping specifying the global random walk distributions defined for each hierarchy in the model.
        :type rw_distributions: dict[str, tuple[float, float]]
        """
        for key, value in rw_distributions.items():
            rw_result: float | None = None

            if self.rw_distributions:
                if key in self.rw_distributions.keys():
                    rw_result = value_rw_delta(
                        self.social_weightings[key],
                        self.rw_distributions[key][0],
                        self.rw_distributions[key][1],
                    )

            if (
                not rw_result
            ):  # No explicit rw distribution was found; use the input ones instead
                rw_result = value_rw_delta(
                    self.social_weightings[key], value[0], value[1]
                )

            # Constrain the result back to [-1, 1] if necessary
            if rw_result < -1.0:
                self.social_weightings[key] = -1.0
            elif rw_result > 1.0:
                self.social_weightings[key] = 1.0
            else:
                self.social_weightings[key] = rw_result
        return None

    def stochastic_opinion(self, opinion_rw: tuple[float, float]) -> None:
        """
        Determine the stochastic direction and magnitude of a shift in the Agent's opinion and apply it.

        This is representative of opinion changes in real social networks in which individual opinions may increase or decrease
        away from the aggregate mean (independent of external influences).

        :param opinion_rw: A (mean, variance) pair which parametrises the Gaussian distribution used for stochastic opinion shift.
        :type opinion_rw: tuple[float, float]
        """
        rw_result: float | None = None

        if self.opinion_rw:
            rw_result = value_rw_delta(
                self.opinion, self.opinion_rw[0], self.opinion_rw[1]
            )

        if not rw_result:
            rw_result = value_rw_delta(self.opinion, opinion_rw[0], opinion_rw[1])

        if rw_result < -1.0:
            self.opinion = -1.0
        elif rw_result > 1.0:
            self.opinion = 1.0
        else:
            self.opinion = rw_result

        return None

    def life_events(self) -> None:
        """
        Experimental function that aims to model the ways in which Agent behaviours change according to major random life events over time
        """
        # TODO: Implement this function
        raise NotImplementedError(
            "Agent life events have not been implemented as a feature yet."
        )

    def __in__(self, iterable: Iterable[Agent]) -> bool:
        """
        Determine if the Agent is contained within an iterable of Agents

        :param iterable: The Agent objects in which membership is being determined.
        :type iterable: Iterable[Agent]
        :return: A flag indicating if this Agent is contained within the iterable.
        :rtype: bool
        """
        for agent in iterable:
            if self == agent:
                return True
        return False

    @override
    def __str__(self) -> str:
        """
        An override to what calling `print()` on this object will output.

        :return: A printable representation of the Agent object
        :rtype: str
        """
        return f"Agent {self.id} which {'is' if self.radicalised else 'is not'} radicalised with an opinion value of {self.opinion}"


class AgentSet:
    """
    An ordered collection of Agent objects that maintains consistency for the Model.

    :param model: The parent model that this AgentSet is being attached to
    :type model: ABModel
    """

    def __init__(self) -> None:
        self.agents: list[Agent] = []
        self.random: rd.Random = rd.Random()

    def save_agentset(self, directory_path: str) -> None:
        """
        Save the Agent objects into a compressed subdirectory representing the the saved AgentSet.

        :param directory_path: The path to the directory where the agentset subdirectory should be created.
        :type directory_path: str
        """
        subdirectory_path: str = f"{directory_path}/_agentset"

        # Removes the subdirectory if it already exists to allow for a new overwrite
        if os.path.isdir(subdirectory_path):
            rmtree(subdirectory_path)

        # Create the _agentset subdirectory
        os.mkdir(subdirectory_path)

        agent_save_paths: list[str] = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            saved_agent_paths = {executor.submit(self.write_agent_pickle, agent, subdirectory_path): agent.id for agent in self.agents}
            for future in concurrent.futures.as_completed(saved_agent_paths):
                agent_id = saved_agent_paths[future]
                try:
                    save_path = future.result()
                except Exception as exc:
                    print(f"Failed to write a pickle for agent {agent_id} with exception: {exc}")
                else:
                    agent_save_paths.append(save_path)

        zip_path: str = f"{subdirectory_path}.zip"

        # Removes the zip file if it already exists to allow for a new overwrite
        if os.path.exists(zip_path):
            os.remove(zip_path)

        # Compress the subdirectory to minimise storage and encapsulate all the Agents into a single object
        with zipfile.ZipFile(
            zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=4
        ) as subdir_zip:
            for agent_path in agent_save_paths:
                subdir_zip.write(agent_path, arcname=f"{os.path.basename(agent_path)}")

        # Remove the uncompressed subdirectory if compression was successful
        if os.path.exists(zip_path):
            rmtree(subdirectory_path)

        return None

    def write_agent_pickle(self, agent: Agent, subdirectory_path: str) -> str:
        """
        A helper function that allows for multithreading of :meth:`~gatoh.agents.agents.AgentSet.save_agentset`.

        :param agent: The agent that is being saved.
        :type agent: Agent
        :param subdirectory_path: The path to the subdirectory in which the agents are being saved.
        :type subdirectory_path: str
        :return: The path to which the agent pickle was saved.
        :rtype: str
        """
        agent_save_path: str = f"{subdirectory_path}/_agent_{agent.id}.pkl"
        with open(agent_save_path, "wb") as agent_pickle:
            pickle.dump(agent, agent_pickle)
        return agent_save_path

    def load_agentset(self, load_path: str) -> None:
        """
        Loads an AgentSet that has been saved following the same process as in the save_agentset() function.

        :param load_path: The path to the model's overall save directory.
        :type load_path: str
        """
        zip_load_path: str = f"{load_path}/_agentset.zip"

        if not os.path.exists(zip_load_path):
            raise FileNotFoundError(
                f"No saved AgentSet was found at the path: {zip_load_path}"
            )

        # The path to the uncompressed agentset subdirectory
        subdirectory_path: str = f"{load_path}/_agentset"

        # Remove any existing subdirectory with the same name to replace it with the newly loaded one
        if os.path.isdir(subdirectory_path):
            rmtree(subdirectory_path)

        # Create the uncompressed directory
        os.mkdir(subdirectory_path)

        # Extract all the Agent pickles to the uncompressed directory
        with zipfile.ZipFile(
            zip_load_path, mode="r", compression=zipfile.ZIP_DEFLATED, compresslevel=4
        ) as subdir_zip:
            subdir_zip.extractall(path=subdirectory_path)

        # Unpickle each Agent object and add it to the AgentSet using multithreading
        with concurrent.futures.ThreadPoolExecutor() as executor:
            agent_objects = {executor.submit(self.extract_agent_pickle, agent_pickle_name, subdirectory_path): agent_pickle_name for agent_pickle_name in os.listdir(subdirectory_path)}
            for future in concurrent.futures.as_completed(agent_objects):
                agent_pickle_path = agent_objects[future]
                try:
                    agent_object = future.result()
                except Exception as exc:
                    print(f"Failed to extract the pickled Agent object at path {agent_pickle_path} with exception: {exc}")
                else:
                    _ = self.add(agent_object)

        return None

    def extract_agent_pickle(self, agent_pickle_name: str, subdirectory_path: str) -> Agent:
        """
        A helper function that allows for multithreading of :meth:`~gatoh.agents.agents.AgentSet.load_agentset`.

        :param agent_pickle_path: The name of the pickled Agent object file.
        :type agent_pickle_path: str
        :param subdirectory_path: The root path to which the pickled agent object was written.
        :type subdirectory_path: str
        :return: The unpickled agent.
        :rtype: Agent
        """
        agent_pickle_path: str = f"{subdirectory_path}/{agent_pickle_name}"
        with open(agent_pickle_path, "rb") as agent_pickle:
            agent_object: Agent = pickle.load(agent_pickle)
        return agent_object

    def __len__(self) -> int:
        """
        A method that defines how an AgentSet object checks its length.

        :return: the number of agents present in the AgentSet.
        :rtype: int
        """
        return len(self.agents)

    def __iter__(self) -> Iterator[Agent]:
        """
        A method that defines how the AgentSet iterates over its Agents.

        :return: An iteration over all the Agents within the AgentSet.
        :rtype: Iterator[Agent]
        """
        return self.agents.__iter__()

    def __in__(self, agent: Agent) -> bool:
        """
        A method defining how an AgentSet checks for an Agent's membership.

        :param agent: The specific Agent object to check for.
        :type agent: Agent
        :return: A flag indicating if the Agent object is in the AgentSet.
        :rtype: bool
        """
        return agent in self.agents

    def __contains__(self, agent: Agent) -> bool:
        """
        A secondary method defining how an AgentSet checks for an Agent's membership.

        :param agent: The specific Agent object to check for.
        :type agent: Agent
        :return: A flag indicating if the specified Agent object is in the AgentSet.
        :rtype: bool
        """
        return self.agents.__contains__(agent)

    def __getitem__(self, item: int | slice) -> Agent | list[Agent]:
        """
        Retrieve an Agent or slice of Agents from the AgentSet.

        :param item: The parameter for selecting the agents.
        :type item: int | slice
        :return: The selected agent or slice of agents based on the specified item.
        :rtype: Agent | list[Agent]
        """
        return self.agents.__getitem__(item)

    def add(self, agent: Agent) -> int:
        """
        Add an Agent to the AgentSet.

        :param agent: The Agent object to be added.
        :type agent: Agent
        :return: The index of the newly added Agent.
        :rtype: int
        """
        self.agents.append(agent)
        self.agents[-1].index = len(self.agents) - 1
        return self.agents[-1].index

    def update_indices(self) -> None:
        """
        Iterate over the AgentSet and update the current Agent object index values.
        """
        for idx, agent in enumerate(self.agents):
            agent.index = idx
        return None

    def discard(self, agent: Agent) -> bool:
        """
        Removes an Agent from the AgentSet which matches the input Agent; does not return an error if the Agent does not exist.

        :param agent: The Agent object that should be removed from the set.
        :type agent: Agent
        :return: A flag indicating if the Agent was removed successfully or not.
        :rtype: bool
        """
        for idx, agnt in enumerate(self.agents):
            if agent == agnt:
                left_half: list[Agent] = self.agents[:idx]
                right_half: list[Agent] = self.agents[idx + 1 :]

                self.agents = deepcopy(left_half) + deepcopy(right_half)

                # Manual garbage collection
                del left_half, right_half
                _ = gc.collect()

                self.update_indices()
                return True
        return False

    def agent_at_index(self, index: int) -> Agent | None:
        """
        Returns the Agent object at the given index in the AgentSet.

        :param index: The index within the AgentSet to inspect.
        :type index: int
        :raises UserWarning: If the input index is out of bounds, raise a warning and return None.
        :return: The Agent object at the specified index.
        :rtype: Agent
        """
        try:
            return self.agents[index]
        except IndexError:
            warnings.warn(
                f"WARNING: Index {index} is out of bounds for the AgentSet. Only {len(self.agents)} Agents have been created.",
                category=UserWarning,
            )
            return None

    def get_agent_by_id(self, id: str) -> Agent | None:
        """
        Searches the AgentSet for an Agent with the given id and returns its object if it exists.

        :param id: The id that was assigned to the Agent object at creation.
        :type id: str
        :raises KeyError: If the input id does not exist in the AgentSet.
        :return: The Agent object with the specified id.
        :rtype: Agent
        """
        for agent in self.agents:
            if agent.id == id:
                return agent

        raise KeyError(
            f"The Agent with id '{id}' does not exist in the AgentSet -- unable to return an Agent object."
        )

    def get_index(self, agent: Agent) -> int:
        """
        Returns the index within the AgentSet of the input Agent object.

        :param agent: The agent whose index is being searched for.
        :type agent: Agent
        :raises KeyError: If the input Agent does not exist in the AgentSet.
        :return: The index of the agent within the AgentSet.
        :rtype: int
        """
        for idx, agt in enumerate(self.agents):
            if agent.id == agt.id:
                return idx

        raise KeyError(
            f"The Agent {agent.id} does not exist in the AgentSet -- unable to return an index."
        )

    def discard_index(self, index: int) -> bool:
        """
        Removes the Agent at the specified index in the AgentSet; does not return an error if the index is out of bounds.

        :param index: The index in the AgentSet which is to be removed.
        :type index: int
        :return: A flag indicating if the Agent was removed successfully or not.
        :rtype: bool
        """
        if 0 < index < len(self.agents):
            left_half: list[Agent] = deepcopy(self.agents[:index])
            right_half: list[Agent] = deepcopy(self.agents[index + 1 :])

            self.agents = deepcopy(left_half) + deepcopy(right_half)
            del left_half, right_half

            self.update_indices()
            return True
        return False

    def remove(self, agent: Agent) -> bool:
        """
        Removes an agent from the AgentSet which matches the input agent; returning an error if such an Agent does not exist.

        :param agent: The agent that should be removed from the set.
        :type agent: Agent
        :raises KeyError: If the input Agent does not exist in the AgentSet.
        :return: A flag indicating that the Agent was removed successfully.
        :rtype: bool
        """
        for idx, agnt in enumerate(self.agents):
            if agent == agnt:
                left_half: list[Agent] = deepcopy(self.agents[:idx])
                right_half: list[Agent] = deepcopy(self.agents[idx + 1 :])

                self.agents = deepcopy(left_half) + deepcopy(right_half)
                del left_half, right_half

                self.update_indices()
                return True
        raise KeyError(
            f"Tried to remove an Agent with id {agent.id} that doesn't exist in the AgentSet"
        )

    def remove_index(self, index: int) -> bool:
        """
        Removes the Agent at the specified index in the AgentSet; returning an error if the index is out of bounds.

        :param index: The index in the AgentSet which is to be removed.
        :type index: int
        :raises IndexError: If the input index is out of bounds for the AgentSet.
        :return: A flag indicating that the Agent was removed successfully.
        :rtype: bool
        """
        if 0 < index < len(self.agents):
            left_half: list[Agent] = deepcopy(self.agents[:index])
            right_half: list[Agent] = deepcopy(self.agents[index + 1 :])

            self.agents = deepcopy(left_half) + deepcopy(right_half)
            del left_half, right_half

            self.update_indices()
            return True
        raise IndexError(
            f"Tried to remove an Agent at out of bounds index {index} from the AgentSet"
        )

    def sample(self, n: int) -> list[Agent]:
        """
        Randomly draw n Agents from the AgentSet without replacement.

        :param n: The number of agents to sample.
        :type n: int
        :return: The agents sampled from the AgentSet.
        :rtype: list[Agent]
        """
        sampled_agents: list[Agent] = self.random.sample(self.agents, n)
        return deepcopy(sampled_agents)

    @override
    def __getstate__(self) -> dict[str, list[Agent] | rd.Random]:
        """
        Retrive the current state of the AgentSet for serialization.

        :return: A representation of the current state of the AgentSet.
        :rtype: dict
        """
        return {"agents": self.agents, "random": self.random}
