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
from math import ceil
from shutil import rmtree
from typing import Any, NotRequired, TypeVar, TypedDict, override

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gatoh.agents import Agent, PersonalityProbs

from gatoh.utils import draw_random_value, value_rw_delta, make_list_with_mode

# Definition of all valid, existing group member personality types
PERSONALITIES: list[str] = ["neutral", "rational", "erratic", "impulsive", "social"]
# Definition of all valid, existing Group cohesion categories
COHESIONS: list[str] = ["intimate", "close", "neutral", "distant", "passing"]

# Definition of global constants to be used instead of "magic numbers" throughout the code

# The absolute maximum value that aggregate group opinions can take
OPINION_MAX: float = 1.0
# The threshold that a group's radicalisation_rate must surpass for it to be considered collectively radicalised
COLLECTIVE_RAD_THRESH: float = 0.2
# The threshold that a group's member_benefit_rate must surpass for it to be considered collectively benefited
COLLECTIVE_BENEFIT_THRESH: float = 0.4
# The threshold that a group's silencing_rate must surpass for it to be considered collectively silenced
COLLECTIVE_SILENCING_THRESH: float = 0.75
# The compression level to use across relevant methods
COMPRESS_LEVEL: int = 4
# The modifier value for groups with predominant personality type ["neutral", "rational", "erratic"] for opinion silencing
OPINION_SILENCING_MODIFIER: float = 0.4
# The absolute maximum value that hierarchy weightings can take
SOCIAL_WEIGHTINGS_MAX: float = 1.0
# The threshold used when determining if stochastic benefit changes occur
BENEFIT_THRESH: float = 0.999
# The threshold used when determining if stochastic personality changes occur
PERSONALITY_THRESH: float = 0.999
# The threshold used when determining if stochastic radicalisation changes occur
RADICALISATION_THRESH: float = 0.999
# The threshold used when determining if stochastic silencing changes occur
SILENCING_THRESH: float = 0.999

# Used for type-checking valid group cohesion types wherever relevant
class CohesionProbs(TypedDict):
    intimate: NotRequired[float]
    close: NotRequired[float]
    neutral: NotRequired[float]
    distant: NotRequired[float]
    passing: NotRequired[float]


# A generic to be used in cases where variables may be any type.
T = TypeVar("T")
# A generic to be used for the case where a fully generic dictionary may be passed (e.g dict[S, T])
S = TypeVar("S")

def draw_personality() -> str:
    """
    A copy of :func:`~gatoh.agents.draw_personality` defined again here to avoid circular imports.

    :return: The string representing the drawn personality type.
    :rtype: str
    """
    drawn_personality: str = rd.choice(PERSONALITIES)
    return drawn_personality

def draw_cohesion() -> str:
    """
    A Group utility function that randomly draws a valid Group cohesion type.

    :return: The string representing the drawn cohesion type.
    :rtype: str
    """
    drawn_cohesion: str = rd.choice(COHESIONS)
    return drawn_cohesion


class Group:
    """
    A class to define Groups of Agent objects that can serve as a way of clustering hierarchies,
    and providing a pseudo-vectorisation of model operations across agents.

    Supported positional arguments:
        - <string> to set the Group's id.
        - <int> to set the Group's maximum size.

    :param id: Positional argument -- provides a unique identifier for a group.
    :type id: str, optional
    :param max_size: Positional argument -- the maximum number of Agents that can be members of this group.
    :type max_size: int, optional
    :param index: Keyword argument -- the group's index within a GroupSet.
    :type index: int, optional
    :param hierarchy: Keyword argument -- the social hierarchy that this group is being formed in.
    :type hierarchy: str, optional
    :param members: Keyword argument -- the IDs of the agents that are members of this group.
    :type members: list[str], optional
    :param aggregate_opinion: Keyword argument -- the aggregate opinion held by members of the group.
    :type aggregate_opinion: float, optional
    :param aggregate_susceptibility: Keyword argument -- the aggregate social susceptibility of group members.
    :type aggregate_susceptibility: float, optional
    :param cohesion: Keyword argument -- the level of cohesion that the group will act with.
    :type cohesion: str, optional
    :param radicalisation_rate: Keyword argument -- The proportion of agents in the group which are radicalised.
    :type radicalisation_rate: float, optional
    :param predominant_personality: Keyword argument -- The most common personality type among members in the group.
    :type predominant_personality: str, optional
    :param member_benefit_rate: Keyword argument -- The proportion of agents in the groups which are personally benefitted from the social contagion.
    :type member_benefit_rate: float, optional
    :param previous_opinion: Keyword argument -- The aggregate opinion held by the group at the immediate past iteration.
    :type previous_opinion: float, optional
    :param aggregate_hierarchy_weighting: Keyword argument -- The aggregate weighting that the members of the group give to the hierarchy they are in.
    :type aggregate_hierarchy_weighting: float, optional
    """

    def __init__(self, *args: T, **kwargs: T) -> None:
        # Attributes declared but without initialisation will be defined by self.generate_group() in a subsequent call if no args are passed
        self.id: str
        self.index: int
        self.max_size: int = -1

        self.hierarchy: str
        self.members: list[str] = []

        self.aggregate_opinion: float
        self.previous_opinion: float = 0.0

        self.member_benefit_rate: float
        self.aggregate_susceptibility: float

        self.cohesion: str = "neutral"
        self.predominant_personality: str = "neutral"

        self.radicalisation_rate: float

        self.aggregate_hierarchy_weighting: float

        self.silencing_rate: float

        # To assign per-group random-walk parameters for the dynamic hierarchy weighting
        self.rw_distribution: tuple[float, float] | None = None

        # To assign per-group random-walk parameters for the stochastic opinion shifts
        self.opinion_rw: tuple[float, float] | None = None

        # If no args have been passed, it is assumed that self.generate_group() will be subsequently called
        if args:
            for arg in args:
                if isinstance(arg, str):
                    self.add_attribute("id", value=arg)
                elif isinstance(arg, int):
                    self.add_attribute("max_size", value=arg)
                else:
                    pass
        if kwargs:
            for key, value in kwargs.items():
                # No checking for duplicate keys; assume that explicitly added kwargs should override any args
                self.add_attribute(key, value=value)

    def generate_group(
        self,
        id: str,
        index: int,
        hierarchy: str,
        members: list[Agent],
        cohesion: str | None = None,
        max_size: int | None = None,
    ) -> Group:
        """
        Generate a Group object based on the input parameters.

        :param id: The id that has been assigned for this specific Group object under the conditions of the model specifications.
        :type id: str
        :param index: The index of the Group object within the model's GroupSet.
        :type index: int
        :param hierarchy: The social hierarchy that the Group is being formed in.
        :type hierarchy: str
        :param members: The Agents that this group will be made up of.
        :type members: list[Agent]
        :param cohesion: A string defining what type of cohesion the group will behave according to (defaults to 'neutral' on Group __init__).
        :type cohesion: str, optional
        :param max_size: An explicit maximum group size to set for the generated group.
        :type max_size: int, optional
        :raises TypeError: If any of the required input parameters are of the incorrect data type.
        :raises ValueError: If the input cohesion is not a supported type.
        :return: The generated Group object.
        :rtype: Group
        """
        # Check that the required parameters are of the correct data type
        if not isinstance(id, str) or not isinstance(index, int) or not isinstance(hierarchy, str) or not isinstance(members, list):
            raise TypeError("One or more of the required parameters 'id', 'index', 'hierarchy', or 'members' are not of the appropriate data type")

        # Begin by setting crucial information
        self.id = id
        self.index = index
        if cohesion is not None:
            if cohesion not in COHESIONS:
                raise ValueError("The specified cohesion type is not supported")
            self.cohesion = cohesion

        if max_size is not None:
            # Assume that any value is valid (with max_size <= 0 meaning explicitly that there's no maximum)
            self.max_size = max_size
        else:
            # Assume that max_size should equal the number of input members
            self.max_size = len(members)

        # Aggregate the required values, adding the member names during the process
        opinion_sum: float = 0.0
        radicalisation_count: int = 0
        benefit_count: int = 0
        silenced_count: int = 0
        susceptibility_sum: float = 0.0
        hierarchy_weighting_total: float = 0.0
        personality_counts: dict[str, int] = {}

        for agent in members:
            self.members.append(agent.id)
            opinion_sum += agent.opinion
            susceptibility_sum += agent.social_susceptibility
            hierarchy_weighting_total += agent.social_weightings[hierarchy]
            if agent.radicalised:
                radicalisation_count += 1
            if agent.personal_benefit:
                benefit_count += 1
            if agent.is_silenced[hierarchy]:
                silenced_count += 1
            _ = personality_counts.setdefault(agent.personality, 0)
            personality_counts[agent.personality] += 1

        # Calculate the final attributes and set them
        self.aggregate_opinion = opinion_sum / len(members)
        self.radicalisation_rate = radicalisation_count / len(members)
        self.member_benefit_rate = benefit_count / len(members)
        self.silencing_rate = silenced_count / len(members)
        self.aggregate_susceptibility = susceptibility_sum / len(members)
        self.aggregate_hierarchy_weighting = hierarchy_weighting_total / len(members)
        self.predominant_personality = max(personality_counts, key=lambda x: personality_counts[x])

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
        Dynamically add an attribute to this Group object. If "value" is passed, an explicit initial value is given;
        if "mean" and "var" are passed through parameters, a value is generated from a random distribution.
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
        :param distribution: The random distribution that will be used to generate the value.
        :type distribution: str, optional
        :param overwrite: A flag indicating if the added attribute should override any existing attributes of the same name.
        :type overwrite: bool, optional
        :raises ValueError: If no valid value or distribution parameters are input, the attribute cannot be added.
        :raises UserWarning: If overwrite is explicitly False but the attribute is existing, a warning is raised without completing the operation.
        """
        if value is None and distribution is None:
            raise ValueError("Either explicit 'value' or distribution and valid distribution parameters are expected when adding Group attributes.")

        if not overwrite and name in self.__dict__:
            # Raise a warning but do not change any attributes or crash the model if overwriting an existing attribute when overwrite=False
            warnings.warn(
                f"WARNING: Attempting to overwrite an existing Group attribute ({name}) without meaning to.",
                category=UserWarning,
            )
        else:
            if value is not None:
                # Assume a given explicit value always overrides (mean, sdev)
                self.__dict__[name] = value
            elif distribution is not None:
                self.__dict__[name] = draw_random_value(
                    distribution, parameters=parameters,
                )
        return None

    def get_attribute(self, name: str) -> T | None:
        """
        Return any existing or dynamically added attribute held by the Group object.

        :param name: The name of the attribute to get.
        :type name: str
        :raises UserWarning: If the attribute does not exist.
        :return: The value stored for the attribute.
        :rtype: Any
        """
        attribute: T | None = self.__dict__.get(name)
        if attribute is None:
            warnings.warn(
                f"WARNING: Attempting to get a group attribute ({name}) which doesn't exist.",
                category=UserWarning,
            )
        return attribute

    def store_previous_opinion(self) -> None:
        """
        A setter method that stores the Group's current opinion into the previous opinion.
        """
        self.previous_opinion = self.aggregate_opinion
        return None

    def get_num_members(self) -> int:
        """
        A getter method that reports the number of members that are contained in this Group.

        :return: The number of agents that are members of this Group.
        :rtype: int
        """
        return len(self.members)

    def recalculate_aggregate_opinion(self, member_opinions: list[float]) -> None:
        """
        A setter method that will calculate a fixed aggregate opinion value from the
        currently held opinion values of members.

        :param member_opinions: The currently held opinion values of group members.
        :type member_opinions: list[float]
        :raises ValueError: If the input list is not the same size as the group.
        """
        if len(member_opinions) != self.get_num_members():
            raise ValueError("The number of member opinions input does not match the number of group members")
        opinion_sum: float = sum(member_opinions)
        self.aggregate_opinion = opinion_sum / self.get_num_members()
        return None

    def recalculate_radicalisation_rate(self, member_radicalisations: list[bool]) -> None:
        """
        A setter method that will calculate a fixed radicalisation rate value from
        the current radicalisation statuses of group members.

        :param member_radicalisations: The current radicalisation status for each group member.
        :type member_radicalisations: list[bool]
        :raises ValueError: If the input list is not the same size as the group.
        """
        if len(member_radicalisations) != self.get_num_members():
            raise ValueError("The number of radicalisation statuses input does not match the number of group members")
        radical_count: int = 0
        for radicalisation in member_radicalisations:
            if radicalisation:
                radical_count += 1
        self.radicalisation_rate = float(radical_count / len(member_radicalisations))
        return None

    def recalculate_member_benefit_rate(self, member_benefits: list[bool]) -> None:
        """
        A setter method that will calculate a fixed member benefit rate value from
        the current personal benefit statuses of group members.

        :param member_benefits: The current personal benefit status for each group member.
        :type member_benefits: list[bool]
        :raises ValueError: If the input list is not the same size as the group.
        """
        if len(member_benefits) != self.get_num_members():
            raise ValueError("The number of benefit statuses input does not match the number of group members")
        benefit_count: int = 0
        for benefit in member_benefits:
            if benefit:
                benefit_count += 1
        self.member_benefit_rate = float(benefit_count / len(member_benefits))
        return None

    def recalculate_silencing_rate(self, members_silenced: list[bool]) -> None:
        """
        A setter method that will calculate a fixed silencing rate value from the current
        opinion silencing statuses of group members.

        :param members_silenced: The current opinion silencing status for each group member.
        :type members_silenced: list[bool]
        :raises ValueError: If the input list is not the same size as the group.
        """
        if len(members_silenced) != self.get_num_members():
            raise ValueError("The number of silencing statuses input does not match the number of group members")
        silenced_count: int = 0
        for silenced in members_silenced:
            if silenced:
                silenced_count += 1
        self.silencing_rate = float(silenced_count / len(members_silenced))
        return None

    def recalculate_hierarchy_weighting(self, member_weightings: list[float]) -> None:
        """
        A setter method that will calculate a fixed member hierarchy weighting value from
        the current hierarchy weightings assigned by each group member.

        :param member_weightings: The current hierarchy weightings assigned by each group member.
        :type member_weightings: list[float]
        :raises ValueError: If the input list is not the same size as the group.
        """
        if len(member_weightings) != self.get_num_members():
            raise ValueError("The number of hierarchy weightings input does not match the number of group members")
        total_weighting: float = sum(member_weightings)
        self.aggregate_hierarchy_weighting = float(total_weighting / len(member_weightings))
        return None

    def determine_predominant_personality(self, member_personalities: list[str]) -> None:
        """
        A setter method that will determine what the predominant personality type amongst
        group members is.

        :param member_personalities: The personality types for each group member.
        :type member_personalities: list[str]
        :raises ValueError: If the input list is not the same size as the group or one or more personalities are invalid.
        """
        if len(member_personalities) != self.get_num_members():
            raise ValueError("The number of personality types input does not match the number of group members")
        personality_counts: dict[str, int] = {}
        for personality in member_personalities:
            if personality not in PERSONALITIES:
                raise ValueError("One or more input personality types are unsupported")
            _ = personality_counts.setdefault(personality, 0)
            personality_counts[personality] += 1
        dict_mode: str = max(personality_counts, key=lambda x: personality_counts[x])
        self.predominant_personality = dict_mode
        return None

    def change_aggregate_opinion(self, opinion_delta: float) -> float:
        """
        A setter method that changes the Group's aggregate opinion by a given delta value.

        This will not cause changes to the opinions of member agents to create the desired aggregate
        opinion -- It should be handled from within the parent model by using the returned
        per-agent opinion delta value.

        :param opinion_delta: The delta value by which to shift the Group's aggregate opinion.
        :type opinion_delta: float
        :raises TypeError: If opinion_delta is not a float.
        :raises AttributeError: If the aggregate_opinion has not been initialised yet.
        :return: The per-agent opinion_delta value that must be applied to each member.
        :rtype: float
        """
        if not isinstance(opinion_delta, float):
            raise TypeError("opinion_delta must be a float")
        if not hasattr(self, "aggregate_opinion"):
            raise AttributeError("aggregate_opinion has not been initialised for this group")

        per_agent_delta: float

        if self.aggregate_opinion + opinion_delta > OPINION_MAX:
            # Set the delta value to the difference between the maximum and the current aggregate opinion (which will be lower than opinion_delta)
            per_agent_delta = OPINION_MAX - self.aggregate_opinion
        elif self.aggregate_opinion + opinion_delta < -OPINION_MAX:
            # Same as above, but using -OPINION_MAX
            per_agent_delta = -OPINION_MAX - self.aggregate_opinion
        else:
            # Simply use the raw opinion_delta (each individual opinion shifted by the delta will cause the aggregate opinion
            # to shift by the same delta on average)
            per_agent_delta = opinion_delta

        # Change the group's aggregate opinion in the meantime
        self.aggregate_opinion += opinion_delta

        # Constrain the value back to the valid range
        if self.aggregate_opinion < -OPINION_MAX:
            self.aggregate_opinion = -OPINION_MAX
        elif self.aggregate_opinion > OPINION_MAX:
            self.aggregate_opinion = OPINION_MAX

        return per_agent_delta

    def change_radicalisation_rate(self, rate_delta: float) -> dict[str, bool]:
        """
        A setter method that changes the Group's radicalisation rate by a given delta value.

        This will not cause changes to the radicalisation status of member agents -- This should
        be handled from within the parent model by using the returned radicalisation status mapping.

        :param rate_delta: The value by which to shift the radicalisation rate of the group.
        :type rate_delta: float
        :raises TypeError: If the rate delta is not a float.
        :raises AttributeError: If the radicalisation_rate has not yet been initialised.
        :return: A <Member ID : radicalisation status> mapping that reports the new radicalisation status for each member.
        :rtype: dict[str, bool]
        """
        # Check that radicalisation rate has been initialised
        if not hasattr(self, "radicalisation_rate"):
            raise AttributeError("radicalisation_rate has not yet been initialised for this group")
        if not isinstance(rate_delta, float):
            raise TypeError("rate_delta must be a float value")

        # Apply the delta
        self.radicalisation_rate += rate_delta

        # Constrain back to a valid proportion
        if self.radicalisation_rate < 0.0:
            self.radicalisation_rate = 0.0
        elif self.radicalisation_rate > 1.0:
            self.radicalisation_rate = 1.0

        # Determine the number of radicalised agents for the new rate
        new_radicalisation_count: int = ceil(self.radicalisation_rate * self.get_num_members())
        selected_indices: list[int] = rd.sample(list(range(self.get_num_members())), k=new_radicalisation_count)

        output_dict: dict[str, bool] = {}

        for idx, member in enumerate(self.members):
            if idx in selected_indices:
                output_dict[member] = True
            else:
                output_dict[member] = False

        self.recalculate_radicalisation_rate(list(output_dict.values()))

        return output_dict

    def change_rw_distribution(self, parameters: tuple[float, float]) -> None:
        """
        A setter method that change the group's explicit random walk parameters for its social hierarchy.

        :param parameters: The new (mean, variance) for the random walk's gaussian distribution.
        :type parameters: tuple[float, float]
        :raises TypeError: If the input contains any invalid data types.
        """
        if not isinstance(parameters, tuple) or not isinstance(parameters[0], float) or not isinstance(parameters[1], float):
            raise TypeError("parameters must be a (float, float) tuple")
        self.rw_distribution = parameters
        return None

    def change_opinion_rw(self, rw_params: tuple[float, float]) -> None:
        """
        A setter method that changes the group's explicit opinion random walk parameters.

        :param rw_params: The new (mean, variance) for the random walk's gaussian distribution.
        :type rw_params: tuple[float, float]
        :raises TypeError: If rw_params is not a (float, float) tuple.
        """
        if not isinstance(rw_params, tuple) or not isinstance(rw_params[0], float) or not isinstance(rw_params[1], float):
            raise TypeError("rw_params must be a (float, float) tuple")
        self.opinion_rw = rw_params
        return None

    def change_predominant_personality(self, personality: str) -> dict[str, str]:
        """
        A setter method that changes the group's predominant personality type.

        This will not cause changes to the personalities of member agents to create the desired predominant
        type -- This should be handled from within the parent model by using the returned per-agent
        personality mapping.

        :param personality: The personality type that should be the new predominant personality.
        :type personality: str
        :raises TypeError: If personality is not a string.
        :raises ValueError: If the personality type is not currently supported.
        :return: A <Member ID : personality type> mapping reporting what each member's new personality type should be.
        :rtype: dict[str, str]
        """
        # Data checks
        if not isinstance(personality, str):
            raise TypeError("The input personality must be a string")
        if personality not in PERSONALITIES:
            raise ValueError("The input personality is not currently supported")

        member_personalities: dict[str, str] = {member: "" for member in self.members}

        # No change is actually ocurring, return empty strings for each member (which will be treated as 'no change' in outer functions)
        if personality == self.predominant_personality:
            return member_personalities
        else:
            self.predominant_personality = personality
            new_personalities: list[str] = make_list_with_mode(PERSONALITIES, self.predominant_personality, n=self.get_num_members())

            # Set the new personality for each member in simple numerical order
            for idx, member in enumerate(member_personalities.keys()):
                member_personalities[member] = new_personalities[idx]

        return member_personalities

    def change_benefit_rate(self, rate_delta: float) -> dict[str, bool]:
        """
        A setter method that changes the group's member_benefit_rate by a given delta value.

        This will not cause changes to the personal benefit status of member agents to create the desired
        benefit rate -- This should be handled from within the parent model by using the returned per-agent
        personal benefit mapping.

        :param rate_delta: The value by which to shift the group's member_benefit_rate.
        :type rate_delta: float
        :raises AttributeError: If the member benefit rate has not yet been initialised for this group.
        :raises TypeError: If the rate delta is not a float
        :return: A <Member ID : personal benefit flag> mapping that defines each member agent's new personal benefit status.
        :rtype: dict[str, bool]
        """
        # Checks
        if not hasattr(self, "member_benefit_rate"):
            raise AttributeError("member_benefit_rate has not yet been initialised for this group")
        if not isinstance(rate_delta, float):
            raise TypeError("rate_delta must be a float value")

        # Apply the delta
        self.member_benefit_rate += rate_delta

        # Constrain back to a valid proportion
        if self.member_benefit_rate < 0.0:
            self.member_benefit_rate = 0.0
        elif self.member_benefit_rate > 1.0:
            self.member_benefit_rate = 1.0

        # Determine the number of agents who experience personal benefit for the new rate
        new_benefit_count: int = ceil(self.member_benefit_rate * self.get_num_members())
        selected_indices: list[int] = rd.sample(list(range(self.get_num_members())), k=new_benefit_count)

        output_dict: dict[str, bool] = {}

        for idx, member in enumerate(self.members):
            if idx in selected_indices:
                output_dict[member] = True
            else:
                output_dict[member] = False

        self.recalculate_member_benefit_rate(list(output_dict.values()))

        return output_dict

    def change_silencing_rate(self, rate_delta: float) -> dict[str, bool]:
        """
        A setter method that changes the Group's silencing rate by a given delta value.

        This will not cause changes to the silencing status of member agents for the relevant hierarchy
        to reach the desired silencing rate -- This should be handled from within the parent model by
        using the returned per-agent is_silenced mapping.

        :param rate_delta: The value by which to shift the silencing_rate.
        :type rate_delta: float
        :raises TypeError: If the rate delta is not a float.
        :raises AttributeError: If the silencing_rate has not yet been initialised.
        :return: A <Member ID : is_silenced flag> mapping that specifies which agents are silenced in this group's hierarchy.
        :rtype: dict[str, bool]
        """
        # Checks
        if not hasattr(self, "silencing_rate"):
            raise AttributeError("silencing_rate has not yet been initialised for this group")
        if not isinstance(rate_delta, float):
            raise TypeError("rate_delta must be a float value")

        # Apply the delta
        self.silencing_rate += rate_delta

        # Constrain back to a valid proportion
        if self.silencing_rate < 0.0:
            self.silencing_rate = 0.0
        elif self.silencing_rate > 1.0:
            self.silencing_rate = 1.0

        # Determine the number of agents who have their opinions silenced for the new rate
        new_silenced_count: int = ceil(self.silencing_rate * self.get_num_members())
        selected_indices: list[int] = rd.sample(list(range(self.get_num_members())), k=new_silenced_count)

        output_dict: dict[str, bool] = {}

        for idx, member in enumerate(self.members):
            if idx in selected_indices:
                output_dict[member] = True
            else:
                output_dict[member] = False

        self.recalculate_silencing_rate(list(output_dict.values()))

        return output_dict

    def change_aggregate_hierarchy_weighting(self, weighting_delta: float) -> float:
        """
        A setter method that changes the Group's aggregate hierarchy weighting by a given delta value.

        This will not cause changes to the weightings of member agents to create the desired aggregate
        hierarchy weighting -- This should be handled from within the parent model by using the returned
        per-agent weighting delta.

        :param weighting_delta: The value by which to shift the aggregate hierarchy weighting.
        :type weighting_delta: float
        :raises TypeError: If the weighting delta is not a float.
        :raises AttributeError: If the aggregate_hierarchy_weighting has not yet been initialised.
        :return: The per-agent weighting delta that must be applied to each member.
        :rtype: float
        """
        if not isinstance(weighting_delta, float):
            raise TypeError("weighting_delta must be a float")
        if not hasattr(self, "aggregate_hierarchy_weighting"):
            raise AttributeError("aggregate_hierarchy_weighting has not been initialised for this group")

        per_agent_delta: float

        if self.aggregate_hierarchy_weighting + weighting_delta > SOCIAL_WEIGHTINGS_MAX:
            # Set the delta value to the difference between the maximum and the current aggregate opinion (which will be lower than opinion_delta)
            per_agent_delta = SOCIAL_WEIGHTINGS_MAX - self.aggregate_hierarchy_weighting
        elif self.aggregate_hierarchy_weighting + weighting_delta < -SOCIAL_WEIGHTINGS_MAX:
            # Same as above, but using -SOCIAL_WEIGHTINGS_MAX
            per_agent_delta = -SOCIAL_WEIGHTINGS_MAX - self.aggregate_hierarchy_weighting
        else:
            # Simply use the raw opinion_delta (each individual opinion shifted by the delta will cause the aggregate opinion
            # to shift by the same delta on average)
            per_agent_delta = weighting_delta

        # Change the group's aggregate opinion in the meantime
        self.aggregate_hierarchy_weighting += weighting_delta

        # Constrain the value back to the valid range
        if self.aggregate_hierarchy_weighting < -SOCIAL_WEIGHTINGS_MAX:
            self.aggregate_hierarchy_weighting = -SOCIAL_WEIGHTINGS_MAX
        elif self.aggregate_hierarchy_weighting > SOCIAL_WEIGHTINGS_MAX:
            self.aggregate_hierarchy_weighting = SOCIAL_WEIGHTINGS_MAX

        return per_agent_delta

    def set_index(self, index: int) -> None:
        """
        A setter function to change the Group's index.

        :param index: The new index to assign to the group.
        :type index: int
        :raises TypeError: If the input index is not an int.
        """
        if not isinstance(index, int):
            raise TypeError("index must be an integer")
        self.index = index
        return None

    def set_max_size(self, max_size: int, no_limit: bool = False) -> None:
        """
        A setter function to change the maximum size of a Group.

        :param max_size: The maximum number of agents that can be members of this group.
        :type max_size: int
        :param no_limit: A flag indicating if the change should remove the group's size limit.
        :type no_limit: bool, optional
        :raises TypeError: If the input max_size is not an integer.
        :raises ValueError: If the input max_size is equal or less than zero.
        """
        if no_limit:
            # no_limit always overrides the max_size value (even if it would otherwise be invalid)
            self.max_size = -1
            return None

        if not isinstance(max_size, int):
            raise TypeError("max_size must be an integer")
        elif max_size <= 0:
            raise ValueError("max_size must be equal or greater to 1")
        self.max_size = max_size
        return None

    def set_hierarchy(self, hierarchy: str) -> None:
        """
        A setter function to change the hierarchy that the Group belongs to.

        :param hierarchy: The name of the hierarchy that the Group will belong in.
        :type hierarchy: str
        :raises TypeError: If the input hierarchy name is not a string.
        """
        if not isinstance(hierarchy, str):
            raise TypeError("hierarchy must be a string")
        self.hierarchy = hierarchy
        return None

    def set_cohesion(self, cohesion: str) -> None:
        """
        A setter function to change the group's cohesion type.

        :param cohesion: The new cohesion type to assign to the Group.
        :type cohesion: str
        :raises TypeError: If the input cohesion type is not a string.
        :raises ValueError: If the cohesion is not one of the supported types.
        """
        if not isinstance(cohesion, str):
            raise TypeError("cohesion must be a string")
        elif cohesion not in COHESIONS:
            raise ValueError(f"The cohesion type '{cohesion}' is not supported -- cannot change the Group's cohesion type")
        self.cohesion = cohesion
        return None

    def group_at_capacity(self) -> bool:
        """
        A function that reports if the group is at its maximum capacity.

        :return: A flag indicating if the group is at its maximum capacity.
        :rtype: bool
        """
        if self.max_size <= 0:
            # Values <= 0 are treated as dummy values that indicate the group has no maximum size
            return False
        elif len(self.members) >= self.max_size:
            return True
        return False

    def is_radicalised(self, threshold: float = COLLECTIVE_RAD_THRESH) -> bool:
        """
        A function that reports if the group is considered to be "radicalised" as a collective.

        :param threshold: The threshold that the radicalisation rate must surpass for the group to be considered radicalised.
        :type threshold: float, optional
        :raises AttributeError: If the radicalisation_rate has not been initialised.
        :return: A flag indicating if the group can be collectively considered to be radicalised.
        :rtype: bool
        """
        if not hasattr(self, "radicalisation_rate"):
            raise AttributeError("radicalisation_rate has not been initialised yet for this group")
        if self.radicalisation_rate >= threshold:
            return True
        return False

    def is_benefited(self, threshold: float = COLLECTIVE_BENEFIT_THRESH) -> bool:
        """
        A function that reports if the group is considered to experience personal benefit from the social contagion
        as a collective.

        :param threshold: The threshold that the member benefit rate must surpass for the group to be considered as benefited.
        :type threshold: float, optional
        :raises AttributeError: If the member_benefit_rate has not been initialised.
        :return: A flag indicating if the group can be collectively considered to be benefited.
        :rtype: bool
        """
        if not hasattr(self, "member_benefit_rate"):
            raise AttributeError("member_benefit_rate has not yet been initialised for this group")
        if self.member_benefit_rate >= threshold:
            return True
        return False

    def is_silenced(self, threshold: float = COLLECTIVE_SILENCING_THRESH) -> bool:
        """
        A function that reports if the group is considered to be silenced in its hierarchy as a collective.

        :param threshold: The threshold that the silencing rate must surpass for the group to be considered as silenced.
        :type threshold: float, optional
        :raises AttributeError: If silencing_rate has not been initialised.
        :return: A flag indicating if the group can be collectively considered to be silenced.
        :rtype: float
        """
        if not hasattr(self, "silencing_rate"):
            raise AttributeError("silencing_rate has not yet been initialised for this group")
        if self.silencing_rate >= threshold:
            return True
        return False

    def add_member(self, member: str, force: bool = False) -> None:
        """
        Add a new member to this group (using the Agent's ID as the identifier).

        :param member: The ID of the Agent being added as a group member.
        :type member: str
        :param force: A flag indicating if this call should disregard the maximum group size.
        :type force: bool, optional
        :raises TypeError: If the input value is not a string.
        :raises UserWarning: If the group is already at maximum capacity
        """
        if self.group_at_capacity() and not force:
            warnings.warn(
                f"WARNING: Attempted to add a member to group {self.id} which is already at maximum capacity",
                category=UserWarning,
            )
            return None

        if not isinstance(member, str):
            raise TypeError("member must be a string value")
        self.members.append(member)
        return None

    def add_members(self, members: Iterable[str], force: bool = False) -> None:
        """
        Add multiple new members to this group (using the Agent IDs as the identifiers).

        :param members: An iterable containing the Agent IDs of the members to be added.
        :type members: Iterable[str]
        :param force: A flag indicating if this call should disregard the maximum group size.
        :type force: bool, optional
        :raises TypeError: If any of the input members are not string values.
        :raises UserWarning: If the group is already at maximum capacity.
        """
        for member in members:
            if self.group_at_capacity() and not force:
                warnings.warn(
                    f"WARNING: Attempted to add a member to group {self.id} which is already at maximum capacity",
                    category=UserWarning,
                )
                return None

            if not isinstance(member, str):
                raise TypeError("One or more members have been passed as an invalid, non-string data type")
            else:
                self.members.append(member)
        return None

    def step(self, weighting_rw: tuple[float, float], opinion_rw: tuple[float, float]) -> dict[str, Any]:
        """
        Step the individual group object:
            1. Handle dynamic hierarchy weightings
            2. Handle the stochastic opinion changes

        :param weighting_rw: The (mean, variance) defining the random walk distribution of the aggregate hierarchy weighting.
        :type weighting_rw: tuple[float, float]
        :param opinion_rw: The (mean, variance) defining the random walk distribution of the aggregate opinion.
        :type opinion_rw: tuple[float, float]
        :return: A <label : information> mapping that provides all relevant information from the group's step to the parent model.
        :rtype: dict[str, Any]
        """
        output_dict: dict[str, Any] = {}

        hierarchy_rw_info: tuple[str, float] = self.evolve_hierarchy(weighting_rw)
        output_dict["hierarchy_rw_info"] = hierarchy_rw_info

        opinion_rw_info: tuple[str, float] = self.stochastic_opinion(opinion_rw)
        output_dict["opinion_rw_info"] = opinion_rw_info

        return output_dict

    def update(self, opinion_silenced: float, negation_ocurred: bool) -> tuple[str, dict[str, bool]]:
        """
        Updates the internal state of the group after the model has stepped:
            1. Updates whether the group is silenced within its hierarchy
            2. Inverts the group's aggregate opinion if opinion negation ocurred

        :param opinion_silenced: A delta value indicating by how much the group's silencing_rate has shifted.
        :type opinion_silenced: float
        :param negation_ocurred: A flag indicating if opinion negation has ocurred in the current iteration.
        :type negation_ocurred: bool
        :raises RuntimeError: If the group has not yet been initialised appropriately.
        :raises TypeError: If either of the input parameters are of an incorrect data type.
        :return: The group's hierarchy, and which group members have become silenced.
        :rtype: tuple[str, dict[str, bool]]
        """
        # Check for initialisation
        if not hasattr(self, "aggregate_opinion"):
            raise AttributeError("The group for which an update is being attempted has not yet been initialised")

        # Type checking
        if not isinstance(opinion_silenced, bool) or not isinstance(negation_ocurred, bool):
            raise TypeError("opinion_silenced and negation_ocurred must both be boolean values")

        # Update is_silenced
        members_silenced: dict[str, bool] = self.change_silencing_rate(opinion_silenced)
        if negation_ocurred:
            # Invert the Group's aggregate opinion
            self.aggregate_opinion *= -1.0
        return self.hierarchy, members_silenced

    def opinion_silencing(self, estimated_opinion_climate: float, silencing_threshold: float | None = None) -> tuple[bool, float]:
        """
        Determines if members in the group will become silenced in their hierarchy based on their collective attributes.

        If no silencing threshold has been passed, the group's aggregate susceptibility is used as the threshold instead.

        Using the information returned by this function, the parent model will then have to handle it appropriately and make a call
        to :meth:`~gatoh.groups.Group.update` with the defined silencing_rate delta due to the reported absolute difference value.

        :param estimated_opinion_climate: The opinion climate perceived by the Group in its hierarchy.
        :type estimated_opinion_climate: float
        :param silencing_threshold: The silencing threshold that must be surpassed for silencing to occur.
        :type silencing_threshold: float, optional
        :raises RuntimeError: If opinion silencing is called before the group has been initialised appropriately.
        :raises TypeError: If the input parameters contain any invalid data types.
        :return: A pair of values indicating if silencing occurs, and the absolute difference between the perceived opinion climate and the group's aggregate opinion.
        :rtype: tuple[bool, float]
        """
        # Check that the group has been initialised
        if not hasattr(self, "aggregate_opinion"):
            raise RuntimeError("The group for which opinion silencing is being determined has not yet been initialised")

        # Type checking
        if not isinstance(estimated_opinion_climate, float):
            raise TypeError("estimated_opinion_climate must be a float")
        if silencing_threshold is not None and not isinstance(silencing_threshold, float):
            raise TypeError("silencing_threshold must either be a float or None")

        # It is assumed that a radicalised group will never silence itself regardless of the perceived opinion climate
        if self.is_radicalised():
            return False, 0.0

        threshold: float
        if silencing_threshold is not None:
            threshold = silencing_threshold
        else:
            threshold = self.aggregate_susceptibility

        absolute_difference: float = 0.0

        if self.predominant_personality in ["neutral", "rational", "erratic"]:
            # Cases where opinion silencing will be less influenced by the surrounding opinion climate
            absolute_difference = abs(estimated_opinion_climate - self.aggregate_opinion) * OPINION_SILENCING_MODIFIER
        elif self.predominant_personality in ["impulsive", "social"]:
            # Cases where opinion silencing will be much more influenced by the surrounding opinion climate
            absolute_difference = abs(estimated_opinion_climate - self.aggregate_opinion)

        return absolute_difference > threshold, absolute_difference

    def opinion_negation(self, absolute_difference: float, threshold: float) -> bool:
        """
        Checks if a group has experienced sufficiently 'overwhelming' social pressure in its hierarchy leading to a complete
        reversal of its members' opinions.

        :param absolute_difference: The absolute difference between the perceived opinion climate and the group's aggregate opinion.
        :type absolute_difference: float
        :param threshold: A global model threshold that has been specified for this effect to occur.
        :type threshold: float
        :raises RuntimeError: If the group has not yet been initialised appropriately.
        :raises TypeError: If the input parameters contain any invalid data types.
        :return: A flag indicating if the group's opinion experienced a total negation.
        :rtype: bool
        """
        # Check that the group is initialised
        if not hasattr(self, "aggregate_susceptibility"):
            raise RuntimeError("The group for which opinion negation is being determined has not yet been initialised")

        # Type checking
        if not isinstance(absolute_difference, float):
            raise TypeError("absolute_difference must be a float")
        if not isinstance(threshold, float):
            raise TypeError("threshold must be a float")

        # It is assumed that a radicalised Group will never experience a total opinion reversal regardless of the perceived opinion climate
        if self.is_radicalised():
            return False

        negation_strength: float = absolute_difference

        # Multiplication by (susceptibility * hierarchy weighting) will always decrease negation strength, whilst division will always increase it
        if self.predominant_personality in ["neutral", "rational"]:
            # Cases where opinion negation is less likely to occur
            if self.aggregate_susceptibility * self.aggregate_hierarchy_weighting != 0:
                negation_strength *= self.aggregate_susceptibility * self.aggregate_hierarchy_weighting
        elif self.predominant_personality in ["erratic", "impulsive", "social"]:
            # Cases where opinion negation is more likely to occur
            if self.aggregate_susceptibility * self.aggregate_hierarchy_weighting != 0:
                negation_strength /= self.aggregate_susceptibility * self.aggregate_hierarchy_weighting

        return negation_strength > threshold

    def evolve_hierarchy(self, weighting_rw: tuple[float, float]) -> tuple[str, float]:
        """
        Experimental function that aims to model the constantly evolving 'intrinsic value' that people palce on
        the social hierarchies that the y belong in over time.

        :param weighting_rw: The (mean, variance) for a normal distribution that will be used to draw random walk values.
        :type weighting_rw: tuple[float, float]
        :raises TypeError: If the input parameter is of an incorrect type.
        :return: The group's hierarchy and the per-agent delta value that must be applied to each agent's hierarchy weighting.
        :rtype: tuple[str, float]
        """
        # Initial type check
        if not isinstance(weighting_rw, tuple):
            raise TypeError("weighting_rw must be a tuple")

        # Inner type check
        if not isinstance(weighting_rw[0], float) or not isinstance(weighting_rw[1], float):
            raise TypeError("One or more of the values in weighting_rw are not a float")

        rw_result: float = value_rw_delta(self.aggregate_hierarchy_weighting, weighting_rw[0], weighting_rw[1])

        # Change the value (all appropriate checks are handled here)
        per_agent_delta: float = self.change_aggregate_hierarchy_weighting(rw_result)

        return self.hierarchy, per_agent_delta

    def stochastic_opinion(self, opinion_rw: tuple[float, float]) -> tuple[str, float]:
        """
        Determine the direction and magnitude of a shift in the Group's aggregate opinion and apply it.

        This is representative of opinion changes in real social networks in which individual opinions may increase or decrease
        away from the aggregate mean (independent of external influences).

        :param opinion_rw: A (mean, variance) pair which parametrises the Gaussian distribution used for stochastic opinion shift.
        :type opinion_rw: tuple[float, float]
        :raises RuntimeError: If the group has not yet been initialised appropriately.
        :raises TypeError: If the input contains any invalid data types.
        :return: The group's hierarchy and per-member delta that must be applied to each member Agent to shift the aggregate opinion by the drawn delta value.
        :rtype: tuple[str, float]
        """
        # Check that the group is initialised
        if not hasattr(self, "aggregate_opinion"):
            raise RuntimeError("The group for which the stochastic opinion is being determined has not yet been initialised")

        # Data type checks
        if not isinstance(opinion_rw, tuple):
            raise TypeError("opinion_rw must be a tuple")
        if not isinstance(opinion_rw[0], float) or not isinstance(opinion_rw[1], float):
            raise TypeError("One or both of the values in opinion_rw are invalid data types -- both must be floats")

        rw_result: float = value_rw_delta(self.aggregate_opinion, opinion_rw[0], opinion_rw[1])

        # Change the value (all appropriate checks are handled here)
        per_agent_delta: float = self.change_aggregate_opinion(rw_result)

        return self.hierarchy, per_agent_delta

    def stochastic_personality_change(self, personality_probs: PersonalityProbs | None = None) -> dict[str, str]:
        """
        Calls to this function are primarily meant to originate from :meth:`~gatoh.groups.Group.life_events`.

        Will redraw a valid predominant personality types from the ones that have been defined, and then change the group's
        predominant personality to this new type.

        Following this, the minimum proportion of members which need to hold this personality is determined, and the members'
        personalities are randomly reassigned so that a proportion equal or greater to the minimum needed for this to be the
        predominant personality type are present.

        :param personality_probs: Specific per-personality type probabilities to be used for drawing the new predominant personality type.
        :type personality_probs: dict[str, float]
        :raises TypeError: If a non-float probability is supplied.
        :raises KeyError: If an unsupported personality type is passed.
        :return: A <member ID : personality> mapping that specifies which (if any) member personalities have changed.
        :rtype: dict[str, str]
        """
        previous_personality: str = self.predominant_personality
        member_personalities: dict[str, str] = {member: "" for member in self.members}

        if personality_probs is None:
            self.predominant_personality = draw_personality()
        else:
            personality_flags: list[str] = list(personality_probs.keys())
            personality_p: list[float] = []
            for key, value in personality_probs.items():
                if isinstance(value, float):
                    personality_p.append(value)
                else:
                    raise TypeError("A non-float probability was supplied in probability_probs when trying to determine a stochastic personality change in a group")
                if key not in PERSONALITIES:
                    raise KeyError("An unsupported personality was specified in personality_probs when trying to determine a stochastic personality change in a group")
            chosen_personality = rd.choices(personality_flags, weights=personality_p, k=1)
            self.predominant_personality = chosen_personality[0]

        # No change has ocurred, then return empty strings for each member (this will be treated as no change needed in outer functions)
        if self.predominant_personality == previous_personality:
            return member_personalities
        else:
            new_personalities: list[str] = make_list_with_mode(PERSONALITIES, self.predominant_personality, n=self.get_num_members())

            # Set the new personality for each member in simple numerical order
            for idx, member in enumerate(member_personalities.keys()):
                member_personalities[member] = new_personalities[idx]

            return member_personalities

    def stochastic_benefit_change(self) -> dict[str, bool]:
        """
        Calls to this function are primarily meant to originate from :meth:`~gatoh.groups.Group.life_events`.

        Will randomly select a new number of group members that should have personal benefit, and then randomly
        assigns these benefit statuses to the group members.

        :raises RuntimeError: If the group has not yet been initialised appropriately.
        :return: A <member ID : benefit flag> mapping that specifies the new personal benefit flags for each agent member.
        :rtype: dict[str, bool]
        """
        # Check that the group has been initialised
        if not hasattr(self, "member_benefit_rate"):
            raise RuntimeError("The group for which a stochastic benefit change is being determined has not yet been initialised")

        new_benefit_count: int = rd.randint(0, self.get_num_members() - 1)
        selected_indices: list[int] = rd.sample(list(range(self.get_num_members())), k=new_benefit_count)

        output_dict: dict[str, bool] = {}

        for idx, member in enumerate(self.members):
            if idx in selected_indices:
                output_dict[member] = True
            else:
                output_dict[member] = False

        self.recalculate_member_benefit_rate(list(output_dict.values()))

        return output_dict

    def stochastic_radicalisation_change(self) -> dict[str, bool]:
        """
        Calls to this function are primarily meant to originate from :meth:`~gatoh.groups.Group.life_events`.

        Will randomly select a new number of group members that are radicalised, and then randomly assigns these
        radicalisation statuses to the group members.

        :raises RuntimeError: If the group has not yet been initialised appropriately.
        :return: A <member ID : radicalisation status> mapping that specifies the new radicalisation status for each agent member.
        :rtype: dict[str, bool]
        """
        # Check that the group has been initialised
        if not hasattr(self, "radicalisation_rate"):
            raise RuntimeError("The group for which a stochastic radicalisation change is being determined has not yet been initialised")

        new_radicalisation_count: int = rd.randint(0, self.get_num_members() - 1)
        selected_indices: list[int] = rd.sample(list(range(self.get_num_members())), k=new_radicalisation_count)

        output_dict: dict[str, bool] = {}

        for idx, member in enumerate(self.members):
            if idx in selected_indices:
                output_dict[member] = True
            else:
                output_dict[member] = False

        self.recalculate_radicalisation_rate(list(output_dict.values()))

        return output_dict

    def stochastic_silencing_change(self) -> dict[str, bool]:
        """
        Calls to this function are primarily meant to originate from :meth:`~gatoh.groups.Group.life_events`.

        Will randomly select a new number of group members that are silenced, and then randomly assigns these
        silencing statuses to the group members.

        :raises RuntimeError: If the group has not yet been initialised appropriately.
        :return: A <Member ID : silencing status> mapping that specifies the new silencing status for each agent member.
        :rtype: dict[str, bool]
        """
        # Check that the group has been initialised
        if not hasattr(self, "silencing_rate"):
            raise RuntimeError("The group for which a stochastic silencing change is being determined has not yet been initialised")

        new_silenced_count: int = rd.randint(0, self.get_num_members() - 1)
        selected_indices: list[int] = rd.sample(list(range(self.get_num_members())), k=new_silenced_count)

        output_dict: dict[str, bool] = {}

        for idx, member in enumerate(self.members):
            if idx in selected_indices:
                output_dict[member] = True
            else:
                output_dict[member] = False

        self.recalculate_silencing_rate(list(output_dict.values()))

        return output_dict

    class LifeEventsDict(TypedDict):
        personality_thresh: NotRequired[float]
        personality_probs: NotRequired[PersonalityProbs]
        benefit_thresh: NotRequired[float]
        radicalisation_thresh: NotRequired[float]
        silencing_thresh: NotRequired[float]

    def life_events(
        self,
        personality_changes: bool = False,
        benefit_changes: bool = False,
        radicalisation_changes: bool = False,
        silencing_changes: bool = False,
        parameters: LifeEventsDict | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Experimental function that aims to model the ways in which behaviours change according to major random life events over time.

        :param personality_changes: A flag indicating if personality changes are allowed to occur due to life events.
        :type personality_changes: bool, optional
        :param benefit_changes: A flag indicating if changes in personal benefit statuses are allowed to occur due to life events.
        :type benefit_changes: bool, optional
        :param radicalisation_changes: A flag indicating if spontaneous changes to radicalisation statuses are allowed to occur due to life events.
        :type radicalisation_changes: bool, optional
        :param silencing_changes: A flag indicating if spontaneous changes to opinion silencing statuses are allowed to occur due to life events.
        :type silencing_changes: bool, optional
        :param parameters: A <key, value> mapping which outlines all relevant threshold or other values used in this function.
        :type parameters: dict[str, Any], optional
        """
        member_changes: dict[str, dict[str, Any]] = {}

        if personality_changes:
            personality_results: dict[str, str]
            if parameters is not None:
                personality_thresh: float | None = parameters.get("personality_thresh")
                personality_probs: PersonalityProbs | None = parameters.get("personality_probs")
                if (personality_thresh is not None and rd.random() >= personality_thresh) or rd.random() >= PERSONALITY_THRESH:
                    personality_results = self.stochastic_personality_change(personality_probs=personality_probs)
                    member_changes["personality"] = personality_results
            elif rd.random() >= PERSONALITY_THRESH:
                personality_results = self.stochastic_personality_change()
                member_changes["personality"] = personality_results

        if benefit_changes:
            benefit_results: dict[str, bool]
            if parameters is not None:
                benefit_thresh: float | None = parameters.get("benefit_thresh")
                if (benefit_thresh is not None and rd.random() >= benefit_thresh) or rd.random() >= BENEFIT_THRESH:
                    benefit_results = self.stochastic_benefit_change()
                    member_changes["benefit"] = benefit_results
            elif rd.random() >= BENEFIT_THRESH:
                benefit_results = self.stochastic_benefit_change()
                member_changes["benefit"] = benefit_results

        if radicalisation_changes:
            radicalisation_results: dict[str, bool]
            if parameters is not None:
                radicalisation_thresh: float | None = parameters.get("radicalisation_thresh")
                if (radicalisation_thresh is not None and rd.random() >= radicalisation_thresh) or rd.random() >= RADICALISATION_THRESH:
                    radicalisation_results = self.stochastic_radicalisation_change()
                    member_changes["radicalisation"] = radicalisation_results
            elif rd.random() >= RADICALISATION_THRESH:
                radicalisation_results = self.stochastic_radicalisation_change()
                member_changes["radicalisation"] = radicalisation_results

        if silencing_changes:
            silencing_results: dict[str, bool]
            if parameters is not None:
                silencing_thresh: float | None = parameters.get("silencing_thresh")
                if (silencing_thresh is not None and rd.random() >= silencing_thresh) or rd.random() >= SILENCING_THRESH:
                    silencing_results = self.stochastic_silencing_change()
                    member_changes["silencing"] = silencing_results
            elif rd.random() >= SILENCING_THRESH:
                silencing_results = self.stochastic_silencing_change()
                member_changes["silencing"] = silencing_results

        return member_changes

    def __in__(self, iterable: Iterable[Group]) -> bool:
        """
        Determine if the Group is contained within an iterable of Groups.

        :param iterable: The Group objects in which membership is being determined.
        :type iterable: Iterable[Group]
        :return: A flag indicating if the Group is contained within the iterable.
        :rtype: bool
        """
        for group in iterable:
            if self == group:
                return True
        return False

    @override
    def __str__(self) -> str:
        """
        An override to what calling 'print()' on this object will output.

        :return: A printable representation of the Group object.
        :rtype: str
        """
        return f"Group {self.id} composed of members: {self.members}, and belonging to hierarchy {self.hierarchy} with an aggregate opinion of {self.aggregate_opinion}"


class GroupSet:
    """
    A collection of Group objects that maintains consistency for the Model.
    """

    def __init__(self) -> None:
        self.groups: list[Group] = []
        self.random: rd.Random = rd.Random()

    def save_groupset(self, directory_path: str) -> None:
        """
        Save the Group objects into a compressed subdirectory representing the saved GroupSet.

        :param directory_path: The path to the directory where the groupset subdirectory should be created.
        :type directory_path: str
        """
        subdirectory_path: str = f"{directory_path}/_groupset"

        # Removes the subdirectory if it already exists to allow for a new overwrite
        if os.path.exists(subdirectory_path):
            rmtree(subdirectory_path)

        # Create the _groupset subdirectory
        os.mkdir(subdirectory_path)

        group_save_paths: list[str] = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            saved_group_paths = {executor.submit(self.write_group_pickle, group, subdirectory_path): group.id for group in self.groups}
            for future in concurrent.futures.as_completed(saved_group_paths):
                group_id = saved_group_paths[future]
                try:
                    save_path = future.result()
                except Exception as exc:
                    print(f"Failed to write a pickle for group {group_id} with exception: {exc}")
                else:
                    group_save_paths.append(save_path)

        zip_path: str = f"{subdirectory_path}.zip"

        # Removes the zip file if it already exists to allow for a new overwrite
        if os.path.exists(zip_path):
            os.remove(zip_path)

        # Compress the subdirectory to minimise storage and encapsulate all the Groups into a single object
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=COMPRESS_LEVEL) as subdir_zip:
            for group_path in group_save_paths:
                subdir_zip.write(group_path, arcname=f"{os.path.basename(group_path)}")

        # Remove the uncompressed subdirectory if compression was successful
        if os.path.exists(zip_path):
            rmtree(subdirectory_path)

        return None

    def write_group_pickle(self, group: Group, subdirectory_path: str) -> str:
        """
        A helper function that allows for multithreading of :meth:`~gatoh.groups.GroupSet.save_groupset`.

        :param group: The group that is being saved.
        :type group: Group
        :param subdirectory_path: The path to the subdirectory in which the groups are being saved.
        :type subdirectory_path: str
        :return: The path to which the group pickle was saved.
        :rtype: str
        """
        group_save_path: str = f"{subdirectory_path}/_group_{group.id}.pkl"
        with open(group_save_path, "wb") as group_pickle:
            pickle.dump(group, group_pickle)
        return group_save_path

    def load_groupset(self, load_path: str) -> None:
        """
        Loads a GroupSet that has been saved following the same process as in the :meth:`~gatoh.groups.GroupSet.save_groupset` function.

        :param load_path: The path to the model's overall save directory.
        :type load_path: str
        :raises FileNotFoundError: If no valid groupset zip was found in the load path.
        """
        zip_load_path: str = f"{load_path}/_groupset.zip"

        if not os.path.exists(zip_load_path):
            raise FileNotFoundError(f"No saved GroupSet was found at the path: {zip_load_path}")

        # The path to the uncompressed groupset subdirectory
        subdirectory_path: str = f"{load_path}/_groupset"

        # Remove any existing subdirectory with the same name to replace it with the newly loaded one
        if os.path.isdir(subdirectory_path):
            rmtree(subdirectory_path)

        # Create the uncompressed directory
        os.mkdir(subdirectory_path)

        # Extract all the Group pickles to the uncompressed directory
        with zipfile.ZipFile(zip_load_path, mode="r", compression=zipfile.ZIP_DEFLATED, compresslevel=COMPRESS_LEVEL) as subdir_zip:
            subdir_zip.extractall(path=subdirectory_path)

        # Unpickle each Group object and add it to the GroupSet using multithreading
        with concurrent.futures.ThreadPoolExecutor() as executor:
            group_objects = {executor.submit(self.extract_group_pickle, group_pickle_name, subdirectory_path): group_pickle_name for group_pickle_name in os.listdir(subdirectory_path)}
            for future in concurrent.futures.as_completed(group_objects):
                group_pickle_path = group_objects[future]
                try:
                    group_object = future.result()
                except Exception as exc:
                    print(f"Failed to extract the pickled Group object at path {group_pickle_path} with exception: {exc}")
                else:
                    _ = self.add(group_object)

        return None

    def extract_group_pickle(self, group_pickle_name: str, subdirectory_path: str) -> Group:
        """
        A helper function that allows for multithreading of :meth:`~gatoh.groups.GroupSet.load_groupset`.

        :param group_pickle_name: The name of the pickled Group object file.
        :type group_pickle_name: str
        :param subdirectory_path: The root path to which the pickled group object was written.
        :type subdirectory_path: str
        :return: The unpickled group.
        :rtype: Group
        """
        group_pickle_path: str = f"{subdirectory_path}/{group_pickle_name}"
        with open(group_pickle_path, "rb") as group_pickle:
            group_object: Group = pickle.load(group_pickle)
        return group_object

    def __len__(self) -> int:
        """
        A method that defines how a GroupSet object checks its length.

        :return: The number of groups present in the GroupSet.
        :rtype: int
        """
        return len(self.groups)

    def __iter__(self) -> Iterator[Group]:
        """
        A method that defines how the GroupSet iterates over its Groups.

        :return: An iteration over all the Groups within the GroupSet.
        :rtype: Iterator[Group]
        """
        return self.groups.__iter__()

    def __in__(self, group: Group) -> bool:
        """
        A method defining how a GroupSet checks for a Group's membership.

        :param group: The specific Groups object to check for.
        :type group: Group
        :return: A flag indicating if the Group object is in the GroupSet.
        :rtype: bool
        """
        return self.id_in_groupset(group.id)

    def __contains__(self, group: Group) -> bool:
        """
        A secondary method defining how a GroupSet checks for a Group's membership.

        :param group: The specific Group object to check for.
        :type group: Group
        :return: A flag indicating if the specified Group object is in the GroupSet.
        :rtype: bool
        """
        return self.id_in_groupset(group.id)

    def __getitem__(self, item: int | slice) -> Group | list[Group]:
        """
        Retrieve a Group or slice of Groups from the GroupSet.

        :param item: The parameter for selecting the groups.
        :type item: int | slice
        :return: The selected group or slice of groups based on the specified item.
        :rtype: Group | list[Group]
        """
        return self.groups.__getitem__(item)

    def add(self, group: Group) -> int:
        """
        Adds a Group to the GroupSet.

        :param group: The Group object to be added.
        :type group: Group
        :return: The index of the newly added Group.
        :rtype: int
        """
        self.groups.append(group)
        self.groups[-1].index = len(self.groups) - 1
        return self.groups[-1].index

    def update_indices(self) -> None:
        """
        Iterate over the GroupSet and update the current Group object index values.
        """
        for idx, group in enumerate(self.groups):
            group.index = idx
        return None

    def id_in_groupset(self, group_id: str) -> bool:
        """
        Report if a group with the given unique ID exists in the groupset.

        :param group_id: The unique ID of the group that is being checked for.
        :type group_id: str
        :return: A flag indicating if an group with the input ID exists in the groupset or not.
        :rtype: bool
        """
        for group in self.groups:
            if group.id == group_id:
                return True
        return False

    def discard(self, group: Group) -> bool:
        """
        Removes a Group from the GroupSet which matches the input Group; does not return an error if the Group does not exist.

        :param group: The Group object that should be removed from the set.
        :type group: Group
        :return: A flag indicating if the Group was removed successfully or not.
        :rtype: bool
        """
        for idx, grp in enumerate(self.groups):
            if group == grp:
                left_half: list[Group] = self.groups[:idx]
                right_half: list[Group] = self.groups[idx + 1 :]

                self.groups = deepcopy(left_half) + deepcopy(right_half)

                # Manual garbage collection
                del left_half, right_half
                _ = gc.collect()

                self.update_indices()
                return True
        return False

    def group_at_index(self, index: int) -> Group | None:
        """
        Returns the Groups object at the given index in the GroupSet.

        :param index: The index within the GroupSet to inspect.
        :type index: int
        :raises UserWarning: If the input index is out of bounds, raise a warning and return None.
        :return: The Group object at the specified index.
        :rtype: Group
        """
        try:
            return self.groups[index]
        except IndexError:
            warnings.warn(
                f"WARNING: Index {index} is out of bounds for the GroupSet. Only {len(self.groups)} Groups have been created.",
                category=UserWarning,
            )
            return None

    def groups_at_indices(self, indices: list[int]) -> list[Group]:
        """
        Returns a list of all the group objects at the specified indices.

        :param indices: A list of all the indices for which the groups should be returned.
        :type indices: list[int]
        :raises UserWarning: If any of the input indices are out of bounds.
        :return: All of the group objects that correspond to the input indices.
        :rtype: list[Group]
        """
        groups_to_return: list[Group] = []
        for index in indices:
            try:
                group: Group = self.groups[index]
                groups_to_return.append(group)
            except IndexError:
                warnings.warn(
                    f"WARNING: Index {index} is out of bounds for the GroupSet. Only {len(self.groups)} Groups have been created.",
                    category=UserWarning,
                )
        return groups_to_return

    def get_group_by_id(self, id: str) -> Group:
        """
        Searches the GroupSet for a Group with the given id and returns its object if it exists.

        :param id: The id that was assigned to the Group object at creation.
        :type id: str
        :raises KeyError: If the input id does not exist in the GroupSet.
        :return: The Group object with the specified id.
        :rtype: Group
        """
        for group in self.groups:
            if group.id == id:
                return group

        raise KeyError(f"The Group with id '{id}' does not exist in the GroupSet -- unable to return a Group object.")

    def get_groups_by_ids(self, ids: list[str]) -> list[Group]:
        """
        Searches the GroupSet for Groups with the given ids and returns their objects if they all exist.

        :param ids: The ids that have been assigned to every Group object at creation.
        :type ids: list[str]
        :return: The Group objects with the specified ids.
        :rtype: list[Group]
        """
        groups_to_return: list[Group] = []
        for id in ids:
            groups_to_return.append(self.get_group_by_id(id))
        return groups_to_return

    def get_index(self, group: Group) -> int:
        """
        Returns the index within the GroupSet of the input Group object.

        :param group: The group whose index is being searched for.
        :type group: Group
        :raises KeyError: If the input Group does not exist in the GroupSet.
        :return: The index of the group within the GroupSet.
        :rtype: int
        """
        for idx, grp in enumerate(self.groups):
            if grp.id == group.id:
                return idx

        raise KeyError(f"The Group {group.id} does not exist in the GroupSet -- unable to return an index.")

    def get_indices(self, groups: list[Group]) -> list[int]:
        """
        Returns the indices within the GroupSet of the input Group objects.

        :param groups: The groups whose indices are being searched for.
        :type groups: list[Group]
        :return: The indices of the groups within the GroupSet.
        :rtype: list[int]
        """
        group_indices: list[int] = []
        for group in groups:
            group_indices.append(self.get_index(group))
        return group_indices

    def discard_index(self, index: int) -> bool:
        """
        Removes the Group at the specified index in the GroupSet; does not raise an error if the index is out of bounds.

        :param index: The index in the GroupSet which is to be removed.
        :type index: int
        :return: A flag indicating if the Group was removed successfully or not.
        :rtype: bool
        """
        if 0 < index < len(self.groups):
            left_half: list[Group] = self.groups[:index]
            right_half: list[Group] = self.groups[index + 1 :]

            self.groups = deepcopy(left_half) + deepcopy(right_half)
            del left_half, right_half

            self.update_indices()
            return True
        return False

    def remove(self, group: Group) -> bool:
        """
        Removes a group from the GroupSet which matches the input group; returning an error if such a Group does not exist.

        :param group: The group that should be removed from the set.
        :type group: Group
        :raises KeyError: If the input Group does not exist in the GroupSet.
        :return: A flag indicating that the Group was removed successfully.
        :rtype: bool
        """
        for idx, grp in enumerate(self.groups):
            if group == grp:
                left_half: list[Group] = self.groups[:idx]
                right_half: list[Group] = self.groups[idx + 1 :]

                self.groups = deepcopy(left_half) + deepcopy(right_half)
                del left_half, right_half

                self.update_indices()
                return True
        raise KeyError(f"Tried to remove a Group with id {group.id} that doesn't exist in the GroupSet")

    def remove_index(self, index: int) -> bool:
        """
        Removes the Group at the specified index in the GroupSet; raising an error if the index is out of bounds.

        :param index: The index in the GroupSet which is to be removed.
        :type index: int
        :raises IndexError: If the input index is out of bounds for the GroupSet.
        :return: A flag indicating that the Group was removed successfully.
        :rtype: bool
        """
        if 0 < index < len(self.groups):
            left_half: list[Group] = self.groups[:index]
            right_half: list[Group] = self.groups[index + 1 :]

            self.groups = deepcopy(left_half) + deepcopy(right_half)
            del left_half, right_half

            self.update_indices()
            return True
        raise IndexError(f"Tried to remove a Group at out of bounds index {index} from the GroupSet")

    def sample(self, n: int) -> list[Group]:
        """
        Randomly draw n groups from the GroupSet without replacement.

        :param n: The number of groups to sample.
        :type n: int
        :return: The groups sampled from the GroupSet.
        :rtype: list[Group]
        """
        sampled_groups: list[Group] = self.random.sample(self.groups, n)
        return deepcopy(sampled_groups)

    def get_total_members(self) -> int:
        """
        A getter function that iterates across all groups in the groupset and adds their member counts to
        report the total number of members that exist in the groupset.

        :return: The total number of members that exist in the grouspet.
        :rtype: int
        """
        total_members: int = 0
        for group in self.groups:
            total_members += group.get_num_members()
        return total_members

    def get_num_members(self, index: int) -> int:
        """
        A wrapper that used the group's getter function to report the number of members of that group.

        :param index: The index of the group for which the number of members is being inspected.
        :type index: int
        :raises IndexError: If the index is out of bounds.
        :return: The number of agents which are members of the group at the specified index.
        :rtype: int
        """
        if 0 < index < len(self.groups):
            return self.groups[index].get_num_members()
        raise IndexError(f"Tried to report the number of members for a group at an out-of-bounds index ({index})")

    def get_group_ids(self) -> list[str]:
        """
        A getter function that returns the IDs of all groups contained within the group set.

        :return: All of the contained groups' IDs.
        :rtype: list[str]
        """
        group_ids: list[str] = []
        for group in self.groups:
            group_ids.append(group.id)
        return group_ids

    def get_member_ids(self) -> list[str]:
        """
        A getter function that returns the unique IDs of all members contained within the group set.

        :return: Every unique member ID.
        :rtype: list[str]
        """
        group_ids: set[str] = set()
        for group in self.groups:
            for member in group.members:
                group_ids.add(member)
        return list(group_ids)

    @override
    def __getstate__(self) -> dict[str, list[Group] | rd.Random]:
        """
        Retrieve the current state of the GroupSet for serialization.

        :return: A representation of the current state of the GroupSet.
        :rtype: dict
        """
        return {"groups": self.groups, "random": self.random}
