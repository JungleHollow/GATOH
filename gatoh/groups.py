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
from typing import NotRequired, TypeVar, TypedDict, override

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gatoh.agents import Agent

from gatoh.utils import draw_random_value, random_coinflip, value_rw_delta

# Definition of all valid, existing Group cohesion categories
COHESIONS: list[str] = ["intimate", "close", "neutral", "distant", "passing"]

# Definition of global constants to be used instead of "magic numbers" throughout the code

# The absolute maximum value that aggregate group opinions can take
OPINION_MAX: float = 1.0
# The compression level to use across relevant methods
COMPRESS_LEVEL: int = 4

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
            _ = personality_counts.setdefault(agent.personality, 0)
            personality_counts[agent.personality] += 1

        # Calculate the final attributes and set them
        self.aggregate_opinion = opinion_sum / len(members)
        self.radicalisation_rate = radicalisation_count / len(members)
        self.member_benefit_rate = benefit_count / len(members)
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
        :raises ValueError: If the input list is not the same size as the group.
        """
        if len(member_personalities) != self.get_num_members():
            raise ValueError("The number of personality types input does not match the number of group members")
        personality_counts: dict[str, int] = {}
        for personality in member_personalities:
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

        num_agents: int = self.get_num_members()

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

    def is_radicalised(self, threshold: float = 0.2) -> bool:
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

    @override
    def __getstate__(self) -> dict[str, list[Group] | rd.Random]:
        """
        Retrieve the current state of the GroupSet for serialization.

        :return: A representation of the current state of the GroupSet.
        :rtype: dict
        """
        return {"groups": self.groups, "random": self.random}
