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

from gatoh.utils import draw_random_value, random_coinflip, value_rw_delta


# The compression level to use across relevant methods
COMPRESS_LEVEL: int = 4

# A generic to be used in cases where variables may be any type.
T = TypeVar("T")
# A generic to be used for the case where a fully generic dictionary may be passed (e.g dict[S, T])
S = TypeVar("S")


class Group:
    """
    A class to define Groups of Agent objects that can serve as a way of clustering hierarchies,
    and providing a pseudo-vectorisation of model operations across agents.
    """

    def __init__(self, *args: T, **kwargs: T) -> None:
        # Attributes declared but without initialisation will be defined by self.generate_group() in a subsequent call if no args are passed
        self.id: str
        self.index: int

        self.hierarchy: str
        self.members: list[str] = []

        self.aggregate_opinion: float

        self.aggregate_susceptibility: float
        self.radicalisation_rate: float

        # If no args have been passed, it is assumed that self.generate_group() will be subsequently called
        if args:
            for arg in args:
                if isinstance(arg, str):
                    self.add_attribute("id", value=arg)
                elif isinstance(arg, float):
                    self.add_attribute("aggregate_opinion", value=arg)
                else:
                    pass
        if kwargs:
            for key, value in kwargs.items():
                # No checking for duplicate keys; assume that explicitly added kwargs should override any args
                self.add_attribute(key, value=value)

    def generate_group(self) -> Group:
        """
        Randomly generate a Group object based on the input parameters.
        """
        # TODO: Implement this function
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
                f"WARNING: Attempting to get an agent attribute ({name}) which doesn't exist.",
                category=UserWarning,
            )
        return attribute

    def get_num_members(self) -> int:
        """
        A getter method that reports the number of members that are contained in this Group.

        :return: The number of agents that are members of this Group.
        :rtype: int
        """
        return len(self.members)

    def change_aggregate_opinion(self, opinion_delta: float) -> None:
        """
        A setter method that changes the Group's aggregate opinion by a given delta value.

        This will cause changes to the opinions of member agents to create the desired aggregate
        opinion.

        :param opinion_delta: The delta value by which to shift the Group's aggregate opinion.
        :type opinion_delta: float
        :raises TypeError: If opinion_delta is not a float.
        """
        if not isinstance(opinion_delta, float):
            raise TypeError("opinion_delta must be a float")

        num_agents: int = self.get_num_members()

        per_agent_delta: float = opinion_delta / num_agents

        # TODO: Finish this function

        return None

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

    def add_member(self, member: str) -> None:
        """
        Add a new member to this group (using the Agent's ID as the identifier).

        :param member: The ID of the Agent being added as a group member.
        :type member: str
        :raises TypeError: If the input value is not a string.
        """
        if not isinstance(member, str):
            raise TypeError("member must be a string value")
        self.members.append(member)
        return None

    def add_members(self, members: Iterable[str]) -> None:
        """
        Add multiple new members to this group (using the Agent IDs as the identifiers).

        :param members: An iterable containing the Agent IDs of the members to be added.
        :type members: Iterable[str]
        :raises TypeError: If any of the input members are not string values.
        """
        for member in members:
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

    def agents_at_indices(self, indices: list[int]) -> list[Group]:
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
