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
