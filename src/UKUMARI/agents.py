from __future__ import annotations

import warnings
from collections.abc import Callable, Generator, Iterable
from random import Random
from typing import Any, override

import numpy as np
import polars as pl

from .model import ABModel


class Agent:
    """
    A class to define the Agent objects that will interact with each other in an agent-based model.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Supported positional arguments:
            - <string> to set the Agent's id
            - <dict> of {hierarchy_name : weight} for the personal value that this Agent assigns to each social hierarchy
            - <float> in the range [-1, 1] to set the Agent's initial opinion on the topic of interest
            - (x, y) for the initial position of the Agent

        :param args: positional arguments that can be passed to each Agent
        :param kwargs: keyword arguments that can be passed to each Agent
        """

        self.id: str
        self.opinion: float = 0.0
        self.previous_opinion: float = (
            0.0  # Used to handle updating during model iterations
        )
        self.social_weightings: dict[str, float] = {}
        self.personality: str = "neutral"
        self.position: tuple[int, int]
        self.radicalised: bool = False

        if args:
            for arg in args:
                match arg:
                    case dict():
                        self.add_attribute("social_weightings", arg)
                    case float():
                        self.add_attribute("opinion", arg)
                    case tuple():
                        self.add_attribute("position", arg)
                    case str():
                        self.id = arg

        if kwargs:
            for key, value in kwargs.items():
                # No checking for duplicate keys; assume that explicitly added kwargs should override any args.
                self.add_attribute(key, value)

    def add_attribute(
        self,
        name: str,
        value: Any | None = None,
        mean: float | None = None,
        sdev: float | None = None,
        distribution: str | None = None,
        overwrite: bool = False,
    ) -> None:
        # TODO: Add support for additional random distributions
        """
        Dynamically add an attribute to this Agent object. If "value" is passed, an explicit initial value is given;
        if "mean" and "sdev" are passed, a value is generated from a random distribution.
        Supported random distributions are:
            - "normal"
            - "uniform" -- In the case of uniform, `mean` will be treated as the median value, and `sdev` as the distance between the median and the boundaries

        :param name: The name of the attribute to be added.
        :param value: Optional initial value of the attribute.
        :param mean: Optional mean of the random distribution from which to generate the value
        :param sdev: Optional standard deviation of the random distribution from which to generate the value
        :param distribution: Optional string to select which random distribution will be used to generate the value
        """
        if not value and (not mean and not sdev):
            raise ValueError(
                "Either explicit `value` or distribution `mean` and `sdev` are expected when adding Agent attributes."
            )

        if not overwrite and name in self.__dict__.keys():
            # Raise a warning but do not change any attributes or crash the model if overwriting an existing attribute without meaning to.
            warnings.warn(
                "WARNING: Attempting to overwrite an existing Agent attribute without meaning to.",
                category=UserWarning,
            )
        else:
            if value:
                # Assume a given explicit value always overrides (mean, sdev)
                self.__dict__[name] = value
            elif mean and sdev:
                match distribution:
                    case "normal":
                        self.__dict__[name] = np.random.normal(loc=mean, scale=sdev)
                    case "uniform":
                        uniform_range = (mean - sdev, mean + sdev)
                        self.__dict__[name] = np.random.uniform(
                            low=uniform_range[0], high=uniform_range[1]
                        )
                    case None:
                        # Fall back on the normal distribution
                        self.__dict__[name] = np.random.normal(loc=mean, scale=sdev)

    def get_attribute(self, name: str) -> Any:
        try:
            return self.__dict__[name]
        except KeyError:
            warnings.warn(
                "WARNING: Attempting to get an Agent attribute which doesn't exist.",
                category=UserWarning,
            )
            return None

    def step(self):
        """
        Step the individual agent object
        """
        pass

    def update_state(self):
        """
        Updates the internal state of the agent after the model has stepped.
        """
        pass

    def radicalisation(self, neighbours: Iterable[Agent]) -> bool:
        """
        Uses the agent's own opinion as well as the neighbours' opinions to determine if
        the agent has become radicalised in their actions.

        :param neighbours: A list of all agents that "neighbour" this agent in any model layer.
        """
        # If the Agent is already radicalised, always return True
        if self.radicalised:
            return True

        match self.__getattribute__("personality"):
            case "rational":
                pass
            case "erratic":
                pass
            case "impulsive":
                pass
            case None:
                pass
        return False  # TODO: Finish this method (returning False to suppress typing warnings)

    def evolve_relationships(self):
        """
        Experimental function that aims to model the constantly evolving relationships between Agents over time
        """
        raise NotImplementedError(
            "Agent relationship evolution has not been implemented as a feature yet."
        )

    def life_events(self):
        """
        Experimental function that aims to model the ways in which Agent behaviours change according to major random life events over time
        """
        raise NotImplementedError(
            "Agent life events have not been implemented as a feature yet."
        )

    def __in__(self, iterable: Iterable[Agent]) -> bool:
        """
        Determine if the Agent is contained within an iterable of Agents

        :param iterable: The iterable of Agent objects in which membership is being determined
        """
        for agent in iterable:
            if self == agent:
                return True
        return False

    def __str__(self) -> str:
        """
        An override to what calling `print()` on this object will output
        """
        return f"Agent {self.id} which {'is' if self.radicalised else 'is not'} radicalised with an opinion value of {self.opinion}"


class AgentSet:
    """
    An ordered collection of Agent objects that maintains consistency for the Model
    """

    def __init__(self, model: ABModel) -> None:
        """
        :param model: The parent ABModel object that this AgentSet is being attached to
        """
        self.parent_model = model
        self.agents: pl.Series = pl.Series()
        self.random: Random = Random()

    def __iter__(self) -> Generator[Any]:
        """
        Override of what calling __iter__ on this object will return
        """
        return self.agents.__iter__()

    def __len__(self) -> int:
        """
        :return: the number of agents present in the AgentSet
        """
        return len(self.agents)

    def __contains__(self, agent: Agent) -> bool:
        """
        :param agent: the specific Agent object to check for
        :return: a boolean indicating if the specified Agent object is in the AgentSet
        """
        return self.agents.__contains__(agent)

    def select(
        self,
        filter_func: Callable[[Agent], bool],
        inplace: bool = False,
        k: int = int(np.inf),
    ) -> AgentSet:
        """
        Select a subset of Agent objects from the AgentSet.

        :param filter_func: a function used to filter the Agent objects
        :param inplace: if True, modify the existing AgentSet, otherwise return a new AgentSet
        :param k: the maximum number of Agent objects to include in the subset
        :return: an AgentSet containing a filtered subset of Agents
        """
        set_mask = []
        for agent in self.agents:
            set_mask.append(filter_func(agent))

        reduced_set: pl.Series = self.agents.filter(set_mask)
        if k < len(reduced_set):
            reduced_set = reduced_set.sample(n=k)

        if inplace:
            self.agents = reduced_set

        return self

    def __getitem__(self, item: int | slice) -> pl.Series | Any:
        """
        Retrieve an Agent or slice of Agents from the AgentSet.
        :param item: the index or slice for selecting the agents
        :return: the selected agent or slice of agents based on the specified item
        """
        return self.agents.__getitem__(item)

    def add(self, agent: Agent) -> int:
        """
        Add an Agent to the AgentSet.
        :param agent: the Agent object to be added
        """
        for idx, agnt in enumerate(self.agents):
            if agnt is None:
                self.agents[idx] = agent
                self.agents[idx].id = idx
                return 1
        self.agents.append(pl.Series(agent))
        self.agents[-1].id = len(self.agents) - 1
        return 1

    def discard(self, agent: Agent) -> int:
        for idx, agnt in enumerate(self.agents):
            if agent == agnt:
                self.agents[idx] = None
                return 1
        return 0

    def remove(self, agent: Agent) -> int:
        for idx, agnt in enumerate(self.agents):
            if agent == agnt:
                self.agents[idx] = None
                return 1
        raise KeyError("Tried to remove an Agent that doesn't exist in the AgentSet")

    @override
    def __getstate__(self) -> dict:
        """
        Retrive the current state of the AgentSet for serialization.
        :return: a dictionary representing the current state of the AgentSet
        """
        return {"agents": list(self.agents), "random": self.random}
