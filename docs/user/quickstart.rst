================
GATOH Quickstart
================

.. currentmodule:: gatoh

.. testsetup::

    >>> import gatoh
    >>> import sys

Prerequisites
=============

A basic knowledge of Python is assumed. For further reference, see the `Python tutorial <https://docs.python.org/tutorial/>`__.

To work the examples, you'll need the GATOH framework and its included dependencies installed.

**Learner Profile**

This is a quick overview of agent-based modelling, and its implementation in GATOH. It demonstrates how agents are defined, and how they
interact with each other and their environment to produce emergent behaviours. If you are looking for a basic introduction to agent-based
modelling principles in the GATOH framework, then this article may be of use to you.

**Learning Objectives**

After reading, you should be able to:

- Understand what an "agent" is, and how they are defined;
- Understand the basic process of agent-based modelling from start to finish;
- Understand the different use-cases for agent-based modelling.

.. _quickstart.the-basics:

The Basics
==========

To start with, the fundamental modelling method that GATOH is based around is that of Agent-Based Modelling (ABM). The implementation of
ABMs can be approached in a variety of ways depending on the nature of the scenario that they are simulating. The scenario must be
properly defined -- and there must be a clear idea of the type of behaviours that emerge from the scenario, as well as who or what
produces those behaviours. It is also important to consider the environment in the scenario, and whether or not it has a significant
impact on the emergent behaviours.

GATOH is primarily designed with sociological or anthropological simulations in mind. The simplest general scenario that is intended
to be modelled with GATOH is that of social contagion in some community or social network; with the emergent behaviour in this scenario
being the spread of some idea or belief between the people who live in the community. But, social interactions between community members
are not simple, and are heavily influenced by a number of factors -- primarily the different social hierarchies that exist in the
community, that agents may or may not belong to.

In the next sections, we will explore the different components that make up a GATOH model, and the ways that they interact with
each other.

------
Agents
------

**Definition:** An agent is an entity that is capable of acting without external input, and can interact with other agents and the environment
around it.

One of the core objects of the GATOH framework is the "Agent". In the GATOH framework, an Agent represents an "average" person that lives
in some community, and interacts with their neighbours in different social contexts.

---------------
Social Networks
---------------

.. note::

  For the context of GATOH, "social network" refers to a network of people that interact with each other socially, *not* a social media
  platform.

------
Graphs
------

-----------------
Multilayer Graphs
-----------------
