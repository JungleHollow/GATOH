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

To work the examples, you'll need the :abbr:`GATOH (Generalised Agent Transformation of Opinions in Hierarchies)` framework and its included dependencies installed.

**Learner Profile**

This is a quick overview of agent-based modelling, and its implementation in :abbr:`GATOH (Generalised Agent Transformation of Opinions in Hierarchies)`. It demonstrates how agents are defined, and how they
interact with each other and their environment to produce emergent behaviours. If you are looking for a basic introduction to :abbr:`ABM (Agent-Based Modelling)` principles in the
:abbr:`GATOH (Generalised Agent Transformation of Opinions in Hierarchies)` framework, then this article may be of use to you.

**Learning Objectives**

After reading, you should be able to:

- Understand what an "agent" is, and how they are defined;
- Understand the basic process of :abbr:`ABM (Agent-Based Modelling)` from start to finish;
- Understand the different use-cases for :abbr:`ABM (Agent-Based Modelling)`.

.. _quickstart.the-basics:

The Basics
==========

To start with, the fundamental modelling method that :abbr:`GATOH (Generalised Agent Transformation of Opinions in Hierarchies)` is based around is that of :abbr:`ABM (Agent-Based Modelling)`. The implementation of
:abbr:`ABMs (Agent-Based Models)` can be approached in a variety of ways depending on the nature of the scenario that they are simulating. The scenario must be
properly defined -- and there must be a clear idea of the type of behaviours that emerge from the scenario, as well as who or what
produces those behaviours. It is also important to consider the environment in the scenario, and whether or not it has a significant
impact on the emergent behaviours.

:abbr:`GATOH (Generalised Agent Transformation of Opinions in Hierarchies)` is primarily designed with sociological or anthropological simulations in mind. The simplest general scenario that is intended
to be modelled with :abbr:`GATOH (Generalised Agent Transformation of Opinions in Hierarchies)` is that of social contagion in some community or social network; with the emergent behaviour in this scenario
being the spread of some idea or belief between the people who live in the community. But, social interactions between community members
are not simple, and are heavily influenced by a number of factors -- primarily the different social hierarchies that exist in the
community, that agents may or may not belong to.

In the next sections, we will explore the different components that make up a :abbr:`GATOH (Generalised Agent Transformation of Opinions in Hierarchies)` model, and the ways that they interact with
each other.

---------------------
Agent-Based Modelling
---------------------

:abbr:`ABM (Agent-Based Modelling)` is a computational modelling strategy that has grown in popularity in recent years as computer
hardware has become more powerful.

The first :abbr:`ABMs (Agent-Based Models)` were simple, cell-based systems, where each cell represented a single agent. The most famous
example of these types of :abbr:`ABMs (Agent-Based Models)` is Conway's Game of Life\ :cite:p:`Conway1970` . In this rudimentary
:abbr:`ABM (Agent-Based Model)`, simple rules are defined which determine how and when each cell 'comes to life' or 'dies' based
on the cell's current state, and the state of its immediate neighbours. From these simple rules, relatively complex behaviours can emerge.
In the depiction below, we can see some examples of the behaviours that can emerge from such a simple system:

.. parsed-literal::

                Conway's Game of Life (1970):
    ================================================================
     Still life patterns:   ⫼          Oscillators:
    ================================================================
    ┌─╥─╥─╥─┐   ┌─╥─╥─╥─┐   ⫼
    │ ║ ║ ║ │   │ ║ ║ ║ │   ⫼   ┌─╥─╥─┐   ┌─╥─╥─┐   ┌─╥─╥─┐
    ╞═╬═╬═╬═╡   ╞═╬═╬═╬═╡   ⫼   │ ║ ║ │   │ ║█║ │   │ ║ ║ │
    │ ║█║█║ │   │ ║█║█║ │   ⫼   ╞═╬═╬═╡   ╞═╬═╬═╡   ╞═╬═╬═╡
    ╞═╬═╬═╬═╡ → ╞═╬═╬═╬═╡   ⫼   │█║█║█│ → │ ║█║ │ → │█║█║█│ →   ⦁⦁⦁
    │ ║█║█║ │   │ ║█║█║ │   ⫼   ╞═╬═╬═╡   ╞═╬═╬═╡   ╞═╬═╬═╡
    ╞═╬═╬═╬═╡   ╞═╬═╬═╬═╡   ⫼   │ ║ ║ │   │ ║█║ │   │ ║ ║ │
    │ ║ ║ ║ │   │ ║ ║ ║ │   ⫼   └─╨─╨─┘   └─╨─╨─┘   └─╨─╨─┘
    └─╨─╨─╨─┘   └─╨─╨─╨─┘   ⫼
    ================================================================

Whilst it was revolutionary at its time of creation, the Game of Life was developed in a context where computers were still in their infancy,
and they were still basically fancy calculators. Now, in 2026, the newest :abbr:`CPUs (Computer Processing Units)` are capable of performing
around 5 billion operations per second, and they have more memory available in just their L3 cache than multiple hard drives would have had collectively
in the 1970s(!). Considering this, it is easy to see how :abbr:`ABMs (Agent-Based Models)` would have started evolving past these simple 2-dimensional,
on/off cells.

Now it is possible to model extremely complex systems such as the spread of global pandemics, or the nature of peak-hour traffic in a large city. 

------
Agents
------

**Definition:** An agent is an entity that is capable of acting without external input, and can interact with other agents and the environment
around it.

One of the core objects of the :abbr:`GATOH (Generalised Agent Transformation of Opinions in Hierarchies)` framework is the "Agent". In the :abbr:`GATOH (Generalised Agent Transformation of Opinions in Hierarchies)`
framework, an Agent represents an "average" person that lives in some community, and interacts with their neighbours in different social contexts.

---------------
Social Networks
---------------

.. note::

  For the context of :abbr:`GATOH (Generalised Agent Transformation of Opinions in Hierarchies)`, "social network" refers to a network of people that interact with each other socially, *not* a social media
  platform.

------
Graphs
------

-----------------
Multilayer Graphs
-----------------


.. bibliography::
