****************************************
GATOH: The absolute basics for beginners
****************************************

.. currentmodule:: gatoh

.. testsetup::

    >>> import gatoh
    >>> import sys

.. _user/abs-beginners:

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
and they were still basically just a step above calculators. To give an example, the Cray-1 -- a 5.5 ton supercomputer released in 1976 -- had
a processor capable of running 80 million operations per second, and had 8.4 megabytes of memory available.

Now, in 2026, the newest consumer :abbr:`CPUs (Computer Processing Units)` are capable of performing around 5 billion operations per second, and they
have more memory available in just their L3 cache than 20 Cray-1 supercomputers would have had collectively in the 1970s(!). Considering this,
it is easy to see how :abbr:`ABMs (Agent-Based Models)` would have started evolving past these simple 2-dimensional, binary cells in the following
decades.

Now it is possible to model extremely complex systems such as the spread of global pandemics, or the nature of peak-hour traffic in a large city.
Although :abbr:`ABM (Agent-Based Modelling)` can be applied to many use-cases, there is still some care that must be taken when creating such models.

Each area of interest has very specific characterisations, rules, and interactions, which must all be appropriately quantified and defined for the model
to be representative of real life. Using the examples of global pandemics and car traffic, we can see that models for these scenarios would have inherently
different approaches. A model that simulates a global pandemic might treat interactions between agents as the driving force that spreads the 'emergent behaviour'
(as interactions between agents would enable viral transmissions), whereas the 'interaction' between agents in a traffic model would in itself be the emergent
behaviour (as the 'interaction' of a large number of agents would imply that traffic has built up).

For the purposes of this guide, let's look at another famous :abbr:`ABM (Agent-Based Model)` from the same era as Conway's game of life -- Schelling's
model of segregation :cite:p:`Schelling1971`. In the following sections, we will break down :abbr:`ABMs (Agent-Based Models)` into their core components,
and see how each fits into the overall modeling process; using Schelling's model as an example.

--------------------
Defining the problem
--------------------

The first, and arguably most important, step in the agent-based modeling process is the concrete definition of the problem and its computational representation.
If the problem is not defined correctly from the start, then this just leads to a lot of headache and time lost further along the line, as any changes that have
to be made later -- no matter how minor or isolated they may appear -- will likely be connected to a number of other components that will have to be redefined
and tweaked as well. This is particularly important when we are talking about highly complicated models that take multiple hours to run *one* iteration; it's not
a great experience when you realise that a model parameter is wrong and has to be changed when your model is halfway done with its iterations and has already been
running for the past week...

All of this to say that it is very important and worthwhile to spend some extra time at the start to think through, and properly define the representation of the
problem for your model.

In the case of Schelling's model of segregation, the foundation of the model is based on the assumption that no outside influences or explicit prejudices exist (
for the purposes of the model); with the only factor affecting possible segregation being the presence of "mild in-group preferences" in people. Following this
assumption, the model is defined as follows:

  - A community is represented by a :math:`N \times N` grid of cells.
  - Each cell in the community represents a space that can be inhabitted by an agent, or be empty.
  - Every agent in the community is assigned to one of two "groups".
  - At every iteration, each agent:

    #. Observes its immediate neighbours (ignoring empty spaces).
    #. If the proportion of neighbours belonging to the same "group" is equal or greater to some threshold, the agent is content and remains stationary.
    #. Otherwise, the agent is dissatisfied, and will move from its current space to another empty space in the community.

In this way, a seemingly complex problem (such as racial segregation in the case of Schelling) can be defined in a more deterministic manner that can be more easily
managed computationally. In some cases, this may mean a simplification (i.e. reducing the level of detail of, or completely removing a concept from the model), whilst
in other cases this may mean having to think outside the box a bit to be able to quantify some property or behaviour that by nature isn't already quantified (e.g.
treating the strength of a person's beliefs as a value on a scale from 0-10).

.. raw:: html

    <figure>
        <img class="img-centered" src="../../_static/Schellings.gif" alt="Schelling segregation">
        <figcaption>
            <p>An animation of Schelling's model of segregation in action</p>
            <p class="caption-sub">By <a href="//commons.wikimedia.org/w/index.php?title=User:Blaqdolphin&amp;action=edit&amp;redlink=1" class="new" title="User:Blaqdolphin (page does not exist)">Blaqdolphin</a> - <span class="int-own-work" lang="en">Own work</span>, <a href="https://creativecommons.org/licenses/by-sa/4.0" title="Creative Commons Attribution-Share Alike 4.0">CC BY-SA 4.0</a>, <a href="https://commons.wikimedia.org/w/index.php?curid=91228415">Link</a></p>
        </figcaption>
    </figure>

Above, I have included a short GIF depicting the runtime of Schelling's model of segregation so that you can understand better how everything will fit together.
As you can see, the agents start in a relatively random distribution within the community, but as more iterations happen, the community slowly becomes more
and more segregated into distinct "tribes".

------
Agents
------

**Definition:** An agent is an entity that is capable of acting without external input, and can interact with other agents and the environment
around it.

One of the core objects of the :abbr:`GATOH (Generalised Agent Transformation of Opinions in Hierarchies)` framework is the "Agent". In the :abbr:`GATOH (Generalised Agent Transformation of Opinions in Hierarchies)`
framework, an Agent represents an "average" person that lives in some community, and interacts with their neighbours in different social contexts.

-----------------------------
Observing Emergent Behaviours
-----------------------------


Beyond Traditional ABMs
=======================

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
