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

Agents can be defined in a number of different ways depending on the exact nature of the problem, but there is rarely an 'optimal' definition; each alternative has its own pros and cons.
The core definition of an agent will obviously vary depending on the problem definition, as different types of problems will be interested in tracking and simulating different types of characteristics
or traits in their agents. For example, if the model is being constructed to simulate car traffic, each agent might represent an individual car on the streets.

But even if the problem definition remains fixed, there are multiple ways of defining agents. If a model is instead simulating the spread of avian flu in birds, each agent could represent a single bird,
or a flock of birds that are migrating in an area. In this way, the exact definition of the agents in the model will have an impact on what exactly happens when different "agents" interact with each
other; as the interaction can be representative of individual or group effects.

Another thing to keep in mind when defining an agent is how, or if, the agent will interact with the space around them (if there even is a "space" defined for your particular model). Depending on the mechanism
that you have chosen to use for agent-space interaction, it may be necessary for agents to hold additional attributes so that they can differentiate the information that they provide for agent-agent interactions
versus agent-space interactions, as these will likely not involve the same behaviours.

In the case of Schelling's model of segregation, an agent represents a person that lives within some arbitrary community, and they may belong to one of two different groups. The agents in Schelling's model do not
"interact" with each other in the traditional sense, but rather they observe the traits of their neighbours. Agents do, however, interact with the space around them; moving from their current cells to empty ones
if they are not satisfied with the group makeup of their neighbours. Although, in this particular case no additional information is needed to differentiate between agent-agent and agent-space interactions, as each
cell contains all the needed information (either the agent's group if the cell is occupied, or nothing if the cell is empty).

-----------
Model Space
-----------

The model "space" was briefly touched on in the previous section, but let's look at it in some more detail here.

Just how we live in and interact with the space around us (our homes, neighbourhoods, cities, etc.) an agent in an :abbr:`ABM (Agent-Based Model)` may interact with some form of "space" around them if the model
requires it. For some models, space is a core factor for the emergent behaviour -- it would be impossible to simulate the presence of car traffic if agents had no space to create traffic in, for instance. In other
models, space is simply a way of capturing additional information; the inclusion of "space" and "distance" may allow for a stronger, more accurate model, but the underlying behaviours that are being simulated do not
directly depend on the space itself. And of course, it may be possible for some models to not require a "space" at all.

Similarly to how there are many ways of defining an agent, each with their own pros and cons, there are many ways of defining model space. The model space can be an explicit "map" with coordinates that the agents inhabit,
or it could follow a graph-based approach. It is also possible for model space to be more of an abstract concept; with agents posessing attributes that note the different cities that they live in, for example.

For Schelling's model, a :math:`N \times N` cell grid is used, where :math:`N \times N` must be greater than the total number of agents to allow for agents to move into empty cells. Schelling's model is somewhat different to
the "traditional" :abbr:`ABM (Agent-Based Model)` when it comes to the model space, with the cells doubling as both the spaces and the agents themselves.

-----------
Model Rules
-----------

The rules of a model provide the necessary conditions for emergent behaviours to occur. To simplify things a bit, let's separate the rules into 3 different categories and look at each one individually:

  #. Restrictive Rules
  #. Interaction Rules
  #. Behavioural Rules

~~~~~~~~~~~~~~~~~~~~
1. Restrictive Rules
~~~~~~~~~~~~~~~~~~~~

Restrictive rules can be thought of -- as the name suggests -- as any model rules that restrict the model's components in some way.

The use of restrictive rules can be for a number of reasons, but the 2 main reasons that you will typically be creating these rules for are: to simplify the model, and to ensure that the model is accurate.

When there is a need to simplify a model so that it can run more efficiently, it may be possible to do this by reducing the number of attributes, but this will usually not have much of an effect as attributes do not cause much of an
impact to memory usage or runtime. Instead, restrictive rules can be created to reduce or completely remove specific behaviours from the model. This is much more likely to lead to performance gains, as behaviours require interactions
between agents to occur, and if the behaviour is not of particular interest, then the number of interactions can be reduced.

The other case when you would likely want to introduce restrictive rules is when the problem definition itself requires these restrictions to be more accurate. A good example of this happens in Schelling's model, where agents can only
move into empty spaces. Given that you want the model to be representative of real life, then it would not make sense for the people in one household to be able to move into a household that is already occupied.

~~~~~~~~~~~~~~~~~~~~
2. Interaction Rules
~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~
3. Behavioural Rules
~~~~~~~~~~~~~~~~~~~~

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
