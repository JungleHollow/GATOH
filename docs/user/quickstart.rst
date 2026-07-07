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

One of the core objects of the GATOH framework is the "Agent". In the GATOH framework, an Agent represents an "average" person that lives
in some community, and interacts with their neighbours in different social contexts.
