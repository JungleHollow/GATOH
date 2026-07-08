*************
Release Notes
*************

.. release-notes::

0.2.0-alpha
===========

This release involves the optimisation of core package functionality to reduce
model runtimes and simplify the modelling process.

New Features
------------

- A new way of tracking changes and updating model's base graphs, composed of the following:

  - :class:`~gatoh.utils.utils.NodeChanges`
  - :class:`~gatoh.utils.utils.EdgeChanges`
  - :meth:`~gatoh.graphs.graphs.Graph.register_edge_change`
  - :meth:`~gatoh.graphs.graphs.Graph.get_edge_changes`
  - :attr:`~gatoh.graphs.graphs.Graph.pending_edge_changes`
  - :meth:`~gatoh.model.model.ABModel.init_base_graph`
  - :meth:`~gatoh.model.model.ABModel.update_base_graph`

Fixes
-----

- Added edge case handling for division by zero errors in :class:`~gatoh.model.model.ABModel`
- Corrected the object that was being passed to :func:`rustworkx.visualization.mpl_draw` in :meth:`~gatoh.visualisation.visualisation.ABVisualiser.visualise_hierarchy`
- Fixed value setting error caused by mismatching :obj:`NDArray` data types in :mod:`ResponseParser.py`
  for the planned case study experiment
- Fixed the accessing of Agent :attr:`~gatoh.agents.agents.Agent.personal_benefit` attributes in :meth:`ResponseParser.custom_iterate`
  for the planned case study experiment

0.1.0-alpha
===========

This is the first official release of GATOH.

The main purpose of this release is to start preparing the code and its
supplementary materials for publication as a proper package.

New Features
------------

- Start of the creation of documentation material. This includes:

  - index.rst
  - license.rst
  - Full API reference
  - Developer and user manuals
  - Getting Started section
  - CITATION.cff

- Mostly working functionality across all main package modules
- Basic model visualisation functionality
- Support for four commonly used random graph generation algorithms:

  - small-world
  - scale-free
  - random
  - stochastic blockmodel

- Support for random generation of :obj:`~gatoh.agents.agents.Agent` objects
- A working data persistence system that allows for loading and saving any models
  created with the framework
- Design and implementation of five framework validation experiments
- Initial prototype for a future case study experiment
- Implementation of unit tests for all core module functionalities
- Enforcing of consistent type checking across modules
- Full docstring coverage across the existing code
- Changed the way that default names were generated for model config files to
  ensure functionality across different operating systems
- Initial, basic conversion of some Python modules to Rust code. This will likely
  be expanded on in the future, with the package adopting a hybrid approach, and
  resource-intensive modules being ported to and optimised in Rust.
- Random-walk mechanisms used to simulate the dynamic nature of agent relationships
  in social networks, as well as the variable attitude of agents towards the different
  social networks.
- Publication of the package code to Zenodo

Fixes
-----

- Fixed the way that :meth:`~gatoh.agents.agents.Agent.radicalisation` was determining agent
  radicalisation for "rational" personality types. Now, the function should properly use the
  "aggregate benefit" of neighbours rather than the agent's personal benefit.
- Multiple restructurings of the ./gatoh file layout; clarifying the separation of package
  features each time.
- Creation of dedicated setter functions for :obj:`~gatoh.agents.agents.Agent` objects to fix the inability
  of setting :obj:`~gatoh.agents.agents.Agent` attributes from :obj:`~gatoh.graphs.graphs.Graph` objects which persist:

  - :meth:`~gatoh.agents.agents.Agent.store_previous_opinion`
  - :meth:`~gatoh.agents.agents.Agent.change_opinion`
  - :meth:`~gatoh.agents.agents.Agent.change_radicalisation`

- Creation of dedicated setter functions for :obj:`~gatoh.graphs.graphs.GraphNode` and :obj:`~gatoh.graphs.graphs.GraphEdge` objects
  to fix the inability of setting persistent attributes from within :obj:`~gatoh.graphs.graphs.Graph` objects:

  - :meth:`~gatoh.graphs.graphs.GraphNode.set_index`
  - :meth:`~gatoh.graphs.graphs.GraphEdge.set_index`
  - :meth:`~gatoh.graphs.graphs.GraphEdge.set_weighting`
  - :meth:`~gatoh.graphs.graphs.GraphEdge.update_from_node`
  - :meth:`~gatoh.graphs.graphs.GraphEdge.update_to_node`

- Use of :func:`~copy.deepcopy` across multiple functions to fix data inheritance bugs
