*************
Release Notes
*************

.. release-notes::

2026.08.0
==============

Continued addition of new features -- mostly QoL convenience features, but some new core features.

Also, ongoing bugfixes and optimisations where possible.

New Features
------------

- Added the ability to explicitly define if generated graphs should be complete from within :meth:`~gatoh.model.ABModel.generate_graphs`
- Started using :mod:`coverage` to track the coverage of tests across the library modules
- Finished the test cases for :class:`~gatoh.agents.Agent` creation and attributes
- Added input parameter data type checks to various methods in :class:`~gatoh.agents.Agent`
- Finished the test cases for :class:'~gatoh.agents.Agent' populated and empty objects
- Ability to have partial agent iterations in the :class:`~gatoh.model.ABModel`
- Additional getter methods defined for :class:`~gatoh.agents.AgentSet`

Fixes
-----

- Fixed the test cases in :mod:`test_model_multiprocessed.py` and :mod:`test_model_simulation.py` (All 151 existing test cases up to 01/08/2026 are working correctly)
- Fixed a specific :meth:`~gatoh.graphs.Graph.add_edges` case in which edges were not being added to the graph
- Fixed the build parameters for the docs (will now always add new methods when building)

2026.07.0
==============

GATOH changes from a semantic to a calendar versioning scheme with this release for simplicity of version control.

This version involves a large number of logistic or 'housekeeping' changes in preparation for an official release
as a PyPI package.

New Features
------------

- Changed from semantic to calendar versioning
- Changed the layout of the package modules to simplify import statements
- Extended full docstrings to case study experiment code
- Implemented the :meth:`~gatoh.model.ABModel.calculate_navigability` function
- Implemented an initial prototype for :meth:`~gatoh.agents.Agent.life_events`
- Implemented a function to calculate graph density metrics with :meth:`~gatoh.model.ABModel.calculate_density`
- Implemented functionality to automatically remove any nodes in graphs without neighbours after relevant edges are removed
- Added functions that allow for the spontaneous formation and disintegration of relationships in graphs
- Implemented a mechanism that allows for thresholded agent deradicalisation
- Now accounting for the idea that like-minded agents will push each other towards more extreme opinions even if the average opinion
  is currently stable (:meth:`~gatoh.model.ABModel.iteration_opinion_calculation`)
- Added new checks to :meth:`~gatoh.graphs.Graph.agent_opinion_change` that account for agent deradicalisation
- Added multiple QoL setter functions to :class:`~gatoh.model.ABModel`
- Rudimentary prototype for a debugging system in :class:`~gatoh.logging.GATOHLogger`
- Expanded test coverage to include all the new QoL setter functions in :class:`~gatoh.model.ABModel`
- QoL method :meth:`~gatoh.model.ABModel.add_agents_to_hierarchy`
- Created new tests to cover the various calculation methods in :class:`~gatoh.model.ABModel`
- QoL method :meth:`~gatoh.model.ABModel.add_relationships_to_hierarchy`
- Defined multiple global constants throughout modules to clarify "magic numbers"
- Prototype for ability to add "link functions" to model parameters
- Prototype for the ability to track specific model parameters throughout iterations within the :class:`~gatoh.logging.GATOHLogger`

Fixes
-----

- Fixed all the existing imports across scripts to reflect the module restructuring
- Fixed all the code references in the existing documentation
- Fixed all the save/load paths in the experiment scripts to reflect the new layout
- Fixed the layout of agents in the radicalisation heatmap (from shapes of (10, x) to (x, 10))
- Numerous bugs fixed throughout the existing test cases (errors in the test scripts themselves, not in the core code)

Breaking Changes
----------------

- Any pickled GATOH objects from previous versions cannot be used in future versions starting with this one.
  This is due to the simplification from gatoh.X.X.x to gatoh.X.x imports.

0.2.0-alpha
===========

This release involves the optimisation of core package functionality to reduce
model runtimes and simplify the modelling process.

New Features
------------

- A new way of tracking changes and updating models' base graphs, composed of the following:

  - :class:`~gatoh.utils.utils.NodeChanges`
  - :class:`~gatoh.utils.utils.EdgeChanges`
  - :meth:`~gatoh.graphs.graphs.Graph.register_edge_change`
  - :meth:`~gatoh.graphs.graphs.Graph.get_edge_changes`
  - :attr:`~gatoh.graphs.graphs.Graph.pending_edge_changes`
  - :meth:`~gatoh.model.model.ABModel.init_base_graph`
  - :meth:`~gatoh.model.model.ABModel.update_base_graph`

- Multiprocessing added to :meth:`~gatoh.model.model.ABModel.iterate`
- Multiprocessing added to :meth:`~gatoh.model.model.ABModel.update`
- Multiprocessing added to the custom iteration function in the :mod:`CaseStudy` experiment
- Multiprocessing added to :meth:`~gatoh.model.model.ABModel.calculate_layers_polarisation`
- Multithreading added to :meth:`~gatoh.agents.agents.AgentSet.save_agentset`
- Multithreading added to :meth:`~gatoh.graphs.graphs.GraphSet.save_graphset`
- Multiprocessing added to :meth:`~gatoh.model.model.ABModel.calculate_interdependence`
- Creation of a quickstart guide for the documentation
- Multiprocessing and multithreading added to :meth:`~gatoh.agents.agents.AgentSet.load_agentset`
- Multiprocessing and multithreading added to :meth:`~gatoh.graphs.graphs.GraphSet.load_graphset`
- Tests added for AgentSet objects
- Implemented a custom visualisation system and started using this (previously the one provided by rustworkx was used)

Fixes
-----

- Added edge case handling for division by zero errors in :class:`~gatoh.model.model.ABModel`
- Corrected the object that was being passed to :func:`rustworkx.visualization.mpl_draw` in :meth:`~gatoh.visualisation.visualisation.ABVisualiser.visualise_hierarchy`
- Fixed value setting error caused by mismatching :obj:`NDArray` data types in :mod:`ResponseParser.py`
  for the planned case study experiment
- Fixed the accessing of Agent :attr:`~gatoh.agents.agents.Agent.personal_benefit` attributes in :meth:`ResponseParser.custom_iterate`
  for the planned case study experiment
- Fixed the radicalisation checks in :meth:`~gatoh.model.model.ABModel.iteration_opinion_changes` to stop repeated deradicalisation of agents
- **MAJOR:** Fixed numerous variable declarations and added manual garbage collection throughout various functions to prevent memory leaks during model runtimes
- Fixed modulo operations in new :class:`~gatoh.visualisation.visualisation.ABVisualiser` methods
- Removed most uses of "Any" in the code typing
- Stricter type annotations throughout the code

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
