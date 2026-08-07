================
GATOH Quickstart
================

The Generalised Agent Transformation of Opinions in Hierarchies, or GATOH, framework is an agent-based modeling library
for Python. It is designed to provide a highly flexible and generalisable framework that can define and simulate a wide
variety of agent-based models centered on opinion dynamics.

----------------
Installing GATOH
----------------

GATOH has been primarily developed and tested on a x86_64 Linux system, but support for 64 bit Windows has also been
confirmed. The different methods for installing GATOH are outlined below:

.. tabs::

   .. group-tab:: From source

      #. Ensure that Python >=3.14 is installed on your system.
      #. Clone the library's code from the git repository: :code:`git clone https://github.com/JungleHollow/GATOH`.
      #. If needed, create a virtual environment to install the library to.
      #. From the target python environment, simply install GATOH using pip as normal, but specify the path to the GATOH directory rather than the package name: :code:`python -m pip install <path to GATOH directory>`.
      #. At this point, all required dependencies will be installed alongside GATOH, and you can begin using the library.

   .. group-tab:: PyPi

      #. Ensure that Python >= 3.14 is installed on your system.
      #. ...

      **Work in progress**

-----------
Using GATOH
-----------

Once you have installed GATOH, you can use it by importing gatoh in your python scripts. Almost all functions and classes in GATOH are
currently organised into their own submodules, meaning that these will need to be imported and used accordingly. Below is a simple
example of what very basic GATOH usage could look like:

.. code-block:: python

   from gatoh.model import ABModel
   from gatoh.utils import plot_graph

   HIERARCHIES = ["Religious", "Neighbours", "Family", "Friends"]
   HIERARCHY_RW = {
       "Religious": (0.0, 0.3),
       "Neighbours": (0.0, 0.2),
       "Family": (0.0, 0.01),
       "Friends": (0.0, 0.1),
   }

   model = ABModel(
       HIERARCHIES,
       HIERARCHY_RW,
       iterations=50,
       model_id="TEST_MODEL",
   )
   model.generate_agents(
       "TEST",
       {
           "neutral": 0.4,
           "social": 0.4,
           "impulsive": 0.2,
       },
       number=20,
   )
   model.generate_graphs(
       HIERARCHIES,
       model.agents,
       method="blockmodel",
       agent_subsetting=True,
   )

   model.iterate()

   x_vals = {"iterations": [i for i in range(50)]}
   y_vals = {"aggregate_opinions": model.logger.variables.aggregate_opinions}

   plot_graph(
       x_vals,
       y_vals,
       ...
   )

You can refer to the :doc:`Basic tutorial <./basics>` for more information on how to use GATOH,
or the :doc:`Absolute beginner's guide <./abs_beginners>` for an in-depth explanation of agent-based
modeling and the GATOH framework.
