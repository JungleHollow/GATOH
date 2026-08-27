from __future__ import annotations

import os
import unittest as ut
from typing import Any, override
from shutil import rmtree

import gatoh.agents as agt
import gatoh.graphs as gr
import gatoh.model as md
import gatoh.groups as grp

HIERARCHY_NAMES: list[str] = ["Test_1", "Test_2"]
HIERARCHY_RW_DISTRIB: list[tuple[float, float]] = [(0, 0.01), (0, 0.2)]


class TestModelCreation(ut.TestCase):
    @override
    def setUp(self) -> None:
        """
        Initialise a model object with specific parameters for testing purposes.
        """
        self.model: md.ABModel = md.ABModel(
            HIERARCHY_NAMES,
            HIERARCHY_RW_DISTRIB,
            iterations=99,
            negation_threshold=0.89,
            radicalisation_threshold=0.45,
            visualise=False,
            model_id="TEST_MODEL",
        )

    def test_model_creation(self) -> None:
        """
        Check that the created model object is of the correct ABModel type.
        """
        self.assertIsInstance(
            self.model, md.ABModel, "ABModel object is not being created appropriately"
        )

    def test_parameter_setting(self) -> None:
        """
        Test function which checks that all input ABmodel parameters are being assigned correctly.
        """
        self.assertEqual(
            self.model.current_iteration,
            0,
            "Current iteration not being initialised properly",
        )
        self.assertEqual(
            self.model.agent_opinion_rw,
            (0.0, 0.1),
            "Agent opinion rw not being initialised correctly",
        )
        self.assertEqual(
            self.model.max_iterations, 99, "Max iterations not being stored correctly"
        )
        self.assertEqual(
            self.model.silencing_threshold,
            0.95,
            "Default value for silencing_threshold is not being applied correctly",
        )
        self.assertEqual(
            self.model.negation_threshold,
            0.89,
            "Negation threshold not being stored correctly",
        )
        self.assertEqual(
            self.model.radicalisation_threshold,
            0.45,
            "Radicalisation threshold not being stored correctly",
        )
        self.assertEqual(
            self.model.hierarchy_information,
            {
                HIERARCHY_NAMES[0]: HIERARCHY_RW_DISTRIB[0],
                HIERARCHY_NAMES[1]: HIERARCHY_RW_DISTRIB[1],
            },
            "Hierarchy information not being stored correctly as a dictionary",
        )
        self.assertFalse(
            self.model.visualise, "Visualisation flag not being stored correctly"
        )
        self.assertNotHasAttr(
            self.model,
            "visualiser",
            "Model visualiser is being initialised despite setting visualise to false",
        )
        self.assertFalse(
            self.model.debug,
            "Default value for debug is not being applied correctly"
        )
        self.assertIsInstance(
            self.model.logger,
            md.GATOHLogger,
            "Model logger is not being initialised as a GATOHLogger instance",
        )
        self.assertFalse(
            self.model.debug,
            "Default value for debug is not being applied correctly",
        )
        self.assertEqual(
            self.model.model_id,
            "TEST_MODEL",
            "Model ID not being stored correctly"
        )
        self.assertFalse(
            self.model.suppress_warnings,
            "Default value for suppress_warnings is not being applied correctly",
        )
        self.assertTrue(
            self.model.checkpointing,
            "Default value for checkpointing is not being applied correctly",
        )
        self.assertFalse(
            self.model.partial_iterations,
            "Default value for partial_iterations is not being applied correctly",
        )
        self.assertEqual(
            self.model.tracked_agents,
            {},
            "Default value for tracked_agents is not being applied correctly",
        )
        self.assertIsNone(
            self.model.parameters_to_track,
            "Default value for parameters_to_track is not being applied correctly",
        )
        self.assertFalse(
            self.model.simulate_groups,
            "Default value for simulate_groups is not being applied correctly",
        )
        self.assertEqual(
            self.model.data_file,
            "",
            "Default value for data_file is not being applied correctly",
        )
        self.assertEqual(
            self.model.visualisation_dir,
            "",
            "Default value for visualisation_dir is not being applied correctly",
        )
        self.assertIsInstance(
            self.model.agents,
            agt.AgentSet,
            "Model agents is not being initialised as an AgentSet",
        )
        self.assertIsInstance(
            self.model.graphs,
            gr.GraphSet,
            "Model graphs is not being initialised as a GraphSet",
        )
        self.assertIsInstance(
            self.model.groups,
            grp.GroupSet,
            "Model groups is not being initialised as a GroupSet",
        )
        self.assertIsInstance(
            self.model.group_graph,
            gr.GroupGraphSet,
            "Model group_graph is not being initialised as a GroupGraphSet",
        )

    def test_set_hierarchy_information(self) -> None:
        """
        Test that set_hierarchy_information is working correctly.
        """
        self.model.set_hierarchy_information(HIERARCHY_NAMES[0], (0.0, 0.4))
        self.assertEqual(
            self.model.hierarchy_information[HIERARCHY_NAMES[0]],
            (0.0, 0.4),
            "ABModel -- set_hierarchy_information is not updating the specified hierarchy",
        )
        self.assertEqual(
            self.model.hierarchy_information[HIERARCHY_NAMES[1]],
            HIERARCHY_RW_DISTRIB[1],
            "ABModel -- set_hierarchy_information is changing an unspecified hierarchy",
        )

    def test_set_agent_opinion_rw(self) -> None:
        """
        Test that set_agent_opinion_rw is working correctly.
        """
        self.model.set_agent_opinion_rw((0.0, 0.4))
        self.assertEqual(
            self.model.agent_opinion_rw,
            (0.0, 0.4),
            "ABModel -- set_agent_opinion_rw is not updating the agent_opinion_rw",
        )

    def test_set_visualise(self) -> None:
        """
        Test that set_visualise is working correctly.
        """
        self.model.set_visualise(True)
        self.assertTrue(
            self.model.visualise,
            "ABModel -- set_visualise is not updating the visualise flag",
        )
        self.assertIsInstance(
            self.model.visualiser,
            md.ABVisualiser,
            "ABModel -- set_visualise flipping from False to True is not initialising the model visualiser",
        )

    def test_set_visualise_same(self) -> None:
        """
        Test that set_visualise with the existing state does nothing.
        """
        self.model.set_visualise(False)
        self.assertFalse(
            self.model.visualise,
            "ABModel -- set_visualise with the current flag state as input is changing the visualise flag",
        )
        self.assertNotHasAttr(
            self.model,
            "visualiser",
            "ABModel -- set_visualise with False as an input is resulting in an existing model visualiser object",
        )

    def test_set_visualisation_dir(self) -> None:
        """
        Test that set_visualisation_dir is working correctly.

        This test assumes that model_creation is an existing directory in ./tests/test_saves.
        """
        # Ensure that the directory exists
        if not os.path.exists("./tests/test_saves/model_creation"):
            os.mkdir("./tests/test_saves/model_creation")
        self.model.set_visualisation_dir("./tests/test_saves/model_creation")
        self.assertEqual(
            self.model.visualisation_dir,
            "./tests/test_saves/model_creation",
            "ABModel -- set_visualisation_dir with a valid directory is not correctly updating the visualisation_dir attribute",
        )

    def test_forced_set_visualisation_dir(self) -> None:
        """
        Test that set_visualisation_dir with a non-existent directory but force=True is working correctly.
        """
        if os.path.exists("./tests/test_saves/model_creation/forced_dir"):
            os.rmdir("./tests/test_saves/model_creation/forced_dir")
        self.assertFalse(
            os.path.exists("./tests/test_saves/model_creation/forced_dir"),
            "ABModel -- Cannot proceed with the test as the non-existent directory already exists",
        )
        self.model.set_visualisation_dir("./tests/test_saves/model_creation/forced_dir", force=True)
        self.assertTrue(
            os.path.exists("./tests/test_saves/model_creation/forced_dir"),
            "ABModel -- Forced set_visualisation_dir is not creating the specified directory",
        )
        self.assertEqual(
            self.model.visualisation_dir,
            "./tests/test_saves/model_creation/forced_dir",
            "ABModel -- Forced set_visualisation_dir that created a non-existent directory is not updating visualisation_dir in the model",
        )

    def test_set_visualisation_dir_invalid(self) -> None:
        """
        Test that set_visualisation_dir with an invalid directory and no force raises the expected error.
        """
        with self.assertRaises(NotADirectoryError, msg="The path ./foo/bar/1312 does not point to a valid directory -- change the path or set 'force=True' to fix this") as cm:
            self.model.set_visualisation_dir("./foo/bar/1312")

    def test_override_current_iteration(self) -> None:
        """
        Test that override_current_iteration is working correctly.
        """
        self.model.override_current_iteration(44)
        self.assertEqual(
            self.model.current_iteration,
            44,
            "ABModel -- override_current_iteration is not updating the current_iteration correctly",
        )

    def test_set_max_iterations(self) -> None:
        """
        Test that set_max_iterations is working correctly.
        """
        self.model.set_max_iterations(444)
        self.assertEqual(
            self.model.max_iterations,
            444,
            "ABModel -- set_max_iterations is not updating max_iterations correctly",
        )

    def test_set_max_iterations_invalid(self) -> None:
        """
        Test that set_max_iterations with an invalid value raises the expected error.
        """
        with self.assertRaises(ValueError, msg="The max_iterations value -4 is invalid -- Use a positive integer") as cm:
            self.model.set_max_iterations(-4)

    def test_set_silencing_threshold(self) -> None:
        """
        Test that set_silencing_threshold is working correctly.
        """
        self.model.set_silencing_threshold(0.04)
        self.assertEqual(
            self.model.silencing_threshold,
            0.04,
            "ABModel -- set_silencing_threshold is not updating silencing_threshold correctly",
        )

    def test_set_silencing_threshold_invalid(self) -> None:
        """
        Test that set_silencing_threshold with an invalid value raises the expected error.
        """
        with self.assertRaises(ValueError, msg="The silencing threshold value of 10.2 is outside the valid range of [0.0, 1.0]") as cm:
            self.model.set_silencing_threshold(10.2)
        with self.assertRaises(ValueError, msg="The silencing threshold value of -0.4 is outside the valid range of [0.0, 1.0]") as cm:
            self.model.set_silencing_threshold(-0.4)

    def test_set_negation_threshold(self) -> None:
        """
        Test that set_negation_threshold is working correctly.
        """
        self.model.set_negation_threshold(0.04)
        self.assertEqual(
            self.model.negation_threshold,
            0.04,
            "ABModel -- set_negation_threshold is not updating negation_threshold correctly",
        )

    def test_set_negation_threshold_invalid(self) -> None:
        """
        Test that set_negation_threshold with an invalid value raises the expected error.
        """
        with self.assertRaises(ValueError, msg="The negation threshold value of 10.2 is outside the valid range of [0.0, 1.0]") as cm:
            self.model.set_negation_threshold(10.2)
        with self.assertRaises(ValueError, msg="The negation threshold value of -0.4 is outside the valid range of [0.0, 1.0]") as cm:
            self.model.set_negation_threshold(-0.4)

    def test_set_radicalisation_threshold(self) -> None:
        """
        Test that set_radicalisation_threshold is working correctly.
        """
        self.model.set_radicalisation_threshold(0.04)
        self.assertEqual(
            self.model.radicalisation_threshold,
            0.04,
            "ABModel -- set_radicalisation_threshold is not updating radicalisation_threshold correctly",
        )

    def test_set_radicalisation_threshold_invalid(self) -> None:
        """
        Test that set_radicalisation_threshold with an invalid value raises the expected error.
        """
        with self.assertRaises(ValueError, msg="The radicalisation threshold value of 10.2 is outside the valid range of [0.0, 1.0]") as cm:
            self.model.set_radicalisation_threshold(10.2)
        with self.assertRaises(ValueError, msg="The radicalisation threshold value of -0.4 is outside the valid range of [0.0, 1.0]") as cm:
            self.model.set_radicalisation_threshold(-0.4)

    def test_set_suppress_warnings(self) -> None:
        """
        Test that set_suppress_warnings is working correctly.
        """
        self.model.set_suppress_warnings(True)
        self.assertTrue(
            self.model.suppress_warnings,
            "ABModel -- set_suppress_warnings is not updating the suppress_warnings flag",
        )

    def test_set_checkpointing(self) -> None:
        """
        Test that set_checkpointing is working correctly.
        """
        self.model.set_checkpointing(False)
        self.assertFalse(
            self.model.checkpointing,
            "ABModel -- set_checkpointing is not updating the checkpointing flag",
        )

    def test_set_partial_iterations(self) -> None:
        """
        Test that set_partial_iterations is working correctly.
        """
        self.model.set_partial_iterations(True)
        self.assertTrue(
            self.model.partial_iterations,
            "ABModel -- set_partial_iterations is not updating the partial_iterations flag",
        )

    def test_set_simulate_groups(self) -> None:
        """
        Test that set_simulate_groups is working correctly.
        """
        self.model.set_simulate_groups(True)
        self.assertTrue(
            self.model.simulate_groups,
            "ABModel -- set_simulate_groups is not updating the simulate_groups flag",
        )

    def test_set_save_dir(self) -> None:
        """
        Test that set_save_dir is working correctly.
        """
        if not os.path.exists("./tests/test_saves/model_creation"):
            os.mkdir("./tests/test_saves/model_creation")
        self.model.set_save_dir("./tests/test_saves/model_creation")
        self.assertEqual(
            self.model.save_dir,
            "./tests/test_saves/model_creation",
            "ABModel -- set_save_dir with a valid directory is not updating save_dir correctly",
        )

    def test_set_save_dir_forced(self) -> None:
        """
        Test that a forced set_save_dir is working correctly.
        """
        if os.path.exists("./tests/test_saves/model_creation"):
            rmtree("./tests/test_saves/model_creation")
        self.model.set_save_dir("./tests/test_saves/model_creation", force=True)
        self.assertTrue(
            os.path.exists("./tests/test_saves/model_creation"),
            "ABModel -- forced set_save_dir is not creating the specified directory",
        )
        self.assertEqual(
            self.model.save_dir,
            "./tests/test_saves/model_creation",
            "ABModel -- forced set_save_dir that created a directory did not correctly update the model's save_dir",
        )

    def test_set_save_dir_invalid(self) -> None:
        """
        Test that an unforced set_save_dir with an invalid directory raises the expected error.
        """
        with self.assertRaises(NotADirectoryError, msg="The path ./foo/bar/1312 does not point to a valid directory -- change the path or set 'force=True' to fix this") as cm:
            self.model.set_save_dir("./foo/bar/1312")

    def test_set_data_file(self) -> None:
        """
        Test that set_data_file is working correctly.

        This function does not have existence checking like set_save_dir or set_visualisation_dir,
        the error is handled within the GATOHLogger at the time of attempting to write results.
        """
        self.model.set_data_file("./tests/test_saves/model_creation/foobar_1312.csv")
        self.assertEqual(
            self.model.data_file,
            "./tests/test_saves/model_creation/foobar_1312.csv",
            "ABModel -- set_data_file is not updating data_file correctly",
        )

    def test_set_model_id(self) -> None:
        """
        Test that set_model_id is working correctly.
        """
        self.model.set_model_id("MODEL_TEST")
        self.assertEqual(
            self.model.model_id,
            "MODEL_TEST",
            "ABModel -- set_model_id is not updating model_id correctly",
        )

    def test_track_model_parameters_type_error(self) -> None:
        """
        Test that track_model_parameters with an invalid data type will raise the expected error.
        """
        parameters = [3, "silencing_threshold", False]
        with self.assertRaises(TypeError, msg="One or more parameters to track have not been provided as string") as cm:
            self.model.track_model_parameters(parameters)
        self.assertIsNone(
            self.model.parameters_to_track,
            "ABModel -- track_model_parameters with an incorrect data type is creating a parameters_to_track list",
        )

    def test_track_model_parameters_nonexistent(self) -> None:
        """
        Test that track_model_parameters with a non-existent parameter will raise the expected error.
        """
        parameters = ["silencing_threshold", "foobar", "radicalisation_threshold"]
        with self.assertRaises(KeyError, msg="The parameter 'foobar' does not exist in the ABModel object") as cm:
            self.model.track_model_parameters(parameters)
        self.assertIsNone(
            self.model.parameters_to_track,
            "ABModel -- track_model_parameters with a non-existent parameter is creating a parameters_to_track list",
        )

    def test_track_model_parameters(self) -> None:
        """
        Test that track_model_parameters with all valid parameters is working as intended.
        """
        parameters = ["silencing_threshold", "radicalisation_threshold"]
        self.model.track_model_parameters(parameters)
        self.assertIsInstance(
            self.model.parameters_to_track,
            list,
            "ABModel -- a valid track_model_parameters call is not creating a parameters list",
        )
        self.assertEqual(
            self.model.parameters_to_track,
            parameters,
            "ABModel -- a valid track_model_parameters call is not creating the correct parameters list",
        )

    def test_track_model_parameters_multiple(self) -> None:
        """
        Test that repeated, valid calls to track_model_parameters are working as intended.
        """
        parameters_one: list[str] = ["silencing_threshold"]
        parameters_two: list[str] = ["radicalisation_threshold", "negation_threshold"]
        self.model.track_model_parameters(parameters_one)
        self.model.track_model_parameters(parameters_two)
        self.assertEqual(
            len(self.model.parameters_to_track),
            3,
            "ABModel -- repeated, valid calls to track_model_parameters are not adding the correct number of parameters to track",
        )
        all_parameters: list[str] = parameters_one + parameters_two
        self.assertEqual(
            self.model.parameters_to_track,
            all_parameters,
            "ABModel -- repeated, valid calls to track_model_parameters are not adding the appropriate parameters to track",
        )

    def test_get_tracked_parameters_empty(self) -> None:
        """
        Test that get_tracked_parameters on an empty model will return None.
        """
        tracked_parameters = self.model.get_tracked_parameters()
        self.assertIsNone(
            tracked_parameters,
            "ABModel -- get_tracked_parameters when no parameters are being tracked did not return None",
        )

    def test_get_tracked_parameters(self) -> None:
        """
        Test that get_tracked_parameters is working as intended.
        """
        self.model.track_model_parameters(["silencing_threshold", "radicalisation_threshold", "negation_threshold"])
        tracked_parameters = self.model.get_tracked_parameters()
        self.assertIsInstance(
            tracked_parameters,
            dict,
            "ABModel -- get_tracked_parameters when parameters are being tracked did not return a dict",
        )
        self.assertEqual(
            len(tracked_parameters),
            3,
            "ABModel -- get_tracked_parameters is not reporting all of the tracked model parameters",
        )
        self.assertEqual(
            tracked_parameters.get("silencing_threshold"),
            0.95,
            "ABModel -- get_tracked_parameters is not reporting the correct value for one or more tracked parameters",
        )
        self.assertEqual(
            tracked_parameters.get("radicalisation_threshold"),
            0.45,
            "ABModel -- get_tracked_parameters is not reporting the correct value for one or more tracked parameters",
        )
        self.assertEqual(
            tracked_parameters.get("negation_threshold"),
            0.89,
            "ABModel -- get_tracked_parameters is not reporting the correct value for one or more tracked parameters",
        )

    def test_get_tracked_agent_attributes_empty(self) -> None:
        """
        Test that get_tracked_agent_attributes when no agent attributes are being tracked is returning an empty dictionary.
        """
        tracked_agt_attribs: dict[str, dict[str, Any]] = self.model.get_tracked_agent_attributes()
        self.assertIsInstance(
            tracked_agt_attribs,
            dict,
            "ABModel -- get_tracked_agent_attributes when no attributes are being tracked is not returning a dictionary value",
        )
        self.assertEqual(
            tracked_agt_attribs,
            {},
            "ABModel -- get_tracked_agent_attributes when no attributes are being tracked is not returning an empty dictionary",
        )

    def test_get_tracked_agent_attributes(self) -> None:
        """
        Test that get_tracked_agent_attributes when some agent attributes are being tracked is working as intended.
        """
        # TODO: Implement this test

    def test_add_graph(self) -> None:
        """
        Test function that checks if adding Graphs to the ABModel is working correctly.
        """
        test_graph: gr.Graph = gr.Graph(HIERARCHY_NAMES[0], HIERARCHY_RW_DISTRIB[0])
        model_graphset: gr.GraphSet = self.model.add_graph(test_graph)
        self.assertIsInstance(
            model_graphset,
            gr.GraphSet,
            "The add_graph function's return type is not the expected one",
        )
        self.assertIn(
            test_graph,
            self.model.graphs,
            "The add_graph function is not adding the first Graph to the GraphSet",
        )
        test_graph_2: gr.Graph = gr.Graph(HIERARCHY_NAMES[1], HIERARCHY_RW_DISTRIB[1])
        model_graphset = self.model.add_graph(test_graph_2)
        self.assertIn(
            test_graph_2,
            self.model.graphs,
            "The add_graph function is not adding subsequent graphs to the GraphSet",
        )
        self.assertIn(
            test_graph,
            self.model.graphs,
            "The add_graph function is causing existing graphs to disappear from the GraphSet",
        )

    def test_add_graphs(self) -> None:
        """
        Test function that checks if adding multiple Graphs at once to the ABModel is working correctly.
        """
        graphs_to_add: list[gr.Graph] = [
            gr.Graph(HIERARCHY_NAMES[0], HIERARCHY_RW_DISTRIB[0]),
            gr.Graph(HIERARCHY_NAMES[1], HIERARCHY_RW_DISTRIB[1]),
        ]
        graphset: gr.GraphSet = self.model.add_graphs(
            graphs_to_add, HIERARCHY_NAMES, HIERARCHY_RW_DISTRIB
        )
        self.assertIsInstance(
            graphset, gr.GraphSet, "The add_graphs return type is not the expected one"
        )
        for graph in graphs_to_add:
            self.assertIn(
                graph,
                self.model.graphs,
                f"The add_graphs function did not add Graph {graph.name} correctly",
            )
            graph_in_graphset: gr.Graph | None = self.model.graphs.get_hierarchy(
                graph.name
            )
            if graph_in_graphset is not None:
                self.assertEqual(
                    graph.rw_params,
                    graph_in_graphset.rw_params,
                    "The add_graphs function is dropping the rw_params attribute during addition",
                )
                # Checks on further parameters are done from the graph-centric tests scripts

    def test_add_agent(self) -> None:
        """
        Test function that checks if adding Agents to the ABModel is working correctly.
        """
        # Remember to actually add the graphs before agents...
        graphs_to_add: list[gr.Graph] = [
            gr.Graph(HIERARCHY_NAMES[0], HIERARCHY_RW_DISTRIB[0], suppress_warnings=True),
            gr.Graph(HIERARCHY_NAMES[1], HIERARCHY_RW_DISTRIB[1], suppress_warnings=True),
        ]
        _ = self.model.add_graphs(graphs_to_add, HIERARCHY_NAMES, HIERARCHY_RW_DISTRIB)
        test_agent: agt.Agent = agt.Agent(
            "TEST_AGENT",
            {
                "Test_1": 0.2,
                "Test_2": 0.77,
            },
            -0.44,
            True,
            ("social", 0.05),
        )
        agent_index: int = self.model.add_agent(test_agent)
        # Index should equal 0 as no other agents have been previously added
        self.assertGreaterEqual(
            agent_index,
            0,
            "The add_agent function is reporting an incorrect index when adding the first agent",
        )
        self.assertIn(
            test_agent,
            self.model.agents,
            "The add_agent function did not correctly add the first agent to the AgentSet",
        )
        test_agent_2: agt.Agent = agt.Agent(
            "TEST_AGENT_2",
            {
                "Test_1": 0.1,
                "Test_2": 0.98,
            },
            0.3,
            False,
            ("impulsive", 0.99),
        )
        agent_2_index: int = self.model.add_agent(test_agent_2)
        # Now agent_2_index should equal 1 as an agent was previously added
        self.assertGreaterEqual(
            agent_2_index,
            1,
            "The add_agent function is reporting an incorrect index when adding subsequent agents",
        )
        self.assertIn(
            test_agent_2,
            self.model.agents,
            "The add_agent function did not correctly add subsequent agents to the AgentSet",
        )
        # Check that the original agent is still in the AgentSet
        self.assertIn(
            test_agent,
            self.model.agents,
            "The add_agent function caused an existing agent to disappear from the AgentSet",
        )

    def test_add_agents(self) -> None:
        """
        Test function that checks if addings multiple Agents at once to the ABModel is working correctly.
        """
        # Remember to actually add the graphs before the agents...
        graphs_to_add: list[gr.Graph] = [
            gr.Graph(HIERARCHY_NAMES[0], HIERARCHY_RW_DISTRIB[0], suppress_warnings=True),
            gr.Graph(HIERARCHY_NAMES[1], HIERARCHY_RW_DISTRIB[1], suppress_warnings=True),
        ]
        _ = self.model.add_graphs(graphs_to_add, HIERARCHY_NAMES, HIERARCHY_RW_DISTRIB)
        agents_to_add: list[agt.Agent] = [
            agt.Agent(
                "TEST_AGENT_1",
                0.1,
                True,
                ("social", 0.1),
            ),
            agt.Agent(
                "TEST_AGENT_2",
                0.2,
                False,
                ("social", 0.2),
            ),
            agt.Agent(
                "TEST_AGENT_3",
                0.3,
                True,
                ("impulsive", 0.3),
            ),
            agt.Agent(
                "TEST_AGENT_4",
                0.99,
                True,
                ("erratic", 0.01),
            ),
        ]
        agentset: agt.AgentSet = self.model.add_agents(agents_to_add)
        self.assertIsInstance(
            agentset,
            agt.AgentSet,
            "The add_agents function's return type is not the expected one",
        )
        for idx, agent in enumerate(agents_to_add):
            self.assertIn(
                agent,
                self.model.agents,
                f"The add_agents function did not add Agent {agent.id} correctly",
            )
            agent_in_agentset: agt.Agent | None = self.model.agents.get_agent_by_id(
                agent.id
            )
            if (
                agent_in_agentset is not None
            ):  # Simply added to satisfy typing requirements
                self.assertEqual(
                    idx,
                    agent_in_agentset.index,
                    "The add_agents function is adding agent objects out of order",
                )
                self.assertEqual(
                    agent.id,
                    agent_in_agentset.id,
                    "The add_agents function is dropping the id attribute during addition",
                )
                self.assertEqual(
                    agent.opinion,
                    agent_in_agentset.opinion,
                    "The add_agents function is dropping the opinion attribute during addition",
                )
                self.assertEqual(
                    agent.personal_benefit,
                    agent_in_agentset.personal_benefit,
                    "The add_agents function is dropping the personal_benefit attribute during addition",
                )
                self.assertEqual(
                    agent.personality,
                    agent_in_agentset.personality,
                    "The add_agents function is dropping the personality attribute during addition",
                )
                self.assertEqual(
                    agent.social_susceptibility,
                    agent_in_agentset.social_susceptibility,
                    "The add_agents function is dropping the social_susceptibility attribute during addition",
                )

    def test_add_agents_to_hierarchy_nonvalue(self) -> None:
        """
        Test that add_agents_to_hierarchy with a non-Agent object included raises the expected error.
        """
        _ = self.model.add_graph(gr.Graph(HIERARCHY_NAMES[0], HIERARCHY_RW_DISTRIB[0], suppress_warnings=True))
        invalid_agents = [
            agt.Agent("TEST_AGENT_1"),
            agt.Agent("TEST_AGENT_2"),
            "IMPOSTOR",
            agt.Agent("TEST_AGENT_4"),
        ]
        with self.assertRaises(TypeError, msg=f"The object at index 2 of the input iterable is not a valid Agent object -- cannot add it to the hierarchy graph '{HIERARCHY_NAMES[0]}'") as cm:
            self.model.add_agents_to_hierarchy(invalid_agents, HIERARCHY_NAMES[0])

    def test_add_agents_to_hierarchy_nonhierarchy(self) -> None:
        """
        Test that add_agents_to_hierarchy with a non-existent hierarchy raises the expected error.
        """
        agents_to_add: list[agt.Agent] = [
            agt.Agent("TEST_AGENT_1"),
            agt.Agent("TEST_AGENT_2"),
        ]
        with self.assertRaises(KeyError, msg="The specified hierarchy 'foo' does not exist in the GraphSet -- cannot add agents to it") as cm:
            self.model.add_agents_to_hierarchy(agents_to_add, "foo")

    def test_add_agents_to_hierarchy(self) -> None:
        """
        Test that add_agents_to_hierarchy is working correctly.
        """
        graphs_to_add: list[gr.Graph] = [
            gr.Graph(HIERARCHY_NAMES[0], HIERARCHY_RW_DISTRIB[0], suppress_warnings=True),
            gr.Graph(HIERARCHY_NAMES[1], HIERARCHY_RW_DISTRIB[1], suppress_warnings=True),
        ]
        _ = self.model.add_graphs(graphs_to_add, HIERARCHY_NAMES, HIERARCHY_RW_DISTRIB)
        agents_to_add: list[agt.Agent] = [
            agt.Agent("TEST_AGENT_1"),
            agt.Agent("TEST_AGENT_2"),
            agt.Agent("TEST_AGENT_3"),
            agt.Agent("TEST_AGENT_4"),
        ]
        self.model.add_agents_to_hierarchy(agents_to_add, HIERARCHY_NAMES[0])
        self.assertEqual(
            len(self.model.agents),
            4,
            "ABModel -- add_agents_to_hierarchy is not adding non-existing agents to the model's agentset",
        )
        for agent in agents_to_add:
            self.assertIn(
                agent,
                self.model.agents,
                "ABModel -- add_agents_to_hierarchy is not correctly adding some non-existing agent to the model's agentset",
            )
        test_1: gr.Graph | None = self.model.graphs.get_hierarchy(HIERARCHY_NAMES[0])
        test_2: gr.Graph | None = self.model.graphs.get_hierarchy(HIERARCHY_NAMES[1])
        if test_1 is not None and test_2 is not None:
            self.assertEqual(
                test_1.node_count,
                4,
                "ABModel -- add_agents_to_hierarchy did not add the agents to the specified hierarchy"
            )
            self.assertEqual(
                test_2.node_count,
                0,
                "ABModel -- add_agents_to_hierarchy added agents to an unspecified hierarchy"
            )
            for node in test_1.graph.nodes():
                self.assertIn(
                    node.agent,
                    agents_to_add,
                    "ABModel -- add_agents_to_hierarchy added an unexpected agent to the specified hierarchy"
                )
        self.model.add_agents_to_hierarchy(agents_to_add[:-2], HIERARCHY_NAMES[1])
        self.assertEqual(
            len(self.model.agents),
            4,
            "ABModel -- add_agents_to_hierarchy is adding agents which already exist to the model's agentset",
        )
        test_1 = self.model.graphs.get_hierarchy(HIERARCHY_NAMES[0])
        test_2 = self.model.graphs.get_hierarchy(HIERARCHY_NAMES[1])
        if test_1 is not None and test_2 is not None:
            self.assertEqual(
                test_1.node_count,
                4,
                "ABModel -- add_agents_to_hierarchy is changing the agents of a populated, existing hierarchy",
            )
            self.assertEqual(
                test_2.node_count,
                2,
                "ABModel -- add_agents_to_hierarchy is not adding the correct number of agents to the specified hierarchy",
            )
            for node in test_2.graph.nodes():
                self.assertIn(
                    node.agent,
                    agents_to_add[:-2],
                    "ABModel -- add_agents_to_hierarchy added an unexpected agent to the specified hierarchy"
                )

    def test_generate_agents(self) -> None:
        """
        Test function that checks if random Agent generation is working as intended within an ABModel.
        """
        self.model.generate_agents(
            "TEST",  # id_base
            {  # personality_probs
                "social": 0.4,
                "rational": 0.4,
                "impulsive": 0.2,
            },
            number=8,
        )
        self.assertEqual(
            len(self.model.agents),
            8,
            "The generate_agents function is creating an unexpected number of Agents",
        )
        for agent in self.model.agents:
            self.assertStartsWith(
                agent.id,
                "TEST",
                "The generate_agents function is not applying the correct ID base",
            )
            self.assertEndsWith(
                agent.id,
                f"{agent.index:04}",
                "The generate_agents function is not numbering the IDs appropriately",
            )
            self.assertIn(
                agent.personality,
                ["social", "rational", "impulsive"],
                "The generate_agents function is creating agents with non-specified personality types",
            )
            for hierarchy in self.model.hierarchy_information:
                self.assertIn(
                    hierarchy,
                    list(agent.social_weightings.keys()),
                    "The generate_agents function is not generating hierarchy weightings for all model hierarchies",
                )

    def test_add_relationships_to_hierarchy_nonvalue(self) -> None:
        """
        Test that add_relationships_to_hierarchy with a missing value raises the expected error.
        """
        _ = self.model.add_graph(gr.Graph(HIERARCHY_NAMES[0], HIERARCHY_RW_DISTRIB[0], suppress_warnings=True))
        agents_to_add: list[agt.Agent] = [
            agt.Agent("TEST_AGENT_1"),
            agt.Agent("TEST_AGENT_2"),
        ]
        self.model.add_agents_to_hierarchy(agents_to_add, HIERARCHY_NAMES[0])
        invalid_relationships: dict[str, list[Any]] = {
            "from_node": [0, 1],
            "weighting": [0.44, 0.2]
        }
        with self.assertRaises(ValueError, msg="The relationships information is missing one of the required 'from_node' or 'to_node' keys") as cm:
            self.model.add_relationships_to_hierarchy(invalid_relationships, HIERARCHY_NAMES[0])

    def test_add_relationships_to_hierarchy_nonhierarchy(self) -> None:
        """
        Test that add_relationships_to_hierarchy with a non-existent hierarchy raises the expected error.
        """
        relationships_to_add: dict[str, list[Any]] = {
            "from_node": [13, 12],
            "to_node": [12, 13],
        }
        with self.assertRaises(KeyError, msg="The specified hierarchy 'foo' does not exist in the GraphSet -- cannot add relationships to it") as cm:
            self.model.add_relationships_to_hierarchy(relationships_to_add, "foo")

    def test_add_relationships_to_hierarchy(self) -> None:
        """
        Test that add_relationships_to_hierarchy is working correctly.
        """
        graphs_to_add: list[gr.Graph] = [
            gr.Graph(HIERARCHY_NAMES[0], HIERARCHY_RW_DISTRIB[0], suppress_warnings=True),
            gr.Graph(HIERARCHY_NAMES[1], HIERARCHY_RW_DISTRIB[1], suppress_warnings=True),
        ]
        _ = self.model.add_graphs(graphs_to_add, HIERARCHY_NAMES, HIERARCHY_RW_DISTRIB)
        agents_to_add: list[agt.Agent] = [
            agt.Agent("TEST_AGENT_1"),
            agt.Agent("TEST_AGENT_2"),
        ]
        self.model.add_agents_to_hierarchy(agents_to_add, HIERARCHY_NAMES[0])
        self.model.add_agents_to_hierarchy(agents_to_add, HIERARCHY_NAMES[1])
        relationships_to_add: dict[str, list[Any]] = {
            "from_node": [0, 1],
            "to_node": [1, 0],
            "weighting": [0.13, -0.12],
        }
        self.model.add_relationships_to_hierarchy(relationships_to_add, HIERARCHY_NAMES[0])
        test_1: gr.Graph | None = self.model.graphs.get_hierarchy(HIERARCHY_NAMES[0])
        test_2: gr.Graph | None = self.model.graphs.get_hierarchy(HIERARCHY_NAMES[1])
        if test_1 is not None and test_2 is not None:
            self.assertEqual(
                test_1.edge_count,
                2,
                "ABModel -- add_relationships_to_hierarchy did not add the relationships to the specified hierarchy",
            )
            # ==== Check that the added edges have the appropriate attributes ====
            for idx, edge in enumerate(test_1.graph.edges()):
                self.assertEqual(
                    edge.from_node,
                    relationships_to_add["from_node"][idx],
                    "ABModel -- add_relationships_to_hierarchy added a relationship with an incorrect from_node",
                )
                self.assertEqual(
                    edge.to_node,
                    relationships_to_add["to_node"][idx],
                    "ABModel -- add_relationships_to_hierarchy added a relationship with an incorrect to_node",
                )
                self.assertEqual(
                    edge.weighting,
                    relationships_to_add["weighting"][idx],
                    "ABModel -- add_relationships_to_hierarchy added a relationship with an incorrect weighting",
                )
            # ==== End of attribute check ====
            self.assertEqual(
                test_2.edge_count,
                0,
                "ABModel -- add_relationships_to_hierarchy added relationships to an unspecified hierarchy",
            )
        # Further checking for adding relationships to graph test_2 as in test_add_agents_to_hierarchy should not be necessary here...

    def test_add_group(self) -> None:
        """
        Test that add_group is working as intended.
        """
        new_group: grp.Group = grp.Group("TEST")
        index: int = self.model.add_group(new_group)
        self.assertIsInstance(
            index,
            int,
            "ABModel -- add_group is not returning an integer value",
        )
        self.assertEqual(
            index,
            len(self.model.groups) - 1,
            "ABModel -- add_group is not returning the correct index for the added group",
        )
        self.assertIn(
            new_group,
            self.model.groups,
            "ABModel -- add_group is not correctly adding the group object to the model's groupset",
        )

    def test_add_groups(self) -> None:
        """
        Test that add_groups is working as intended.
        """
        new_groups: list[grp.Group] = [grp.Group(f"TEST{i}") for i in range(10)]
        group_set: grp.GroupSet = self.model.add_groups(new_groups)
        self.assertIsInstance(
            group_set,
            grp.GroupSet,
            "ABModel -- add_groups is not returning a reference to the GroupSet",
        )
        self.assertEqual(
            len(self.model.groups),
            10,
            "ABModel -- add_groups is not adding the correct number of groups to the model's group set",
        )
        for idx, new_group in enumerate(new_groups):
            self.assertIn(
                new_group,
                self.model.groups,
                "ABModel -- add_groups is not correctly adding one or more groups to the group set",
            )
            self.assertEqual(
                new_group.index,
                idx,
                "ABModel -- add_groups is not correctly updating one or more added groups' indices",
            )

    def test_generate_groups_invalid(self) -> None:
        """
        Test that generate_groups on an empty model will raise the expected error.
        """
        with self.assertRaises(RuntimeError, msg="Group generation requires for valid agents and graphs to exist in the model") as cm:
            self.model.generate_groups(
                "TEST",
                {
                    "close": 0.33,
                    "neutral": 0.33,
                    "distant": 0.33,
                },
            )

    def test_generate_groups(self) -> None:
        """
        Test that generate_groups is working as intended.
        """
        new_agents: list[agt.Agent] = [agt.Agent(f"AGENT{i}") for i in range(20)]
        graphs_to_add: list[gr.Graph] = [
            gr.Graph(HIERARCHY_NAMES[0], HIERARCHY_RW_DISTRIB[0]),
            gr.Graph(HIERARCHY_NAMES[1], HIERARCHY_RW_DISTRIB[1]),
        ]
        graphset: gr.GraphSet = self.model.add_graphs(
            graphs_to_add, HIERARCHY_NAMES, HIERARCHY_RW_DISTRIB
        )
        self.model.add_agents_to_hierarchy(new_agents[:10], HIERARCHY_NAMES[0])
        self.model.add_agents_to_hierarchy(new_agents[10:], HIERARCHY_NAMES[1])
        self.model.generate_groups(
            "GROUP",
            {
                "close": 0.5,
                "distant": 0.5,
            },
            n_groups=2,
            max_iters=10,
        )
        self.assertEqual(
            len(self.model.groups),
            4,
            "ABModel -- a valid call to generate_groups did not create the expected number of groups",
        )
        model_agent_ids: list[str] = self.model.get_agent_ids()
        for group in self.model.groups:
            self.assertIn(
                group.hierarchy,
                HIERARCHY_NAMES,
                "ABModel -- a valid call to generate_groups created one or more groups belonging to non-existent hierarchies",
            )
            self.assertIn(
                group.cohesion,
                ["close", "distant"],
                "ABModel -- a valid call to generate_groups created one or more groups with cohesion types that were not specified"
            )
            for member in group.members:
                self.assertIn(
                    member,
                    model_agent_ids,
                    "ABModel -- a valid call to generate_groups created one or more groups with agent members that do not exist in the model",
                )


    @override
    def tearDown(self) -> None:
        """
        Reset the model object to run subsequent tests.
        """
        del self.model
