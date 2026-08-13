from __future__ import annotations

import os
import unittest as ut
from multiprocessing import Pool
from typing import override
from shutil import rmtree

import gatoh.agents as agt
import gatoh.graphs as gr
import gatoh.model as md

MODEL_ID: str = "TEST_SIMULATION"
HIERARCHY_NAMES: list[str] = ["A", "B", "C", "D"]
HIERARCHY_RW_DISTRIB: dict[str, tuple[float, float]] = {
    "A": (0.0, 0.0),
    "B": (0.0, 0.1),
    "C": (0.0, 0.4),
    "D": (0.0, 0.05),
}
ROOTDIR: str = "./tests/test_saves/model_simulation_multi"
SAVEPATHS: dict[str, str] = {
    "savedir": f"{ROOTDIR}/model_save",
    "savefile": f"{ROOTDIR}/model_save/{MODEL_ID}_model_variables.csv",
    "visualisation": f"{ROOTDIR}/visualisation_output",
}


class TestModelSimulationMulti(ut.TestCase):
    @override
    def setUp(self) -> None:
        # Ensure that the root directory exists
        if not os.path.exists(ROOTDIR):
            os.mkdir(ROOTDIR)
        # Reset the save dir
        if os.path.exists(SAVEPATHS["savedir"]):
            rmtree(SAVEPATHS["savedir"])
            os.mkdir(SAVEPATHS["savedir"])
        # The multiprocessing worker pool
        self.worker_pool = Pool()
        self.model: md.ABModel = md.ABModel(
            HIERARCHY_NAMES,
            list(HIERARCHY_RW_DISTRIB.values()),
            suppress_warnings=True,
            iterations=2,
            save_dir=SAVEPATHS["savedir"],
            data_file=SAVEPATHS["savefile"],
            visualisation_dir=SAVEPATHS["visualisation"],
            model_id=MODEL_ID,
        )
        # Randomly generate 10 Agents for use in the simulation
        self.model.generate_agents(
            "TEST",  # id_base
            {  # personality_probs
                "social": 0.4,
                "rational": 0.4,
                "impulsive": 0.2,
            },
            number=10,
        )
        # Genertate a graph for "A" with assurance of full agent membership
        self.model.generate_graphs(
            [HIERARCHY_NAMES[0]],
            self.model.agents,
            rw_params=[HIERARCHY_RW_DISTRIB[HIERARCHY_NAMES[0]]],
            ensure_connected={HIERARCHY_NAMES[0]: False},
        )
        # Generate graphs for the remaining hierarchies using agent subsetting
        self.model.generate_graphs(
            HIERARCHY_NAMES[1:],
            self.model.agents,
            agent_subsetting=True,
            rw_params=list(HIERARCHY_RW_DISTRIB.values())[1:],
            ensure_connected={hierarchy: False for hierarchy in HIERARCHY_NAMES[1:]},
        )

    def test_iterate(self) -> None:
        """
        Test function that checks if the ABModel with a multiprocessing worker pool is iterating correctly during runtime.
        """
        self.model.iterate(worker_pool=self.worker_pool)
        self.assertEqual(
            self.model.current_iteration,
            10,
            "The ABModel is not incrementing its current_iteration attribute correctly in the iterate function with multiprocessing"
        )
        logger_data_saved: bool = os.path.exists(SAVEPATHS["savefile"])
        self.assertTrue(
            logger_data_saved,
            "The ABModel's iterate function with multiprocessing did not properly call the logger's save function at the end of the iterations",
        )

    def test_iterate_partial(self) -> None:
        """
        Test function that checks if the ABModel with a multiprocessing worker pool is partially iterating correctly during runtime.
        """
        self.model.set_partial_iterations(True)
        self.model.iterate(worker_pool=self.worker_pool)
        self.assertEqual(
            self.model.current_iteration,
            2,
            "The ABModel is not incrementing its current_iteration attribute correctly in the partial iterate function with multiprocessing",
        )
        logger_data_saved: bool = os.path.exists(SAVEPATHS["savefile"])
        self.assertTrue(
            logger_data_saved,
            "The ABModel's partial iterate function with multiprocessing did not properly call the logger's save function at the end of the iterations",
        )

    def test_save_model(self) -> None:
        """
        Test function that checks if the ABModel with multiprocessing will save itself appropriately.
        """
        self.model.save_model()
        save_dir_exists: bool = os.path.exists(SAVEPATHS["savedir"])
        self.assertTrue(
            save_dir_exists,
            "The ABModel's save_model function did not create the correct save directory after running with multiprocessing",
        )
        savedir_filelist: list[str] = list(os.walk(self.model.save_dir))[0][2]
        graphml_exists: bool = False
        yaml_exists: bool = False
        agentset_exists: bool = False
        graphset_exists: bool = False
        unexpected_file_written: bool = False
        for file in savedir_filelist:
            filename, file_extension = os.path.splitext(file)
            match file_extension:
                case ".graphml":
                    self.assertEqual(
                        filename,
                        "graph_base_graph",
                        "The ABModel's save_model function did not name the base graph's graphml file as expected after running with multiprocessing",
                    )
                    graphml_exists = True
                case ".yaml":
                    self.assertStartsWith(
                        filename,
                        "model_",
                        "The ABModel's save_model function did not save the config file with the correct prefix after running with multiprocessing",
                    )
                    yaml_exists = True
                case ".zip":
                    self.assertStartsWith(
                        filename,
                        ("_agentset", "_graphset"),
                        "The ABModel's save_model function did not create the agentset or graphset zipfiles with the correct names after running with multiprocessing",
                    )
                    if filename == "_agentset":
                        agentset_exists = True
                    elif filename == "_graphset":
                        graphset_exists = True
                case _:
                    unexpected_file_written = True
        self.assertTrue(
            graphml_exists,
            "The ABModel's save_model function did not create a base graph graphml file after running with multiprocessing",
        )
        self.assertTrue(
            yaml_exists,
            "The ABModel's save_model function did not create a model YAML config file after running with multiprocessing",
        )
        self.assertTrue(
            agentset_exists,
            "The ABModel's save_model function did not create an agentset zipfile after running with multiprocessing",
        )
        self.assertTrue(
            graphset_exists,
            "The ABModel's save_model function did not create a graphset zipfile after running with multiprocessing",
        )
        self.assertFalse(
            unexpected_file_written,
            "The ABModel's save_model function wrote an unexpected file to the save directory after running with multiprocessing",
        )

    @override
    def tearDown(self) -> None:
        del self.model
        self.worker_pool.close()
