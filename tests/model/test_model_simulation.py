from __future__ import annotations

import os
import unittest as ut
from typing import override

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
SAVEPATHS: dict[str, str] = {
    "savedir": "./tests/test_saves/model_simulation",
    "savefile": f"./tests/test_saves/model_simulation/{MODEL_ID}_model_variables.csv",
    "visualisation": "./tests/test_saves/model_simulation/visualisation_output",
}


class TestModelSimulation(ut.TestCase):
    @classmethod
    @override
    def setUpClass(cls) -> None:
        cls._model: md.ABModel = md.ABModel(
            HIERARCHY_NAMES,
            list(HIERARCHY_RW_DISTRIB.values()),
            suppress_warnings=True,
            iterations=10,
            save_dir=SAVEPATHS["savedir"],
            data_file=SAVEPATHS["savefile"],
            visualisation_dir=SAVEPATHS["visualisation"],
            model_id=MODEL_ID,
        )
        # Randomly generate 10 Agents for use in the simulation
        cls._model.generate_agents(
            "TEST",  # id_base
            {  # personality_probs
                "social": 0.4,
                "rational": 0.4,
                "impulsive": 0.2,
            },
            number=10,
        )
        # Generate a graph for "A" with assurance of full agent membership
        cls._model.generate_graphs(
            [HIERARCHY_NAMES[0]],
            cls._model.agents,
            rw_params=[HIERARCHY_RW_DISTRIB[HIERARCHY_NAMES[0]]],
        )
        # Generate graphs for the remaining hierarchies using agent subsetting
        cls._model.generate_graphs(
            HIERARCHY_NAMES[1:],
            cls._model.agents,
            agent_subsetting=True,
            rw_params=list(HIERARCHY_RW_DISTRIB.values())[1:],
        )

    def test_iterate(self) -> None:
        """
        Test function that checks if the ABModel without a worker pool is iterating correctly during runtime.
        """
        self._model.iterate()
        self.assertEqual(
            self._model.current_iteration,
            40,
            "The ABModel is not incrementing its current_iteration attribute correctly in the iterate function without multiprocessing",
        )
        logger_data_saved: bool = os.path.exists(SAVEPATHS["savefile"])
        self.assertTrue(
            logger_data_saved,
            "The ABModel's iterate function without multiprocessing did not properly call the logger's save function at the end of the iterations",
        )

    def test_save_model(self) -> None:
        """
        Test function that checks if the ABModel will save itself appropriately.
        """
        self._model.save_model()
        save_dir_exists: bool = os.path.exists(SAVEPATHS["savedir"])
        self.assertTrue(
            save_dir_exists,
            "The ABModel's save_model function did not create the correct save directory after running without multiprocessing",
        )
        savedir_filelist: list[str] = list(os.walk(self._model.save_dir))[0][2]
        graphml_exists: bool = False
        yaml_exists: bool = False
        unexpected_file_written: bool = False
        for file in savedir_filelist:
            filename, file_extension = os.path.splitext(file)
            match file_extension:
                case ".graphml":
                    self.assertEqual(
                        filename,
                        "graph_base_graph",
                        "The ABModel's save_model function did not name the base graph's graphml file as expected after running without multiprocessing",
                    )
                    graphml_exists = True
                case ".yaml":
                    self.assertStartsWith(
                        filename,
                        "model_",
                        "The ABModel's save_model function did not save the config file with the correct prefix after running without multiprocessing",
                    )
                    yaml_exists = True
                case _:
                    unexpected_file_written = True
        self.assertTrue(
            graphml_exists,
            "The ABModel's save_model function did not create a base graph graphml file after running without multiprocessing",
        )
        self.assertTrue(
            yaml_exists,
            "The ABModel's save_model function did not create a model YAML config file after running without multiprocessing",
        )
        self.assertFalse(
            unexpected_file_written,
            "The ABModel's save_model function wrote an unexpected file to the save directory after running without multiprocessing",
        )

    @classmethod
    @override
    def tearDownClass(cls) -> None:
        del cls._model
