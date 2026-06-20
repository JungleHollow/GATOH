from __future__ import annotations

import os
import unittest as ut
from typing import override

import gatoh.agents.agents as agt
import gatoh.graphs.graphs as gr
import gatoh.model.model as md

MODEL_ID: str = "TEST_SIMULATION"
HIERARCHY_NAMES: list[str] = ["A", "B", "C", "D"]
HIERARCHY_RW_DISTRIB: dict[str, tuple[float, float]] = {
    "A": (0.0, 0.0),
    "B": (0.0, 0.1),
    "C": (0.0, 0.4),
    "D": (0.0, 0.05),
}
SAVEPATHS: dict[str, str] = {
    "savedir": "./gatoh/tests/test_saves/model_simulation",
    "savefile": f"./gatoh/tests/test_saves/model_simulation/{MODEL_ID}_model_variables.csv",
    "visualisation": "./gatoh/tests/test_saves/model_simulation/visualisation_output",
}


class TestModelSimulation(ut.TestCase):
    @classmethod
    @override
    def setUpClass(cls) -> None:
        cls._model: md.ABModel = md.ABModel(
            HIERARCHY_NAMES,
            list(HIERARCHY_RW_DISTRIB.values()),
            iterations=40,
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
        Test function that checks if the ABModel is iterating correctly during runtime.
        """
        self._model.iterate()
        self.assertEqual(
            self._model.current_iteration,
            40,
            "The ABModel is not incrementing its current_iteration attribute correctly in the iterate function",
        )
        logger_data_saved: bool = os.path.exists(SAVEPATHS["savefile"])
        self.assertTrue(
            logger_data_saved,
            "The ABModel's iterate function did not properly call the logger's save function at the end of the iterations",
        )
        self.assertIsNone(
            self._model.fig,
            "The ABModel's iterate function did not appropriately close the visualiser's figure at the end of the iterations",
        )

    @classmethod
    @override
    def tearDownClass(cls) -> None:
        del cls._model
