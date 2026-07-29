from __future__ import annotations

import os
import unittest as ut
import pickle
import zipfile
from typing import override

from gatoh.agents import Agent, AgentSet


class TestAgentSetPersistence(ut.TestCase):
    @classmethod
    @override
    def setUpClass(cls) -> None:
        cls._agentset: AgentSet = AgentSet()
        for i in range(4):
            new_agent: Agent = Agent(f"{i}")
            _ = cls._agentset.add(new_agent)
        cls._subdir_path: str = "./tests/test_saves/agentset_persistence"

    def test_agent_pickle(self) -> None:
        """
        Test that write_agent_pickle() and extract_agent_pickle() are working correctly as standalones.
        """
        agent_to_pickle: Agent | None = self._agentset.get_agent_by_id("2")
        # Included for typing...
        if agent_to_pickle is not None:
            pickle_path: str = self._agentset.write_agent_pickle(agent_to_pickle, self._subdir_path)
            self.assertIsInstance(
                pickle_path,
                str,
                "AgentSet -- write_agent_pickle() is not returning a string",
            )
            self.assertEqual(
                pickle_path,
                f"{self._subdir_path}/_agent_2.pkl",
                "AgentSet -- write_agent_pickle() is not writing the agent pickle to the correct path",
            )
            pickle_exists: bool = os.path.exists(pickle_path)
            self.assertTrue(
                pickle_exists,
                "AgentSet -- write_agent_pickle() is not actually writing a file to the specified path",
            )
            # Check that the written pickle corresponds to the input Agent using a method that is known to work
            with open(pickle_path, "rb") as agent_pickle:
                loaded_pickle: Agent = pickle.load(agent_pickle)
            self.assertIsInstance(
                loaded_pickle,
                Agent,
                "AgentSet -- write_agent_pickle() did not write an Agent object to the pickle file",
            )
            self.assertEqual(
                loaded_pickle.id,
                agent_to_pickle.id,
                "AgentSet -- write_agent_pickle() did not write the correct Agent object to the pickle file",
            )
            # Now check that extract_agent_pickle() is working
            extracted_pickle: Agent = self._agentset.extract_agent_pickle("_agent_2.pkl", self._subdir_path)
            self.assertIsInstance(
                extracted_pickle,
                Agent,
                "AgentSet -- extract_agent_pickle() is not returning an Agent object",
            )
            self.assertEqual(
                extracted_pickle.id,
                agent_to_pickle.id,
                "AgentSet -- extract_agent_pickle() is not extracting the pickled Agent object correctly",
            )

    def test_save_agentset(self) -> None:
        """
        Test that save_agentset() is working as intended.
        """
        self._agentset.save_agentset(self._subdir_path)
        zipfile_exists: bool = os.path.exists(f"{self._subdir_path}/_agentset.zip")
        self.assertTrue(
            zipfile_exists,
            "AgentSet -- save_agentset() is not writing an _agentset zipfile to the specified directory",
        )
        # Check that the written contents are correct using a method that is known to work
        with zipfile.ZipFile(f"{self._subdir_path}/_agentset.zip", mode="r", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as subdir_zip:
            subdir_zip.extractall(path=f"{self._subdir_path}/_agentset")
        agent_pickle_names: list[str] = list(os.walk(f"{self._subdir_path}/_agentset"))[0][2]
        self.assertEqual(
            len(agent_pickle_names),
            4,
            "AgentSet -- save_agentset() is writing more Agent pickles than expected to the subdirectory",
        )
        expected_pickle_names: list[str] = [
            "_agent_0.pkl",
            "_agent_1.pkl",
            "_agent_2.pkl",
            "_agent_3.pkl",
        ]
        for expected_pickle_name in expected_pickle_names:
            self.assertIn(
                expected_pickle_name,
                agent_pickle_names,
                "AgentSet -- save_agentset() did not write an Agent pickle for one or more agents in the AgentSet",
            )
            with open(f"{self._subdir_path}/_agentset/{expected_pickle_name}", "rb") as agent_pickle:
                loaded_agent: Agent = pickle.load(agent_pickle)
            self.assertIsInstance(
                loaded_agent,
                Agent,
                "AgentSet -- save_agentset() is pickling one or more non-Agent objects to the subdirectory",
            )
            self.assertIn(
                loaded_agent.id,
                self._agentset.get_agent_ids(),
                "AgentSet -- save_agentset() is somehow pickling one or more Agents that do not exist in the AgentSet",
            )

    def test_load_agentset(self) -> None:
        """
        Test that load_agentset() is working as intended.
        """
        # Ensure that the agentset is always saved
        self._agentset.save_agentset(self._subdir_path)
        new_agentset: AgentSet = AgentSet()
        new_agentset.load_agentset(self._subdir_path)
        self.assertIsInstance(
            new_agentset,
            AgentSet,
            "AgentSet -- load_agentset() is somehow transforming an AgentSet into a different object",
        )
        self.assertEqual(
            len(new_agentset),
            len(self._agentset),
            "AgentSet -- load_agentset() is loading an AgentSet of a different length than was saved",
        )
        for agent in new_agentset:
            self.assertIn(
                agent.id,
                self._agentset.get_agent_ids(),
                "AgentSet -- load_agentset() is somehow loading Agents that do not exist in the saved AgentSet",
            )

    @classmethod
    @override
    def tearDownClass(cls) -> None:
        del cls._agentset
