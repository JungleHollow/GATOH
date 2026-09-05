from __future__ import annotations

import os
import unittest as ut
import pickle
import zipfile
from typing import override

from gatoh.groups import Group, GroupSet


class TestGroupSetPersistence(ut.TestCase):
    @classmethod
    @override
    def setUpClass(cls) -> None:
        cls._groupset: GroupSet = GroupSet()
        for i in range(4):
            group: Group = Group(f"{i}")
            _ = cls._groupset.add(group)
        cls._subdir_path: str = "./tests/test_saves/groupset_persistence"

    def test_group_pickle(self) -> None:
        """
        Test that write_group_pickle() and extract_group_pickle() are working correctly as standalones.
        """
        group_to_pickle: Group | None = self._groupset.get_group_by_id("2")
        # Included for typing...
        if group_to_pickle is not None:
            pickle_path: str = self._groupset.write_group_pickle(group_to_pickle, self._subdir_path)
            self.assertIsInstance(
                pickle_path,
                str,
                "Groupset -- write_group_pickle() is not returning a string",
            )
            self.assertEqual(
                pickle_path,
                f"{self._subdir_path}/_group_2.pkl",
                "Groupset -- write_group_pickle() is not writing the group pickle to the correct path",
            )
            pickle_exists: bool = os.path.exists(pickle_path)
            self.assertTrue(
                pickle_exists,
                "Groupset -- write_group_pickle() is not actually writing a file to the specified path",
            )
            # Check that the written pickle corresponds to the input Group using a method that is known to work
            with open(pickle_path, "rb") as group_pickle:
                loaded_pickle: Group = pickle.load(group_pickle)
            self.assertIsInstance(
                loaded_pickle,
                Group,
                "Groupset -- write_group_pickle() did not write a Group object to the pickle file",
            )
            self.assertEqual(
                loaded_pickle.id,
                group_to_pickle.id,
                "Groupset -- write_group_pickle() did not write the correct Group object to the pickle file",
            )
            # Now check that extract_group_pickle() is working
            extracted_pickle: Group = self._groupset.extract_group_pickle("_group_2.pkl", self._subdir_path)
            self.assertIsInstance(
                extracted_pickle,
                Group,
                "Groupset -- extract_group_pickle() is not returning a Group object",
            )
            self.assertEqual(
                extracted_pickle.id,
                group_to_pickle.id,
                "Groupset -- extract_group_pickle() is not extracting the pickled Group object correctly",
            )

    def test_save_groupset(self) -> None:
        """
        Test that save_groupset() is working as intended.
        """
        self._groupset.save_groupset(self._subdir_path)
        zipfile_exists: bool = os.path.exists(f"{self._subdir_path}/_groupset.zip")
        self.assertTrue(
            zipfile_exists,
            "Groupset -- save_groupset() is not writing a _groupset zipfile to the specified directory",
        )
        # Check that the written contents are correct using a method that is known to work
        with zipfile.ZipFile(f"{self._subdir_path}/_groupset.zip", mode="r", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as subdir_zip:
            subdir_zip.extractall(path=f"{self._subdir_path}/_groupset")
        group_pickle_names: list[str] = list(os.walk(f"{self._subdir_path}/_groupset"))[0][2]
        self.assertEqual(
            len(group_pickle_names),
            4,
            "Groupset -- save_groupset() is writing more group pickles than expected to the subdirectory",
        )
        expected_pickle_names: list[str] = [
            "_group_0.pkl",
            "_group_1.pkl",
            "_group_2.pkl",
            "_group_3.pkl",
        ]
        for expected_pickle_name in expected_pickle_names:
            self.assertIn(
                expected_pickle_name,
                group_pickle_names,
                "Groupset -- save_groupset() did not write a Group pickle for one or more groups in the groupset",
            )
            with open(f"{self._subdir_path}/_groupset/{expected_pickle_name}", "rb") as group_pickle:
                loaded_group: Group = pickle.load(group_pickle)
            self.assertIsInstance(
                loaded_group,
                Group,
                "Groupset -- save_groupset() is pickling one or more non-Group objects to the subdirectory",
            )
            self.assertIn(
                loaded_group.id,
                self._groupset.get_group_ids(),
                "Groupset -- save_groupset() is somehow pickling one or more Groups that do not exist in the groupset",
            )

    def test_load_groupset(self) -> None:
        """
        Test that load_groupset() is working as intended.
        """
        # Ensure that the groupset is always saved
        self._groupset.save_groupset(self._subdir_path)
        new_groupset: GroupSet = GroupSet()
        new_groupset.load_groupset(self._subdir_path)
        self.assertIsInstance(
            new_groupset,
            GroupSet,
            "Groupset -- load_groupset() is somehow transforming a groupset into a different object",
        )
        self.assertEqual(
            len(new_groupset),
            len(self._groupset),
            "Groupset -- load_groupset() is loading a groupset of a different length than was saved",
        )
        for group in new_groupset:
            self.assertIn(
                group.id,
                self._groupset.get_group_ids(),
                "Groupset -- load_groupset() is somehow loading Groups that do not exist in the saved groupset",
            )

    @classmethod
    @override
    def tearDownClass(cls) -> None:
        del cls._groupset, cls._subdir_path
