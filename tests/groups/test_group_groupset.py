from __future__ import annotations

import unittest as ut
import random as rd

import gatoh.groups as grp


class TestGroupSet(ut.TestCase):
    def test_init(self) -> None:
        """
        Test that an empty initialisation of a GroupSet is returning the expected result.
        """
        empty_groupset: grp.GroupSet = grp.GroupSet()
        self.assertIsInstance(
            empty_groupset,
            grp.GroupSet,
            "Groupset -- the initialisation of a group set is not returning a GroupSet object",
        )
        self.assertEqual(
            empty_groupset.groups,
            [],
            "Groupset -- the initialisation of a group set is not creating an empty list for groups",
        )
        self.assertIsInstance(
            empty_groupset.random,
            rd.Random,
            "Groupset -- the initialisation of a group set is not creating the random generator appropriately",
        )

    def test_add(self) -> None:
        """
        Test that add() is working as intended.
        """
        groupset: grp.GroupSet = grp.GroupSet()
        new_group: grp.Group = grp.Group()
        group_index: int = groupset.add(new_group)
        self.assertIsInstance(
            group_index,
            int,
            "Groupset -- add() is not returning an integer index as a result",
        )
        self.assertEqual(
            group_index,
            0,
            "Groupset -- add() is not reporting the correct index for the newly added group",
        )
        self.assertEqual(
            groupset.groups[group_index],
            new_group,
            "Groupset -- add() is not correctly adding the new Group object to groups",
        )
        self.assertEqual(
            new_group.index,
            group_index,
            "Groupset -- add() is not setting the group's index attribute correctly",
        )

    def test_len(self) -> None:
        """
        Test that the __len__ override for a groupset is working correctly.
        """
        groupset: grp.GroupSet = grp.GroupSet()
        self.assertEqual(
            len(groupset),
            0,
            "Groupset -- __len__ is not reporting 0 when empty",
        )
        group: grp.Group = grp.Group()
        _ = groupset.add(group)
        self.assertEqual(
            len(groupset),
            1,
            "Groupset -- __len__ is not reporting the correct number of agents when not empty",
        )

    def test_in(self) -> None:
        """
        Test that the __in__ override for a groupset is working correctly.
        """
        groupset: grp.GroupSet = grp.GroupSet()
        group: grp.Group = grp.Group("ONE")
        _ = groupset.add(group)
        indirect_call: bool = group in groupset
        direct_call: bool = groupset.__in__(group)
        self.assertEqual(
            indirect_call,
            direct_call,
            "Groupset -- the direct and indirect calls of __in__ are not reporting the same status",
        )
        self.assertTrue(
            indirect_call,
            "Groupset -- __in__ on a valid object is not returning True",
        )
        invalid_group: grp.Group = grp.Group("FOO")
        self.assertFalse(
            invalid_group in groupset,
            "Groupset -- __in__ on an invalid oject is not returning False",
        )

    def test_contains(self) -> None:
        """
        Test that the __contains__ override for a groupset is working as intended.
        """
        groupset: grp.GroupSet = grp.GroupSet()
        group: grp.Group = grp.Group("ONE")
        _ = groupset.add(group)
        contains_call: bool = groupset.__contains__(group)
        self.assertTrue(
            contains_call,
            "Groupset -- __contains__ on a valid group is not returning True",
        )
        invalid_group: grp.Group = grp.Group("FOO")
        self.assertFalse(
            groupset.__contains__(invalid_group),
            "Groupset -- __contains__ on an invalid group is not returning False",
        )

    def test_getitem(self) -> None:
        """
        Test that the __getitem__ override for a groupset is working as intended.
        """
        groupset: grp.GroupSet = grp.GroupSet()
        group: grp.Group = grp.Group("ONE")
        second_group: grp.Group = grp.Group("TWO")
        _ = groupset.add(group)
        _ = groupset.add(second_group)
        get_group: grp.Group | list[grp.Group] = groupset.__getitem__(group.index)
        self.assertIsInstance(
            get_group,
            grp.Group,
            "Groupset -- __getitem__ on a valid, single index is not returning a group object",
        )
        self.assertEqual(
            get_group,
            group,
            "Groupset -- __getitem__ is not returning the expected group object",
        )

    def test_update_indices(self) -> None:
        """
        Test that update_indices() is working as intended.
        """
        groupset: grp.GroupSet = grp.GroupSet()
        group: grp.Group = grp.Group()
        _ = groupset.add(group)
        groupset.groups[0].index = 404
        self.assertEqual(
            groupset.groups[0].index,
            404,
            "Groupset -- direct setting of contained group object indices is not persistent",
        )
        groupset.update_indices()
        self.assertEqual(
            groupset.groups[0].index,
            0,
            "Groupset -- update_indices is not correctly updating the indices of contained group objects",
        )

    def test_discard_simple(self) -> None:
        """
        Test that discard() is working as intended.
        """
        groupset: grp.GroupSet = grp.GroupSet()
        group_one: grp.Group = grp.Group("ONE")
        group_two: grp.Group = grp.Group("TWO")
        _ = groupset.add(group_one)
        discard_one: bool = groupset.discard(group_one)
        self.assertEqual(
            len(groupset),
            0,
            "Groupset -- discard is not correctly removing group objects (simple case)",
        )
        self.assertTrue(
            discard_one,
            "Groupset -- discard is not reporting that a group was removed correctly (simple case)",
        )
        discard_two: bool = groupset.discard(group_two)
        self.assertFalse(
            discard_two,
            "Groupset -- discard is not reporting that a group was not removed correctly (simple case)",
        )

    def test_discard_complex(self) -> None:
        """
        Test that a complex case of discard is working correctly.
        """
        groupset: grp.GroupSet = grp.GroupSet()
        external_groups: list[grp.Group] = []
        for i in range(10):
            group: grp.Group = grp.Group(f"{i}")
            external_groups.append(group)
            _ = groupset.add(group)
        self.assertEqual(
            len(groupset),
            10,
            "Groupset -- adding multiple groups in a loop is not working correctly",
        )
        discard_call: bool = groupset.discard(external_groups[4])
        self.assertEqual(
            len(groupset),
            9,
            "Groupset -- discard is not correctly removing group objects (complex case)",
        )
        self.assertTrue(
            discard_call,
            "Groupset -- discard is not reporting that a group was correctly removed (complex case)",
        )
        self.assertNotIn(
            external_groups[4],
            groupset,
            "Groupset -- discard did not remove the expected group from the group set (complex case)",
        )
        for idx, group in enumerate(groupset):
            self.assertEqual(
                group.index,
                idx,
                "Groupset -- discard did not correctly call update_indices after removing the group",
            )

    def test_group_at_index_invalid(self) -> None:
        """
        Test that group_at_index() with an invalid index raises the expected warning.
        """
        groupset: grp.GroupSet = grp.GroupSet()
        with self.assertWarns(UserWarning) as cm:
            group_return: grp.Group | None = groupset.group_at_index(44)
        self.assertIsNone(
            group_return,
            "Groupset -- group_at_index with an invalid index is not returning None",
        )

    def test_group_at_index(self) -> None:
        """
        Test that group_at_index is working as intended.
        """
        groupset: grp.GroupSet = grp.GroupSet()
        group: grp.Group = grp.Group()
        _ = groupset.add(group)
        group_return: grp.Group | None = groupset.group_at_index(0)
        self.assertIsInstance(
            group_return,
            grp.Group,
            "Groupset -- group_at_index with a valid index is not returning a group object",
        )
        self.assertEqual(
            group,
            group_return,
            "Groupset -- group_at_index with a valid index is not returning the correct group object",
        )

    def test_get_group_by_id_invalid(self) -> None:
        """
        Test that get_group_by_id with an invalid ID raises the expected error.
        """
        groupset: grp.GroupSet = grp.GroupSet()
        with self.assertRaises(KeyError) as cm:
            group_return: grp.Group | None = groupset.get_group_by_id("foo")

    def test_get_group_by_id(self) -> None:
        """
        Test that get_group_by_id is working as intended.
        """
        groupset: grp.GroupSet = grp.GroupSet()
        group: grp.Group = grp.Group("bar")
        _ = groupset.add(group)
        group_return: grp.Group | None = groupset.get_group_by_id("bar")
        self.assertIsInstance(
            group_return,
            grp.Group,
            "Groupset -- get_group_by_id with a valid ID is not returning a group object",
        )
        self.assertEqual(
            group,
            group_return,
            "Groupset -- get_group_by_id with a valid ID is not returning the correct group",
        )

    def test_get_index_invalid(self) -> None:
        """
        Test that get_index() with an invalid group raises the expected error.
        """
        groupset: grp.GroupSet = grp.GroupSet()
        group: grp.Group = grp.Group("foo")
        with self.assertRaises(KeyError) as cm:
            index_return: int = groupset.get_index(group)

    def test_get_index(self) -> None:
        """
        Test that get_index() is working as intended.
        """
        groupset: grp.GroupSet = grp.GroupSet()
        group: grp.Group = grp.Group("foo")
        _ = groupset.add(group)
        index_return: int = groupset.get_index(group)
        self.assertIsInstance(
            index_return,
            int,
            "Groupset -- get_index with a valid group is not returning an integer index",
        )
        self.assertEqual(
            index_return,
            0,
            "Groupset -- get_index with a valid group is not returning the correct group index",
        )

    def test_discard_index_invalid(self) -> None:
        """
        Test that discard_index with an invalid index is working correctly.
        """
        groupset: grp.GroupSet = grp.GroupSet()
        for i in range(4):
            group: grp.Group = grp.Group(f"{i}")
            _ = groupset.add(group)
        index_discarded: bool = groupset.discard_index(4444)
        self.assertIsInstance(
            index_discarded,
            bool,
            "Groupset -- discard_index with an invalid index is not returning a boolean",
        )
        self.assertFalse(
            index_discarded,
            "Groupset -- discard_index with an invalid index is reporting that an index was removed",
        )
        self.assertEqual(
            len(groupset),
            4,
            "Groupset -- discard_index with an invalid index is removing one or more groups from the groupset",
        )

    def test_discard_index(self) -> None:
        """
        Test that discard_index with a valid index is working correctly.
        """
        groupset: grp.GroupSet = grp.GroupSet()
        external_groups: list[grp.Group] = []
        for i in range(4):
            group: grp.Group = grp.Group(f"{i}")
            external_groups.append(group)
            _ = groupset.add(group)
        index_discarded: bool = groupset.discard_index(2)
        self.assertIsInstance(
            index_discarded,
            bool,
            "Groupset -- discard_index with a valid index is not returning a boolean",
        )
        self.assertTrue(
            index_discarded,
            "Groupset -- discard_index with a valid index is not reporting that an index was removed",
        )
        self.assertEqual(
            len(groupset),
            3,
            "Groupset -- discard_index with a valid index is not removing a group object",
        )
        self.assertNotIn(
            external_groups[2],
            groupset,
            "Groupset -- discard_index with a valid index is not removing the correct group object",
        )
        for idx, group in enumerate(groupset):
            self.assertEqual(
                group.index,
                idx,
                "Groupset -- discard_index with a valid index is not calling update_indices correctly",
            )

    def test_remove_invalid(self) -> None:
        """
        Test that remove() with an invalid group raises the expected error.
        """
        groupset: grp.GroupSet = grp.GroupSet()
        for i in range(4):
            group: grp.Group = grp.Group(f"{i}")
            _ = groupset.add(group)
        invalid_group: grp.Group = grp.Group("foo")
        with self.assertRaises(KeyError) as cm:
            group_removed: bool = groupset.remove(invalid_group)
        self.assertEqual(
            len(groupset),
            4,
            "Groupset -- remove() with an invalid group is removing one or more groups despite raising an error",
        )

    def test_remove(self) -> None:
        """
        Test that remove() is working as intended.
        """
        groupset: grp.GroupSet = grp.GroupSet()
        external_groups: list[grp.Group] = []
        for i in range(4):
            group: grp.Group = grp.Group(f"{i}")
            external_groups.append(group)
            _ = groupset.add(group)
        group_removed: bool = groupset.remove(external_groups[2])
        self.assertIsInstance(
            group_removed,
            bool,
            "Groupset -- a valid call to remove() is not returning a boolean",
        )
        self.assertTrue(
            group_removed,
            "Groupset -- a valid call to remove() is not reporting that a group was removed",
        )
        self.assertEqual(
            len(groupset),
            3,
            "Groupset -- a valid call to remove() is not removing a group object from the groupset",
        )
        self.assertNotIn(
            external_groups[2],
            groupset,
            "Groupset -- a valid call to remove() is not removing the correct group object",
        )
        for idx, group in enumerate(groupset):
            self.assertEqual(
                group.index,
                idx,
                "Groupset -- a valid call to remove() is not calling update_indices correctly",
            )

    def test_remove_index_invalid(self) -> None:
        """
        Test that remove_index() with an invalid index will raise the expected error.
        """
        groupset: grp.GroupSet = grp.GroupSet()
        for i in range(4):
            group: grp.Group = grp.Group(f"{i}")
            _ = groupset.add(group)
        with self.assertRaises(IndexError) as cm:
            index_removed: bool = groupset.remove_index(4444)
        self.assertEqual(
            len(groupset),
            4,
            "Groupset -- remove_index with an invalid index is removing one or more groups despite raising an error",
        )

    def test_remove_index(self) -> None:
        """
        Test that remove_index is working as intended.
        """
        groupset: grp.GroupSet = grp.GroupSet()
        external_groups: list[grp.Group] = []
        for i in range(4):
            group: grp.Group = grp.Group(f"{i}")
            external_groups.append(group)
            _ = groupset.add(group)
        index_removed: bool = groupset.remove_index(2)
        self.assertIsInstance(
            index_removed,
            bool,
            "Groupset -- a valid call to remove_index is not returning a boolean",
        )
        self.assertTrue(
            index_removed,
            "Groupset -- a valid call to remove_index is not reporting that a group was removed",
        )
        self.assertEqual(
            len(groupset),
            3,
            "Groupset -- a valid call to remove_index is not actually removing a group object",
        )
        self.assertNotIn(
            external_groups[2],
            groupset,
            "Groupset -- a valid call to remove_index is not removing the correct group object",
        )
        for idx, group in enumerate(groupset):
            self.assertEqual(
                group.index,
                idx,
                "Groupset -- a valid call to remove_index is not calling update_indices correctly",
            )

    def test_sample_invalid(self) -> None:
        """
        Test that sample() with n larger than the groupset will raise the expected error.
        """
        groupset: grp.GroupSet = grp.GroupSet()
        for i in range(4):
            group: grp.Group = grp.Group(f"{i}")
            _ = groupset.add(group)
        with self.assertRaises(ValueError) as cm:
            group_sample: list[grp.Group] = groupset.sample(40)

    def test_sample(self) -> None:
        """
        Test that sample() is working as intended.
        """
        groupset: grp.GroupSet = grp.GroupSet()
        external_groups: list[grp.Group] = []
        for i in range(10):
            group: grp.Group = grp.Group(f"{i}")
            external_groups.append(group)
            _ = groupset.add(group)
        group_sample: list[grp.Group] = groupset.sample(2)
        self.assertIsInstance(
            group_sample,
            list,
            "Groupset -- a valid call to sample() is not returning a list",
        )
        self.assertEqual(
            len(group_sample),
            2,
            "Groupset -- a valid call to sample() is not returning a sample of the appropriate size",
        )
        for group in group_sample:
            self.assertIsInstance(
                group,
                grp.Group,
                "Groupset -- a valid call to sample() is returning a sample containing a non-Group object",
            )
            self.assertIn(
                group.id,
                groupset.get_group_ids(),
                "Groupset -- a valid call to sample() is somehow returning a group which is not contained in the groupset",
            )

    def test_getstate(self) -> None:
        """
        Test that the __getstate__ override is producing the expected result.
        """
        groupset: grp.GroupSet = grp.GroupSet()
        for i in range(4):
            group: grp.Group = grp.Group(f"{i}")
            _ = groupset.add(group)
        groupset_state = groupset.__getstate__()
        self.assertIsInstance(
            groupset_state,
            dict,
            "Groupset -- __getstate__ is not returning a dictionary",
        )
        self.assertIn(
            "groups",
            groupset_state.keys(),
            "Groupset -- the __getstate__ dict does not contain a 'groups' key",
        )
        self.assertIn(
            "random",
            groupset_state.keys(),
            "Groupset -- the __getstate__ dict does not contain a 'random' key",
        )
        self.assertIsInstance(
            groupset_state["groups"],
            list,
            "Groupset -- the 'groups' value from __getstate__ is not the same object type as 'groups' in the groupset",
        )
        self.assertEqual(
            groupset_state["groups"],
            groupset.groups,
            "Groupset -- the groups from __getstate__ are not identical to the groups in the groupset",
        )
        self.assertIsInstance(
            groupset_state["random"],
            rd.Random,
            "Groupset -- the 'random' value from __getstate__ is not the same object type as 'random' in the groupset",
        )
