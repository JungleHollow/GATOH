from __future__ import annotations

import unittest as ut
from typing import override

import gatoh.groups as grp


class TestGroupObjects(ut.TestCase):
    @override
    def setUp(self) -> None:
        self.group_members: list[str] = ["AGENT0001", "AGENT0002", "AGENT0003", "AGENT0004"]
        self.group: grp.Group = grp.Group(
            "GROUP0001",
            10,
            index=4,
            members=self.group_members,
            hierarchy="FooBar",
            aggregate_opinion=-0.44,
            member_benefit_rate=0.75,
            aggregate_susceptibility=0.9,
            predominant_personality="impulsive",
            cohesion="distant",
            radicalisation_rate=0.25,
            aggregate_hierarchy_weighting=0.7,
            silencing_rate=0.0
        )

    def test_update_invalid_args(self) -> None:
        """
        Test that update() with an invalid data type for the arguments will produce the expected error.
        """
        # Test the first argument
        with self.assertRaises(TypeError) as cm:
            update_info = self.group.update("0.2", False)
        # Test the second argument
        with self.assertRaises(TypeError) as cm:
            update_info = self.group.update(0.2, "no")
        # Test both arguments
        with self.assertRaises(TypeError) as cm:
            update_info = self.group.update("0.2", "no")

    def test_update_silencing(self) -> None:
        """
        Test that update() with only silencing ocurring is working as intended.
        """
        update_info: tuple[str, dict[str, bool]] = self.group.update(0.4, False)
        self.assertIsInstance(
            update_info,
            tuple,
            "Group -- a valid call to update is not returning a tuple result (silencing)",
        )
        self.assertIsInstance(
            update_info[0],
            str,
            "Group -- a valid call to update is not returning a hierarchy string as the first value (silencing)",
        )
        self.assertIsInstance(
            update_info[1],
            dict,
            "Group -- a valid call to update is not returning a dictionary as the second value (silencing)",
        )
        self.assertAlmostEqual(
            self.group.aggregate_opinion,
            -0.44,
            2,
            "Group -- a valid call to update is inverting the aggregate opinion when it shouldn't (silencing)",
        )
        self.assertEqual(
            update_info[0],
            "FooBar",
            "Group -- a valid call to update is not reporting the correct group hierarchy (silencing)"
        )
        silenced_count: int = sum(int(flag) for flag in update_info[1].values())
        self.assertEqual(
            silenced_count,
            2,
            "Group -- a valid call to update is not reporting the correct number of silenced members produced by the silencing change (silencing)",
        )
        for group_member in self.group_members:
            self.assertIn(
                group_member,
                update_info[1].keys(),
                "Group -- a valid call to update is not reporting the silencing status for all group members (silencing)",
            )

    def test_update_negation(self) -> None:
        """
        Test that update() with only negation ocurring is working as intended.
        """
        update_info: tuple[str, dict[str, bool]] = self.group.update(0.0, True)
        self.assertIsInstance(
            update_info,
            tuple,
            "Group -- a valid call to update is not returning a tuple result (negation)",
        )
        self.assertIsInstance(
            update_info[0],
            str,
            "Group -- a valid call to update is not returning a hierarchy string as the first value (negation)",
        )
        self.assertIsInstance(
            update_info[1],
            dict,
            "Group -- a valid call to update is not returning a dictionary as the second value (negation)",
        )
        self.assertAlmostEqual(
            self.group.aggregate_opinion,
            0.44,
            2,
            "Group -- a valid call to update is not inverting the aggregate opinion when it should (negation)",
        )
        self.assertEqual(
            update_info[0],
            "FooBar",
            "Group -- a valid call to update is not reporting the correct group hierarchy (negation)",
        )
        silenced_count: int = sum(int(flag) for flag in update_info[1].values())
        self.assertEqual(
            silenced_count,
            0,
            "Group -- a valid call to update is not reporting the correct number of silenced members produced by the silencing change (negation)",
        )
        for group_member in self.group_members:
            self.assertIn(
                group_member,
                update_info[1].keys(),
                "Group -- a valid call to update is not reporting the silencing status for all group members (negation)",
            )

    def test_update_all(self) -> None:
        """
        Test that update() with both negation and silencing ocurring is working as intended.
        """
        update_info: tuple[str, dict[str, bool]] = self.group.update(0.4, True)
        self.assertIsInstance(
            update_info,
            tuple,
            "Group -- a valid call to update is not returning a tuple result (all)",
        )
        self.assertIsInstance(
            update_info[0],
            str,
            "Group -- a valid call to update is not returning a hierarchy string as the first result (all)",
        )
        self.assertIsInstance(
            update_info[1],
            dict,
            "Group -- a valid call to update is not returning a dictionary as the second value (all)",
        )
        self.assertAlmostEqual(
            self.group.aggregate_opinion,
            0.44,
            2,
            "Group -- a valid call to update is not inverting the aggregate opinion when it should (all)",
        )
        self.assertEqual(
            update_info[0],
            "FooBar",
            "Group -- a valid call to update is not reporting the correct group hierarchy (all)",
        )
        silenced_count: int = sum(int(flag) for flag in update_info[1].values())
        self.assertEqual(
            silenced_count,
            2,
            "Group -- a valid call to update is not reporting the correct number of silenced members produced by the silencing change (all)",
        )
        for group_member in self.group_members:
            self.assertIn(
                group_member,
                update_info[1].keys(),
                "Group -- a valid call to update is not reporting the silencing status for all group members (all)",
            )

    def test_opinion_silencing_invalid_op_clim(self) -> None:
        """
        Test that opinion_silencing() with an invalid estimated_opinion_climate will raise the expected error.
        """
        with self.assertRaises(TypeError) as cm:
            silencing_info: tuple[bool, float] = self.group.opinion_silencing("zero")

    def test_opinion_silencing_invalid_thresh(self) -> None:
        """
        Test that opinion_silencing() with an invalid threshold will raise the expected error.
        """
        with self.assertRaises(TypeError) as cm:
            silencing_info: tuple[bool, float] = self.group.opinion_silencing(0.2, silencing_threshold="zero")

    def test_opinion_silencing_true_no_thresh(self) -> None:
        """
        Test that opinion_silencing() with no threshold passed will correctly report that silencing should occur.
        """
        opinion_silencing: tuple[bool, float] = self.group.opinion_silencing(0.8)
        # Worked example:
        #   1. No explicit silencing_threshold was passed, so the group's aggregate social susceptibility of 0.9 is used to calculate the threshold
        #   2. The group's predominant personality is "impulsive", so the absolute difference is calculated as abs(estimated opinion climate - aggregate opinion)
        #   3. Therefore: absolute difference = abs(0.8 - -0.44) = abs(1.24) = 1.24
        #   4. 1.24 is larger than the threshold of (1 - 0.9) = 0.1, so silencing will occur
        self.assertIsInstance(
            opinion_silencing,
            tuple,
            "Group -- a valid call to opinion_silencing is not returning a tuple as the result (no thresh - true)",
        )
        self.assertIsInstance(
            opinion_silencing[0],
            bool,
            "Group -- a valid call to opinion_silencing is not returning a boolean as the first value (no thresh - true)",
        )
        self.assertIsInstance(
            opinion_silencing[1],
            float,
            "Group -- a valid call to opinion_silencing is not returning a float as the second value (no thresh - true)",
        )
        self.assertTrue(
            opinion_silencing[0],
            "Group -- a valid call to opinion_silencing is not correctly reporting that silencing has ocurred (no thresh - true)",
        )
        self.assertAlmostEqual(
            opinion_silencing[1],
            1.24,
            5,
            "Group -- a valid call to opinion_silencing is not calculating and reporting the correct absolute difference value (no thresh - true)",
        )

    def test_opinion_silencing_false_no_thresh(self) -> None:
        """
        Test that opinion_silencing() with no threshold passed will correctly report that silencing should not occur.
        """
        opinion_silencing: tuple[bool, float] = self.group.opinion_silencing(-0.48)
        # Worked example:
        #   1. No explicit silencing threshold was passed, so the group's aggregate social susceptibility of 0.9 is used to calculate the threshold.
        #   2. The group's predominant personality is "impulsive", so the absolute difference is calculated as abs(estimated opinion climate - aggregate opinion)
        #   3. Therefore: absolute difference = abs(-0.48 - -0.44) = abs(-0.04) = 0.04
        #   4. 0.04 is smaller than the threshold of (1.0 - 0.9) = 0.1, so silencing will not occur
        self.assertIsInstance(
            opinion_silencing,
            tuple,
            "Group -- opinion_silencing is not returning a tuple as the result (no thresh - false)",
        )
        self.assertIsInstance(
            opinion_silencing[0],
            bool,
            "Group -- opinion_silencing is not returning a boolean as the first value (no thresh - false)",
        )
        self.assertIsInstance(
            opinion_silencing[1],
            float,
            "Group -- opinion_silencing is not returning a float as the second value (no thresh - false)",
        )
        self.assertFalse(
            opinion_silencing[0],
            "Group -- opinion_silencing is not correctly reporting that no silencing should occur (no thresh - false)",
        )
        self.assertAlmostEqual(
            opinion_silencing[1],
            0.04,
            5,
            "Group -- opinion_silencing is not calculating and reporting the correct absolute difference value (no thresh - false)",
        )

    def test_opinion_silencing_true(self) -> None:
        """
        Test that opinion_silencing() with a threshold will correctly report that silencing should occur.
        """

    @override
    def tearDown(self) -> None:
        del self.group_members, self.group
