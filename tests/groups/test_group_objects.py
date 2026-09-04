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
        _ = self.group.change_radicalisation_rate(-0.25)
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
        _ = self.group.change_radicalisation_rate(-0.25)
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
        _ = self.group.change_radicalisation_rate(-0.25)
        opinion_silencing: tuple[bool, float] = self.group.opinion_silencing(-0.48, silencing_threshold=0.01)
        # Worked example:
        #   1. An explicit threshold of 0.01 was passed
        #   2. The group's predominant personality is "impulsive", so the absolute difference is calculated as abs(estimated opinion climate - aggregate opinion)
        #   3. Therefore: absolute_difference = abs(-0.48 - -0.44) = abs(-0.04) = 0.04
        #   4. 0.04 is greater than the threshold of 0.01, so silencing will occur
        self.assertIsInstance(
            opinion_silencing,
            tuple,
            "Group -- opinion_silencing is not returning a tuple as the result (thresh - true)",
        )
        self.assertIsInstance(
            opinion_silencing[0],
            bool,
            "Group -- opinion_silencing is not returning a boolean as the first value (thresh - true)",
        )
        self.assertIsInstance(
            opinion_silencing[1],
            float,
            "Group -- opinion_silencing is not returning a float as the second value (thresh - true)",
        )
        self.assertTrue(
            opinion_silencing[0],
            "Group -- opinion_silencing is not correctly reporting that silencing should occur (thresh - true)",
        )
        self.assertAlmostEqual(
            opinion_silencing[1],
            0.04,
            5,
            "Group -- opinion_silencing is not calculating and reporting the correct absolute difference value (thresh - true)",
        )

    def test_opinion_silencing_false(self) -> None:
        """
        Test that opinion_silencing() with a threshold will correctly report that silencing should not occur.
        """
        _ = self.group.change_radicalisation_rate(-0.25)
        opinion_silencing: tuple[bool, float] = self.group.opinion_silencing(0.9, silencing_threshold=1.99)
        # Worked example:
        #   1. An explicit silencing threshold of 1.99 was passed
        #   2. The group's predominant personality is "impulsive", so the absolute difference is calculated as abs(estimated opinion climate - aggregate opinion)
        #   3. Therefore: absolute difference = abs(0.9 - -0.44) = abs(1.34) = 1.34
        #   4. 1.34 is smaller than the threshold of 1.99, so silencing will not occur
        self.assertIsInstance(
            opinion_silencing,
            tuple,
            "Group -- opinion_silencing is not returning a tuple as the result (thresh - false)",
        )
        self.assertIsInstance(
            opinion_silencing[0],
            bool,
            "Group -- opinion_silencing is not returning a boolean as the first value (thresh - false)",
        )
        self.assertIsInstance(
            opinion_silencing[1],
            float,
            "Group -- opinion_silencing is not returning a float as the second value (thresh - false)",
        )
        self.assertFalse(
            opinion_silencing[0],
            "Group -- opinion_silencing is not correctly reporting that silencing should not occur (thresh - false)",
        )
        self.assertAlmostEqual(
            opinion_silencing[1],
            1.34,
            5,
            "Group -- opinion_silencing is not calculating and reporting the correct absolute difference value (thresh - false)",
        )

    def test_opinion_silencing_radicalised(self) -> None:
        """
        Test that opinion_silencing() on a radicalised group will return the expected result.
        """
        opinion_silencing: tuple[bool, float] = self.group.opinion_silencing(1.0, silencing_threshold=0.01)
        self.assertIsInstance(
            opinion_silencing,
            tuple,
            "Group -- opinion_silencing is not returning a tuple as the result (radicalised)",
        )
        self.assertIsInstance(
            opinion_silencing[0],
            bool,
            "Group -- opinion_silencing is not returning a boolean as the first value (radicalised)",
        )
        self.assertIsInstance(
            opinion_silencing[1],
            float,
            "Group -- opinion_silencing is not returning a float as the second value (radicalised)",
        )
        self.assertFalse(
            opinion_silencing[0],
            "Group -- opinion_silencing is not correctly reporting that silencing should not occur (radicalised)",
        )
        self.assertAlmostEqual(
            opinion_silencing[1],
            0.0,
            5,
            "Group -- opinion_silencing is not reporting an absolute difference of 0.0 (radicalised)",
        )

    def test_opinion_negation_invalid_diff(self) -> None:
        """
        Test that opinion_negation() with an absolute difference of invalid data type will raise the expected error.
        """
        with self.assertRaises(TypeError) as cm:
            negation: bool = self.group.opinion_negation("thirteen", 0.12)

    def test_opinion_negation_invalid_thresh(self) -> None:
        """
        Test that opinion_negation() with a threshold of invalid data type will raise the expected error.
        """
        with self.assertRaises(TypeError) as cm:
            negation: bool = self.group.opinion_negation(0.13, "twelve")

    def test_opinion_negation_false(self) -> None:
        """
        Test that opinion_negation() will correctly report that negation should not occur.
        """
        _ = self.group.change_radicalisation_rate(-0.25)
        negation: bool = self.group.opinion_negation(0.03, 1.99)
        # Worked example:
        #   1. negation_strength is initialised as the reported absolute difference 0.03
        #   2. Predominant personality is "impulsive", so negation_strength is modified by /= ((1 - aggregate susceptibility) * aggregate hierarchy weighting)
        #   3. Therefore, negation_strength = 0.03 / ((1 - 0.9) * 0.2) = 0.03 / 0.02 = 1.5
        #   4. The negation_strength of 1.5 is not above the threshold of 1.99, so negation does not occur
        self.assertIsInstance(
            negation,
            bool,
            "Group -- opinion_negation is not returning a boolean result (negation=false)",
        )
        self.assertFalse(
            negation,
            "Group -- opinion_negation is not correctly reporting that negation should not occur",
        )

    def test_opinion_negation_true(self) -> None:
        """
        Test that opinion_negation() will correctly report that negation should occur.
        """
        _ = self.group.change_radicalisation_rate(-0.25)
        negation: bool = self.group.opinion_negation(6.9, 13.12)
        # Worked example:
        #   1. negation_strength is initialised as the reported absolute difference 6.9
        #   2. Predominant personality is "impulsive", so negation_strength is modified by /= ((1 - aggregate susceptibility) * aggregate hierarchy weighitng)
        #   3. Therefore, negation_strength = 6.9 / ((1 - 0.9) * 0.2) = 6.9 / 0.02 = 345
        #   4. The negation_strength of 345 is above the threshold of 13.12, so negation does occur
        self.assertIsInstance(
            negation,
            bool,
            "Group -- opinion_negation is not returning a boolean result (negation=true)",
        )
        self.assertTrue(
            negation,
            "Group -- opinion_negation is not correctly reporting that negation should occur",
        )

    def test_opinion_negation_radicalised(self) -> None:
        """
        Test that opinion_negation() on a radicalised group will correctly report that negation does not occur.
        """
        negation: bool = self.group.opinion_negation(6.9, 13.12)
        self.assertIsInstance(
            negation,
            bool,
            "Group -- opinion_negation is not returning a boolean result (radicalised)",
        )
        self.assertFalse(
            negation,
            "Group -- opinion_negation on a radicalised group is not reporting that negation will not occur",
        )

    def test_evolve_hierarchy_invalid_input(self) -> None:
        """
        Test that evolve_hierarchy() with an invalid rw_distribution data type will produce the expected error.
        """
        with self.assertRaises(TypeError) as cm:
            evolve_info: tuple[str, float] = self.group.evolve_hierarchy([0.0, 0.1])

    def test_evolve_hierarchy(self) -> None:
        """
        Test that evolve_hierarchy() is working as intended.
        """
        evolve_info: tuple[str, float] = self.group.evolve_hierarchy((0.0, 0.5))
        self.assertIsInstance(
            evolve_info,
            tuple,
            "Group -- a valid call to evolve_hierarchy is not returning a tuple value",
        )
        self.assertIsInstance(
            evolve_info[0],
            str,
            "Group -- a valid call to evolve_hierarchy is not returning a string as the first value",
        )
        self.assertEqual(
            evolve_info[0],
            "FooBar",
            "Group -- a valid call to evolve_hierarchy is not reporting the group's hierarchy correctly"
        )
        self.assertIsInstance(
            evolve_info[1],
            float,
            "Group -- a valid call to evolve_hierarchy is not returning a float as the second value",
        )
        self.assertIsInstance(
            self.group.aggregate_hierarchy_weighting,
            float,
            "Group -- a valid call to evolve_hierarchy is causing a non-float value to be set for the aggregate hierarchy weighting",
        )
        self.assertTrue(
            (self.group.aggregate_hierarchy_weighting >= -grp.SOCIAL_WEIGHTINGS_MAX) and (self.group.aggregate_hierarchy_weighting <= grp.SOCIAL_WEIGHTINGS_MAX),
            "Group -- a valid call to evolve_hierarchy is not producing an aggregate weighting value in the valid range",
        )
        self.assertNotAlmostEqual(
            self.group.aggregate_hierarchy_weighting,
            0.7,
            5,
            "Group -- a valid call to evolve_hierarchy is not changing the aggregate hierarchy value of the group correctly",
        )

    def test_stochastic_opinion_invalid(self) -> None:
        """
        Test that stochastic_opinion() with invalid data types will raise the expected errors.
        """
        with self.assertRaises(TypeError) as cm:
            opinion_info = self.group.stochastic_opinion([0.0, 0.1])
        with self.assertRaises(TypeError) as cm:
            opinion_info = self.group.stochastic_opinion(("zero", 0.1))
        with self.assertRaises(TypeError) as cm:
            opinion_info = self.group.stochastic_opinion((0.0, "point_one"))

    def test_stochastic_opinion(self) -> None:
        """
        Test that stochastic_opinion() is working as intended.
        """
        opinion_info: tuple[str, float] = self.group.stochastic_opinion((0.0, 0.5))
        self.assertIsInstance(
            opinion_info,
            tuple,
            "Group -- stochastic_opinion is not returning a tuple result",
        )
        self.assertIsInstance(
            opinion_info[0],
            str,
            "Group -- stochastic_opinion is not returning a string as the first value",
        )
        self.assertEqual(
            opinion_info[0],
            "FooBar",
            "Group -- stochastic_opinion is not correctly reporting the group's hierarchy"
        )
        self.assertIsInstance(
            opinion_info[1],
            float,
            "Group -- stochastic_opinion is not returning a float as the second value",
        )
        self.assertIsInstance(
            self.group.aggregate_opinion,
            float,
            "Group -- a valid call to stochastic_opinon is causing a non-float value to be set for the aggregate opinion",
        )
        self.assertTrue(
            (self.group.aggregate_opinion >= -grp.OPINION_MAX) and (self.group.aggregate_opinion <= grp.OPINION_MAX),
            "Group -- a valid call to stochastic_opinion is not producing an aggregate opinion value in the valid range",
        )
        self.assertNotAlmostEqual(
            self.group.aggregate_opinion,
            -0.44,
            5,
            "Group -- a valid call to stochastic_opinion is not changing the group's aggregate opinion value correctly",
        )

    def test_stochastic_personality_change_invalid_prob(self) -> None:
        """
        Test that stochastic_personality_change with an invalid probability value will produce the expected error.
        """
        personalities_p_invalid: dict[str, float] = {"neutral": "half", "rational": "half"}
        with self.assertRaises(TypeError) as cm:
            member_personalities: dict[str, str] = self.group.stochastic_personality_change(personalities_p_invalid)

    def test_stochastic_personality_change_invalid_personality(self) -> None:
        """
        Test that stochastic_personality_change with an invalid personality type will produce the expected error.
        """
        personalities_p_invalid: dict[str, float] = {"neutral": 0.9, "rare": 0.1}
        with self.assertRaises(KeyError) as cm:
            member_personalities: dict[str, str] = self.group.stochastic_personality_change(personalities_p_invalid)

    def test_stochastic_personality_change(self) -> None:
        """
        Test that stochastic_personality_change is working as intended.
        """
        member_personalities: dict[str, str] = self.group.stochastic_personality_change()
        self.assertIsInstance(
            member_personalities,
            dict,
            "Group -- a valid call to stochastic_personality_change is not returning a dictionary value (no p)",
        )
        self.assertEqual(
            len(member_personalities.keys()),
            len(self.group_members),
            "Group -- a valid call to stochastic_personality_change is not reporting a personality for the correct number of members (no p)",
        )
        personality_counts: dict[str, int] = {personality: 0 for personality in grp.PERSONALITIES}
        for member, personality in member_personalities:
            self.assertIn(
                member,
                self.group_members,
                "Group -- a valid call to stochastic_personality_change is reporting a personality type for one or more non-existent members (no p)",
            )
            self.assertIn(
                personality,
                grp.PERSONALITIES,
                "Group -- a valid call to stochastic_personality_change is assigning an unsupported personality type to one or more agents (no p)",
            )
            personality_counts[personality] += 1
        self.assertEqual(
            max(personality_counts, key=lambda x: personality_counts[x]),
            self.group.predominant_personality,
            "Group -- a valid call to stochastic_personality_change is not assigning personalities to members so that the predominant personality is such (no p)",
        )

    def test_stochastic_personality_change_probs(self) -> None:
        """
        Test that stochastic_personality_change with explicit probabilities is working as intended.
        """
        member_personalities: dict[str, str] = self.group.stochastic_personality_change(personality_probs={"neutral": 0.5, "rational": 0.5})
        self.assertIsInstance(
            member_personalities,
            dict,
            "Group -- a valid call to stochastic_personality_change is not returning a dictionary value (p)",
        )
        self.assertEqual(
            len(member_personalities.keys()),
            len(self.group_members),
            "Group -- a valid call to stochastic_personality_change is not reporting a personality for the correct number of members (p)",
        )
        personality_counts: dict[str, int] = {personality: 0 for personality in grp.PERSONALITIES}
        for member, personality in member_personalities:
            self.assertIn(
                member,
                self.group_members,
                "Group -- a valid call to stochastic_personality_change is reporting a personality type for one or more non-existent members (p)",
            )
            self.assertIn(
                personality,
                grp.PERSONALITIES,
                "Group -- a valid call to stochastic_personality_change is assigning an unsupported personality type to one or more agents (p)",
            )
            personality_counts[personality] += 1
        self.assertEqual(
            max(personality_counts, key=lambda x: personality_counts[x]),
            self.group.predominant_personality,
            "Group -- a valid call to stochastic_personality_change is not assigning personalities to members so that the predominant personality is such (p)",
        )
        self.assertIn(
            self.group.predominant_personality,
            ["neutral", "rational"],
            "Group -- a valid call to stochastic_personality_change is assigning an impossible predominant personality (p)",
        )

    def test_stochastic_benefit_change(self) -> None:
        """
        Test that stochastic_benefit_change is working as intended.
        """
        member_benefits: dict[str, bool] = self.group.stochastic_benefit_change()
        self.assertIsInstance(
            member_benefits,
            dict,
            "Group -- a valid call to stochastic_benefit_change is not returning a dictionary value",
        )
        for member in self.group_members:
            self.assertIn(
                member,
                member_benefits.keys(),
                "Group -- a valid call to stochastic_benefit_change is not reporting a benefit status for one or more members",
            )
            self.assertIsInstance(
                member_benefits[member],
                bool,
                "Group -- a valid call to stochastic_benefit_change is not assigning a boolean flag as the value for one or more members",
            )

    def test_stochastic_radicalisation_change(self) -> None:
        """
        Test that stochastic_radicalisation_change is working as intended.
        """
        member_radicalisations: dict[str, bool] = self.group.stochastic_radicalisation_change()
        self.assertIsInstance(
            member_radicalisations,
            dict,
            "Group -- a valid call to stochastic_radicalisation_change is not returning a dictionary value",
        )
        for member in self.group_members:
            self.assertIn(
                member,
                member_radicalisations.keys(),
                "Group -- a valid call to stochastic_radicalisation_change is not reporting a radicalisation status for one or more members",
            )
            self.assertIsInstance(
                member_radicalisations[member],
                bool,
                "Group -- a valid call to stochastic_radicalisation_change is not assigning a boolean flag as the value for one or more members",
            )

    def test_stochastic_silencing_change(self) -> None:
        """
        Test that stochastic_silencing_change is working as intended.
        """
        members_silenced: dict[str, bool] = self.group.stochastic_silencing_change()
        self.assertIsInstance(
            members_silenced,
            dict,
            "Group -- a valid call to stochastic_silencing_change is not returning a dictionary value",
        )
        for member in self.group_members:
            self.assertIn(
                member,
                members_silenced.keys(),
                "Group -- a valid call to stochastic_silencing_change is not reporting a silencing status for one or more members",
            )
            self.assertIsInstance(
                members_silenced[member],
                bool,
                "Group -- a valid call to stochastic_silencing_change is not assigning a boolean flag as the value for one or more members",
            )

    @override
    def tearDown(self) -> None:
        del self.group_members, self.group
