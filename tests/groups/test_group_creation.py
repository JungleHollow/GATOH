from __future__ import annotations

import unittest as ut

from gatoh.agents import Agent
import gatoh.groups as grp


class TestGroupCreation(ut.TestCase):
    def test_draw_cohesion(self) -> None:
        """
        Test that the draw_cohesion utility function is working as intended.
        """
        drawn_cohesion: str = grp.draw_cohesion()
        self.assertIsInstance(
            drawn_cohesion,
            str,
            "Group -- draw_cohesion is not returning a string result",
        )
        self.assertIn(
            drawn_cohesion,
            grp.COHESIONS,
            "Group -- draw_cohesion is not drawing a valid cohesion type",
        )

    def test_generate_group_invalid(self) -> None:
        """
        Test that generate_group() with an invalid required parameter produces the expected error.
        """
        empty_group: grp.Group = grp.Group()
        with self.assertRaises(TypeError, msg="One or more of the required parameters 'id', 'index', 'hierarchy', or 'members' are not of the appropriate data type") as cm:
            _ = empty_group.generate_group(
                "FooGroup",
                1,
                False,
                [Agent("Foo"), Agent("Bar")],
            )

    def test_generate_group_invalid_cohesion(self) -> None:
        """
        Test that generate_group() with an unsupported cohesion type produces the expected error.
        """
        empty_group: grp.Group = grp.Group()
        with self.assertRaises(ValueError, msg="The specified cohesion type is not supported") as cm:
            _ = empty_group.generate_group(
                "FooGroup",
                1,
                "Foobar",
                [],
                cohesion="friendly",
            )

    def test_generate_group(self) -> None:
        """
        Test that generate_group() is working as intended.
        """
        empty_group: grp.Group = grp.Group()
        empty_group = empty_group.generate_group(
            "FooGroup",
            1,
            "Foobar",
            [
                Agent(
                    "Foo",
                    0.13,
                    ("social", 0.22),
                    {"Foobar": 0.32},
                    False
                ),
                Agent(
                    "Bar",
                    0.12,
                    ("social", 0.1),
                    {"Foobar": -0.8},
                    True,
                    radicalised=True,
                ),
            ],
            cohesion="close",
            max_size=10,
        )
        self.assertHasAttr(
            empty_group,
            "id",
            "Group -- a valid call to generate_group is not initialising the id attribute",
        )
        self.assertEqual(
            empty_group.id,
            "FooGroup",
            "Group -- a valid call to generate_group is not setting the id to the correct value",
        )
        self.assertHasAttr(
            empty_group,
            "index",
            "Group -- a valid call to generate_group is not initialising the index attribute",
        )
        self.assertEqual(
            empty_group.index,
            1,
            "Group -- a valid call to generate_group is not setting the index to the correct value",
        )
        self.assertEqual(
            empty_group.cohesion,
            "close",
            "Group -- a valid call to generate_group is not setting the cohesion to the correct value",
        )
        self.assertEqual(
            empty_group.max_size,
            10,
            "Group -- a valid call to generate_group is not setting the max_size to the correct value",
        )
        self.assertHasAttr(
            empty_group,
            "hierarchy",
            "Group -- a valid call to generate_group is not initialising the hierarchy attribute",
        )
        self.assertEqual(
            empty_group.hierarchy,
            "Foobar",
            "Group -- a valid call to generate_group is not setting the hierarchy to the correct value",
        )
        self.assertEqual(
            len(empty_group.members),
            2,
            "Group -- a valid call to generate_group is not adding the correct number of members to the group",
        )
        for member in ["Foo", "Bar"]:
            self.assertIn(
                member,
                empty_group.members,
                "Group -- one or more input members are not being added to members in a valid call to generate_group",
            )
        self.assertHasAttr(
            empty_group,
            "aggregate_opinion",
            "Group -- a valid call to generate_group is not initialising the aggregate_opinion attribute",
        )
        self.assertAlmostEqual(
            empty_group.aggregate_opinion,
            0.125,
            2,
            "Group -- a valid call to generate_group is not calculating the correct aggregate_opinion value",
        )
        self.assertEqual(
            empty_group.previous_opinion,
            0.0,
            "Group -- a valid call to generate_group is changing the previous_opinion from the default value of 0.0",
        )
        self.assertHasAttr(
            empty_group,
            "member_benefit_rate",
            "Group -- a valid call to generate_group is not initialising the member_benefit_rate attribute",
        )
        self.assertAlmostEqual(
            empty_group.member_benefit_rate,
            0.5,
            1,
            "Group -- a valid call to generate_group is not calculating the correct member_benefit_rate value",
        )
        self.assertHasAttr(
            empty_group,
            "aggregate_susceptibility",
            "Group -- a valid call to generate_group is not initialising the aggregate_susceptibility attribute",
        )
        self.assertAlmostEqual(
            empty_group.aggregate_susceptibility,
            0.16,
            2,
            "Group -- a valid call to generate_group is not calculating the correct aggregate_susceptibility value",
        )
        self.assertEqual(
            empty_group.cohesion,
            "close",
            "Group -- a valid call to generate_group is not setting the correct cohesion type for the group",
        )
        self.assertEqual(
            empty_group.predominant_personality,
            "social",
            "Group -- a valid call to generate_group is not determining the correct predominant personality type",
        )
        self.assertHasAttr(
            empty_group,
            "radicalisation_rate",
            "Group -- a valid call to generate_group is not initialising the radicalisation_rate attribute",
        )
        self.assertAlmostEqual(
            empty_group.radicalisation_rate,
            0.5,
            1,
            "Group -- a valid call to generate_group is not calculating the correct radicalisation_rate value",
        )
        self.assertHasAttr(
            empty_group,
            "aggregate_hierarchy_weighting",
            "Group -- a valid call to generate_group is not initialising the aggregate_hierarchy_weighting attribute",
        )
        self.assertAlmostEqual(
            empty_group.aggregate_hierarchy_weighting,
            -0.24,
            2,
            "Group -- a valid call to generate_group is not calculating the correct aggregate_hierarchy_weighting value",
        )

    def test_generate_group_max_size(self) -> None:
        """
        Test that generate_group() with no explicit max_size is working as intended.
        """
        # No need to repeat all checks from test_generate_group, just check the max_size attribute
        empty_group: grp.Group = grp.Group()
        empty_group = empty_group.generate_group(
            "FooGroup",
            1,
            "Foobar",
            [
                Agent(
                    "Foo",
                    0.13,
                    ("social", 0.22),
                    {"Foobar": 0.32},
                    False
                ),
                Agent(
                    "Bar",
                    0.12,
                    ("social", 0.1),
                    {"Foobar": -0.8},
                    True,
                    radicalised=True,
                ),
            ],
            cohesion="close",
        )
        self.assertEqual(
            empty_group.max_size,
            2,
            "Group -- a valid call to generate_group with no explicit max_size is not setting the max_size to the number of input members",
        )

    def test_add_attribute_no_args(self) -> None:
        """
        Test that add_attribute with neither 'value' or 'parameters' and 'distribution' keyword arguments will produce the expected error.
        """
        empty_group: grp.Group = grp.Group()
        with self.assertRaises(ValueError, msg="Either explicit 'value' or distribution and valid distribution parameters are expected when adding Group attributes.") as cm:
            empty_group.add_attribute("foo")

    def test_add_attribute_noverwrite(self) -> None:
        """
        Test that add_attribute on an existing attribute when overwrite=False produces the expected warning.
        """
        empty_group: grp.Group = grp.Group()
        with self.assertWarns(UserWarning, msg="WARNING: Attempting to overwrite an existing Group attribute (cohesion) without meaning to.") as cm:
            empty_group.add_attribute("cohesion", value="close", overwrite=False)
        self.assertEqual(
            empty_group.cohesion,
            "neutral",
            "Group -- add_attribute when overwrite=False is changing the specified attribute's value",
        )

    def test_add_attribute_value(self) -> None:
        """
        Test that add_attribute with an explicit value is working as intended.
        """
        empty_group: grp.Group = grp.Group()
        empty_group.add_attribute("foobar", value=True)
        self.assertHasAttr(
            empty_group,
            "foobar",
            "Group -- add_attribute on a new attribute with an explicit value did not add the attribute to the group",
        )
        self.assertIsInstance(
            empty_group.foobar,
            bool,
            "Group -- add_attribute on a new attribute with an explicit value did not add the attribute as the expected data type",
        )
        self.assertTrue(
            empty_group.foobar,
            "Group -- add_attribute on a new attribute with an explicit value did not store the attribute's value correctly",
        )

    def test_add_attribute_distribution(self) -> None:
        """
        Test that add_attribute with a distribution is working as intended.
        """
        empty_group: grp.Group = grp.Group()
        empty_group.add_attribute("aggression", distribution="gaussian")
        self.assertHasAttr(
            empty_group,
            "aggression",
            "Group -- add_attribute on a new attribute with a distribution did not add the attribute to the group",
        )
        self.assertIsInstance(
            empty_group.aggression,
            float,
            "Group -- add_attribute on a new attribute with a distribution did not assign a float value to the attribute",
        )
        self.assertTrue(
            0.0 <= empty_group.aggression <= 1.0,
            "Group -- add_attribute on a new attribute with a distribution and default parameters did not generate a value in the valid range",
        )
