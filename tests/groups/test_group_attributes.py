from __future__ import annotations

import unittest as ut

import gatoh.groups as grp


class TestGroupAttributes(ut.TestCase):
    def test_no_attributes(self) -> None:
        """
        Test that unassigned variables are None at Group init.
        """
        empty_group: grp.Group = grp.Group()
        self.assertIsInstance(
            empty_group,
            grp.Group,
            "Group -- group not initialised correctly",
        )
        self.assertNotHasAttr(
            empty_group,
            "id",
            "Group -- empty group is being created with an id attribute",
        )
        self.assertNotHasAttr(
            empty_group,
            "index",
            "Group -- empty group is being created with an index attribute",
        )
        self.assertEqual(
            empty_group.max_size,
            -1,
            "Group -- empty group is not being created with the default uncapped max_size",
        )
        self.assertNotHasAttr(
            empty_group,
            "hierarchy",
            "Group -- empty group is being created with a hierarchy attribute",
        )
        self.assertEqual(
            empty_group.members,
            [],
            "Group -- empty group is not being created with an empty members list",
        )
        self.assertNotHasAttr(
            empty_group,
            "aggregate_opinion",
            "Group -- empty group is being created with an aggregate_opinion attribute",
        )
        self.assertEqual(
            empty_group.previous_opinion,
            0.0,
            "Group -- empty group is not being created with the default previous_opinion of 0.0",
        )
        self.assertNotHasAttr(
            empty_group,
            "member_benefit_rate",
            "Group -- empty group is being created with a member_benefit_rate attribute",
        )
        self.assertNotHasAttr(
            empty_group,
            "aggregate_susceptibility",
            "Group -- empty group is being created with an aggregate_susceptibility attribute",
        )
        self.assertEqual(
            empty_group.cohesion,
            "neutral",
            "Group -- empty group is not being created with the default cohesion of neutral",
        )
        self.assertEqual(
            empty_group.predominant_personality,
            "neutral",
            "Group -- empty group is not being created with the default predominant_personality of neutral",
        )
        self.assertNotHasAttr(
            empty_group,
            "radicalisation_rate",
            "Group -- empty group is being created with a radicalisation_rate attribute",
        )
        self.assertNotHasAttr(
            empty_group,
            "aggregate_hierarchy_weighting",
            "Group -- empty group is being created with an aggregate_hierarchy_weighting attribute",
        )

    def test_group_in(self) -> None:
        """
        Test that __in__() correctly sees a Group in an iterable.
        """
        empty_group: grp.Group = grp.Group()
        group_iterable: list[grp.Group] = []
        self.assertFalse(
            empty_group in group_iterable,
            "Group -- __in__() is not returning False when the Group is not in an iterable",
        )
        group_iterable.append(empty_group)
        self.assertTrue(
            empty_group in group_iterable,
            "Group -- __in__() is not returning True when the Group is in an iterable",
        )

    def test_initial_str(self) -> None:
        """
        Test that the __str__ representation is returning the expected output initially.
        """
        empty_group: grp.Group = grp.Group()
        with self.assertRaises(AttributeError) as cm:
            str_repr: str = empty_group.__str__()

    def test_init_args(self) -> None:
        """
        Test that *args passed to a Group __init__ are recognised correctly.
        """
        group: grp.Group = grp.Group(
            "GroupFoo",
            1312,
        )
        self.assertEqual(
            group.id,
            "GroupFoo",
            "Group -- a string in __init__ *args is not being assigned to the id attribute",
        )
        self.assertEqual(
            group.max_size,
            1312,
            "Group -- an integer in __init__ *args is not being assigned to the max_size attribute",
        )

    def test_init_kwargs_override(self) -> None:
        """
        Test that existing Group attributes passed in the __init__ **kwargs are correctly overriding the default values.
        """
        group: grp.Group = grp.Group(
            max_size=1312,
            members=["Foo", "Bar"],
            previous_opinion=0.21,
            cohesion="close",
            predominant_personality="rational",
            hierarchy="FooBar",
        )
        self.assertEqual(
            group.max_size,
            1312,
            "Group -- explicit max_size attribute as a kwarg is not overwriting the default value",
        )
        self.assertEqual(
            group.members,
            ["Foo", "Bar"],
            "Group -- explicit members attribute as a kwarg is not overwriting the default value",
        )
        self.assertEqual(
            group.previous_opinion,
            0.21,
            "Group -- explicit previous_opinion attribute as a kwarg is not overwriting the default value",
        )
        self.assertEqual(
            group.cohesion,
            "close",
            "Group -- explicit cohesion attribute as a kwarg is not overwriting the default value",
        )
        self.assertEqual(
            group.predominant_personality,
            "rational",
            "Group -- explicit predominant_personality attribute as a kwarg is not overwriting the default value",
        )
        self.assertEqual(
            group.hierarchy,
            "FooBar",
            "Group -- explicit hierarchy attribute as a kwarg is not setting the attribute correctly",
        )

    def test_init_kwargs_unseen(self) -> None:
        """
        Test that unseen **kwargs passed to a Group __init__ are being assigned correctly.
        """
        group: grp.Group = grp.Group(
            cohesion_level=0.34,
            influential=True,
            influential_members={"Foo": True, "Bar": False},
            cohesion_rw=(0.0, 0.01),
            aggression="very strong",
        )
        self.assertHasAttr(
            group,
            "cohesion_level",
            "Group -- unseen attribute passed in **kwargs is not being assigned correctly (float)",
        )
        self.assertIsInstance(
            group.cohesion_level,
            float,
            "Group -- unseen attribute is not being stored as the correct type (float)",
        )
        self.assertEqual(
            group.cohesion_level,
            0.34,
            "Group -- unseen attribute value is not being stored correctly (float)",
        )
        self.assertHasAttr(
            group,
            "influential",
            "Group -- unseen attribute passed in **kwargs is not being assigned correctly (bool)",
        )
        self.assertIsInstance(
            group.influential,
            bool,
            "Group -- unseen attribute is not being stored as the correct type (bool)",
        )
        self.assertTrue(
            group.influential,
            "Group -- unseen attribute value is not being stored correctly (bool)",
        )
        self.assertHasAttr(
            group,
            "influential_members",
            "Group -- unseen attribute passed in **kwargs is not being assigned correctly (dict)",
        )
        self.assertIsInstance(
            group.influential_members,
            dict,
            "Group -- unseen attribute is not being stored as the correct type (dict)",
        )
        self.assertEqual(
            group.influential_members,
            {"Foo": True, "Bar": False},
            "Group -- unseen attribute value is not being stored correctly (dict)",
        )
        self.assertHasAttr(
            group,
            "cohesion_rw",
            "Group -- unseen attribute passed in **kwargs is not being assigned correctly (tuple)",
        )
        self.assertIsInstance(
            group.cohesion_rw,
            tuple,
            "Group -- unseen attribute is not being stored as the correct type (tuple)",
        )
        self.assertEqual(
            group.cohesion_rw,
            (0.0, 0.01),
            "Group -- unseen attribute value is not being stored correctly (tuple)",
        )
        self.assertHasAttr(
            group,
            "aggression",
            "Group -- unseen attribute passed in **kwargs is not being assigned correctly (str)",
        )
        self.assertIsInstance(
            group.aggression,
            str,
            "Group -- unseen attribute is not being stored as the correct type (str)",
        )
        self.assertEqual(
            group.aggression,
            "very strong",
            "Group -- unseen attribute value is not being stored correctly (str)",
        )

    def test_init_kwargs_mixed(self) -> None:
        """
        Test that a mix of explicit and unseen kwargs are initialised correctly.
        """
        group: grp.Group = grp.Group(
            max_size=1312,
            hierarchy="Foobar",
            cohesion_level=0.34,
        )
        self.assertHasAttr(
            group,
            "hierarchy",
            "Group -- explicit hierarchy attribute in mixed **kwargs is not being assigned correctly",
        )
        self.assertHasAttr(
            group,
            "cohesion_level",
            "Group -- unseen attribute in mixed **kwargs is not being assigned correctly",
        )
        self.assertEqual(
            group.max_size,
            1312,
            "Group -- explicit max_size attribute in mixed **kwargs is not overwriting the default value",
        )
        self.assertEqual(
            group.hierarchy,
            "Foobar",
            "Group -- explicit hierarchy attribute in mixed **kwargs is not storing the value correctly",
        )
        self.assertEqual(
            group.cohesion_level,
            0.34,
            "Group -- unseen attribute in mixed **kwargs is not storing its value correctly",
        )

    def test_init_mixed(self) -> None:
        """
        Test that __init__ with a mix of *args and **kwargs is working correctly.
        """
        group: grp.Group = grp.Group(
            "Foobar",
            1312,
            cohesion_level=0.34,
        )
        self.assertHasAttr(
            group,
            "id",
            "Group -- id is not being assigned correctly from a mixed args and kwargs init",
        )
        self.assertHasAttr(
            group,
            "cohesion_level",
            "Group -- unseen attribute is not being assigned correctly from a mixed args and kwargs init",
        )
        self.assertEqual(
            group.id,
            "Foobar",
            "Group -- id is not being stored correctly from a mixed args and kwargs init",
        )
        self.assertEqual(
            group.max_size,
            1312,
            "Group -- max_size is not being stored correctly from a mixed args and kwargs init",
        )
        self.assertEqual(
            group.cohesion_level,
            0.34,
            "Group -- unseen attribute is not being stored correctly from a mixed args and kwargs init",
        )

    def test_valid_str(self) -> None:
        """
        Test that __str__ on a Group with all required attributes returns the expected representation.
        """
        group: grp.Group = grp.Group(
            "Foobar",
            members=["Foo", "Bar"],
            hierarchy="Barfoo",
            aggregate_opinion=0.22
        )
        string_repr: str = group.__str__()
        self.assertEqual(
            string_repr,
            "Group Foobar composed of members: [\"Foo\", \"Bar\"], and belonging to hierarchy Barfoo with an aggregate opinion of 0.22"
        )
