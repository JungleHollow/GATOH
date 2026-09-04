from __future__ import annotations

import unittest as ut

from experiments.Base.SocialSusceptibility.plot_results import aggregate_opinion
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
        self.assertNotHasAttr(
            empty_group,
            "silencing_rate",
            "Group -- empty group is being created with a silencing_rate attribute",
        )
        self.assertIsNone(
            empty_group.rw_distribution,
            "Group -- empty group is not using the default rw_distribution value of None",
        )
        self.assertIsNone(
            empty_group.opinion_rw,
            "Group -- empty group is not using the default opinion_rw value of None",
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
            rw_distribution=(0.0, 0.13),
            opinion_rw=(0.0, 0.12),
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
        self.assertEqual(
            group.rw_distribution,
            (0.0, 0.13),
            "Group -- explicit rw_distribution attribute as a kwarg is not setting the attribute correctly",
        )
        self.assertEqual(
            group.opinion_rw,
            (0.0, 0.12),
            "Group -- explicit opinion_rw attribute as a kwarg is not setting the attribute correctly",
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

    def test_get_attribute_empty(self) -> None:
        """
        Test that get_attribute() on a non-existing Group attribute raises a warning and returns None.
        """
        empty_group: grp.Group = grp.Group()
        with self.assertWarns(UserWarning, msg="WARNING: Attempting to get a group attribute (foobar) which doesn't exist.") as cm:
            attribute = empty_group.get_attribute("foobar")
        self.assertIsNone(
            attribute,
            "Group -- get_attribute() on a non-existing attribute is not returning None",
        )

    def test_store_previous_opinion(self) -> None:
        """
        Test that store_previous_opinion() is working as intended.
        """
        group: grp.Group = grp.Group(aggregate_opinion=0.44)
        group.store_previous_opinion()
        self.assertAlmostEqual(
            group.previous_opinion,
            0.44,
            5,
            "Group -- store_previous_opinion() is not storing the group's aggregate opinion into its previous_opinion correctly",
        )

    def test_get_num_members(self) -> None:
        """
        Test that get_num_members() is working as intended.
        """
        group: grp.Group = grp.Group(members=["Foo", "Bar"])
        num_members: int = group.get_num_members()
        self.assertIsInstance(
            num_members,
            int,
            "Group -- get_num_members() is not returning an int value",
        )
        self.assertEqual(
            num_members,
            2,
            "Group -- get_num_members() is not reporting the correct number of members in a group",
        )

    def test_recalculate_aggregate_opinion_invalid(self) -> None:
        """
        Test that recalculate_aggregate_opinion() with an invalid number of input opinions will raise the expected error.
        """
        group: grp.Group = grp.Group(members=["foo", "bar"])
        with self.assertRaises(ValueError, msg="The number of member opinions input does not match the number of group members") as cm:
            group.recalculate_aggregate_opinion([0.13, 0.12, 0.99, 0.1])

    def test_recalculate_aggregate_opinion(self) -> None:
        """
        Test that recalculate_aggregate_opinion() is working as intended.
        """
        group: grp.Group = grp.Group(members=["foo", "bar"])
        group.recalculate_aggregate_opinion([0.13, 0.12])
        self.assertHasAttr(
            group,
            "aggregate_opinion",
            "Group -- a valid call to recalculate_aggregate_opinion on a new group is not initialising the aggregate_opinion attribute",
        )
        self.assertAlmostEqual(
            group.aggregate_opinion,
            0.125,
            2,
            "Group -- a valid call to recalculate_aggregate_opinion is not setting aggregate_opinion to the correct value",
        )

    def test_recalculate_radicalisation_rate_invalid(self) -> None:
        """
        Test that recalculate_radicalisation_rate() with an invalid number of input radicalisation statuses will raise the expected error.
        """
        group: grp.Group = grp.Group(members=["foo", "bar"])
        with self.assertRaises(ValueError, msg="The number of radicalisation statuses input does not match the number of group members") as cm:
            group.recalculate_radicalisation_rate([False, False, True, False])

    def test_recalculate_radicalisation_rate(self) -> None:
        """
        Test that recalculate_radicalisation_rate() is working as intended.
        """
        group: grp.Group = grp.Group(members=["foo", "bar"])
        group.recalculate_radicalisation_rate([True, False])
        self.assertHasAttr(
            group,
            "radicalisation_rate",
            "Group -- a valid call to recalculate_radicalisation_rate on a new group is not initialising the radicalisation_rate attribute",
        )
        self.assertAlmostEqual(
            group.radicalisation_rate,
            0.5,
            1,
            "Group -- a valid call to recalculate_radicalisation_rate is not setting radicalisation_rate to the correct value",
        )

    def test_recalculate_member_benefit_rate_invalid(self) -> None:
        """
        Test that recalculate_member_benefit_rate() with an invalid number of benefits input will raise the expected error.
        """
        group: grp.Group = grp.Group(members=["foo", "bar"])
        with self.assertRaises(ValueError, msg="The number of benefit statuses input does not match the number of group members") as cm:
            group.recalculate_member_benefit_rate([False, False, False, True])

    def test_recalculate_member_benefit_rate(self) -> None:
        """
        Test that recalculate_member_benefit_rate() is working as intended.
        """
        group: grp.Group = grp.Group(members=["foo", "bar"])
        group.recalculate_member_benefit_rate([True, False])
        self.assertHasAttr(
            group,
            "member_benefit_rate",
            "Group -- a valid call to recalculate_member_benefit_rate on a new group is not initialising the member_benefit_rate attribute",
        )
        self.assertAlmostEqual(
            group.member_benefit_rate,
            0.5,
            1,
            "Group -- a valid call to recalculate_member_benefit_rate is not setting member_benefit_rate to the correct value",
        )

    def test_recalculate_silencing_rate_invalid(self) -> None:
        """
        Test that recalculate_silencing_rate() with an invalid number of weightings input will raise the expected error.
        """
        group: grp.Group = grp.Group(members=["foo", "bar"])
        with self.assertRaises(ValueError, msg="The number of silencing statuses input does not match the number of group members") as cm:
            group.recalculate_silencing_rate([True, True, False, False, False])

    def test_recalculate_silencing_rate(self) -> None:
        """
        Test that recalculate_silencing_rate() is working as intended.
        """
        group: grp.Group = grp.Group(members=["foo", "bar"])
        group.recalculate_silencing_rate([True, False])
        self.assertHasAttr(
            group,
            "silencing_rate",
            "Group -- a valid call to recalculate_silencing_rate on a new group is not initialising the silencing_rate attribute",
        )
        self.assertAlmostEqual(
            group.silencing_rate,
            0.5,
            1,
            "Group -- a valid call to recalculate_silencing_rate is not setting silencing_rate to the correct value",
        )

    def test_recalculate_hierarchy_weighting_invalid(self) -> None:
        """
        Test that recalculate_hierarchy_weighting() with an invalid number of weightings input will raise the expected error.
        """
        group: grp.Group = grp.Group(members=["foo", "bar"])
        with self.assertRaises(ValueError, msg="The number of hierarchy weightings input does not match the number of group members") as cm:
            group.recalculate_hierarchy_weighting([0.13, 0.12, 0.99, 0.2, 0.45])

    def test_recalculate_hierarchy_weighting(self) -> None:
        """
        Test that recalculate_hierarchy_weighting() is working as intended.
        """
        group: grp.Group = grp.Group(members=["foo", "bar"])
        group.recalculate_hierarchy_weighting([0.13, 0.12])
        self.assertHasAttr(
            group,
            "aggregate_hierarchy_weighting",
            "Group -- a valid call to recalculate_hierarchy_weighting on a new group is not initialising the aggregate_hierarchy_weighting attribute",
        )
        self.assertAlmostEqual(
            group.aggregate_hierarchy_weighting,
            0.125,
            2,
            "Group -- a valid call to recalculate_hierarchy_weighting is not setting aggregate_hierarchy_weighting to the correct value",
        )

    def test_determine_predominant_personality_invalid(self) -> None:
        """
        Test that determine_predominant_personality() with an invalid number of personalities input will raise the expected error.
        """
        group: grp.Group = grp.Group(members=["foo", "bar"])
        with self.assertRaises(ValueError, msg="The number of personality types input does not match the number of group members") as cm:
            group.determine_predominant_personality(["neutral", "neutral", "neutral"])

    def test_determine_predominant_personality(self) -> None:
        """
        Test that determine_predominant_personality() is working as intended.
        """
        group: grp.Group = grp.Group(members=["foo", "bar", "foobar", "barfoo"])
        group.determine_predominant_personality(["rational", "impulsive", "impulsive", "social"])
        self.assertEqual(
            group.predominant_personality,
            "impulsive",
            "Group -- a valid call to determine_predominant_personality is not reporting the correct predominant personality type among members",
        )

    def test_change_aggregate_opinion_empty(self) -> None:
        """
        Test that change_aggregate_opinion() when aggregate_opinion has not been initialised will raise the expected error.
        """
        empty_group: grp.Group = grp.Group()
        with self.assertRaises(AttributeError, msg="aggregate_opinion has not been initialised for this group") as cm:
            per_agt_delta: float = empty_group.change_aggregate_opinion(0.1312)

    def test_change_aggregate_opinion_dtype(self) -> None:
        """
        Test that change_aggregate_opinion() with an invalid data type will raise the expected error.
        """
        group: grp.Group = grp.Group(members=["foo", "bar"], aggregate_opinion=0.125)
        with self.assertRaises(TypeError, msg="opinion_delta must be a float") as cm:
            per_agt_delta: float = group.change_aggregate_opinion("0.1")

    def test_change_aggregate_opinion(self) -> None:
        """
        Test that change_aggregate_opinion() is working as intended.
        """
        group: grp.Group = grp.Group(members=["foo", "bar"], aggregate_opinion=0.125)
        per_agt_delta: float = group.change_aggregate_opinion(0.1)
        self.assertIsInstance(
            per_agt_delta,
            float,
            "Group -- a valid call to change_aggregate_opinion() is not returning a float value (no cap)",
        )
        self.assertAlmostEqual(
            per_agt_delta,
            0.1,
            1,
            "Group -- a valid call to change_aggregate_opinion() is not returning the correct per-agent delta (no cap)",
        )

    def test_change_aggregate_opinion_capped(self) -> None:
        """
        Test that change_aggregate_opinion() with a delta that would reach the value cap is working as intended.
        """
        group: grp.Group = grp.Group(members=["foo", "bar"], aggregate_opinion=0.94)
        per_agt_delta: float = group.change_aggregate_opinion(0.47)
        self.assertIsInstance(
            per_agt_delta,
            float,
            "Group -- a valid call to change_aggregate_opinion() is not returning a float value (cap)",
        )
        self.assertAlmostEqual(
            per_agt_delta,
            0.06,
            2,
            "Group -- a valid call to change_aggregate_opinion() is not returning the correct per-agent delta (cap)",
        )

    def test_change_radicalisation_rate_empty(self) -> None:
        """
        Test that change_radicalisation_rate() when radicalisation_rate has not yet been initialised will raise the expected error.
        """
        empty_group: grp.Group = grp.Group()
        with self.assertRaises(AttributeError, msg="radicalisation_rate has not yet been initialised for this group") as cm:
            agent_radicalisations: dict[str, bool] = empty_group.change_radicalisation_rate(0.1312)

    def test_change_radicalisation_rate_dtype(self) -> None:
        """
        Test that change_radicalisation_rate() with an invalid data type will raise the expected error.
        """
        group: grp.Group = grp.Group(radicalisation_rate=0.1)
        with self.assertRaises(TypeError, msg="rate_delta must be a float value") as cm:
            agent_radicalisations: dict[str, bool] = group.change_radicalisation_rate("seven")

    def test_change_radicalisation_rate(self) -> None:
        """
        Test that change_radicalisation_rate() is working as intended.
        """
        group_members: list[str] = ["foo", "bar"]
        group: grp.Group = grp.Group(members=group_members, radicalisation_rate=0.0)
        agent_radicalisations: dict[str, bool] = group.change_radicalisation_rate(0.4)
        self.assertIsInstance(
            agent_radicalisations,
            dict,
            "Group -- change_radicalisation_rate is not returning a dictionary value",
        )
        self.assertAlmostEqual(
            group.radicalisation_rate,
            0.4,
            1,
            "Group -- change_radicalisation_rate is not actually changing the group's radicalisation_rate attribute",
        )
        self.assertEqual(
            len(agent_radicalisations.keys()),
            2,
            "Group -- change_radicalisation_rate is not reporting the correct number of group members",
        )
        radical_count: int = sum(int(flag) for flag in agent_radicalisations.values())
        self.assertEqual(
            radical_count,
            1,
            "Group -- change_radicalisation_rate is not producing the correct number of radicalised agents for the new radicalisation_rate value",
        )
        for member in agent_radicalisations:
            self.assertIn(
                member,
                group_members,
                "Group -- change_radicalisation_rate is reporting a radicalisation status for a non-existent member",
            )

    def test_change_rw_distribution_invalid_outer(self) -> None:
        """
        Test that change_rw_distribution() with a non-tuple input will raise the expected error.
        """
        group: grp.Group = grp.Group()
        with self.assertRaises(TypeError, msg="parameters must be a (float, float) tuple") as cm:
            group.change_rw_distribution([0.0, 0.1])

    def test_change_rw_distribution_invalid_inner(self) -> None:
        """
        Test that change_rw_distribution() with an invalid data type in the tuple input will raise the expected error.
        """
        group: grp.Group = grp.Group()
        with self.assertRaises(TypeError, msg="parameters must be a (float, float) tuple") as cm:
            group.change_rw_distribution((0.0, "0.3"))

    def test_change_rw_distribution(self) -> None:
        """
        Test that change_rw_distribution() is working as intended.
        """
        group: grp.Group = grp.Group()
        group.change_rw_distribution((0.0, 0.1))
        self.assertIsNotNone(
            group.rw_distribution,
            "Group -- a valid call to change_rw_distribution on a new group is not assigning a non-None value to rw_distribution",
        )
        self.assertEqual(
            group.rw_distribution,
            (0.0, 0.1),
            "Group -- a valid call to change_rw_distribution is not setting rw_distribution to the correct value",
        )

    def test_change_opinion_rw_invalid_outer(self) -> None:
        """
        Test that change_opinion_rw() with a non-tuple input will raise the expected error.
        """
        group: grp.Group = grp.Group()
        with self.assertRaises(TypeError, msg="rw_params must be a (float, float) tuple") as cm:
            group.change_opinion_rw([0.0, 0.1])

    def test_change_opinion_rw_invalid_inner(self) -> None:
        """
        Test that change_opinion_rw() with an invalid data type in the tuple input will raise the expected error.
        """
        group: grp.Group = grp.Group()
        with self.assertRaises(TypeError, msg="rw_params must be a (float, float) tuple") as cm:
            group.change_opinion_rw((0.0, "0.3"))

    def test_change_opinion_rw(self) -> None:
        """
        Test that change_opinion_rw() is working as intended.
        """
        group: grp.Group = grp.Group()
        group.change_opinion_rw((0.0, 0.1))
        self.assertIsNotNone(
            group.opinion_rw,
            "Group -- a valid call to change_opinion_rw on a new group is not assigning a non-None value to opinion_rw",
        )
        self.assertEqual(
            group.opinion_rw,
            (0.0, 0.1),
            "Group -- a valid call to change_opinion_rw is not setting opinion_rw to the correct value",
        )

    def test_change_predominant_personality_dtype(self) -> None:
        """
        Test that change_predominant_personality with an invalid data type will raise the expected error.
        """
        group: grp.Group = grp.Group()
        with self.assertRaises(TypeError, msg="The input personality must be a string") as cm:
            member_personalities: dict[str, str] = group.change_predominant_personality(False)

    def test_change_predominant_personality_invalid(self) -> None:
        """
        Test that change_predominant_personality with an invalid personality type will raise the expected error.
        """
        group: grp.Group = grp.Group()
        with self.assertRaises(ValueError, msg="The input personality is not currently supported") as cm:
            member_personalities: dict[str, str] = group.change_predominant_personality("stubborn")

    def test_change_predominant_personality(self) -> None:
        """
        Test that change_predominant_personality is working as intended.
        """
        group_members: list[str] = ["foo", "bar", "oof", "rab"]
        group: grp.Group = grp.Group(members=group_members)
        member_personalities: dict[str, str] = group.change_predominant_personality("social")
        self.assertIsInstance(
            member_personalities,
            dict,
            "Group -- a valid call to change_predominant_personality is not returning a dict value",
        )
        self.assertEqual(
            group.predominant_personality,
            "social",
            "Group -- a valid call to change_predominant_personality is not changing the predominant_personality attribute",
        )
        personality_counts: dict[str, int] = {personality: 0 for personality in grp.PERSONALITIES}
        for member, personality in member_personalities.items():
            self.assertIn(
                member,
                group_members,
                "Group -- a valid call to change_predominant_personality is reporting a new personality type for a non-existent member",
            )
            self.assertIn(
                personality,
                grp.PERSONALITIES,
                "Group -- a valid call to change_predominant_personality is reporting a new personality type that is unsupported",
            )
            personality_counts[personality] += 1
        self.assertEqual(
            max(personality_counts, key=lambda x: personality_counts[x]),
            "social",
            "Group -- a valid call to change_predominant_personality is not generating member personalities which result in the correct predominant type",
        )

    def test_change_benefit_rate_empty(self) -> None:
        """
        Test that change_benefit_rate on an uninitialised group will raise the expected error.
        """
        group: grp.Group = grp.Group()
        with self.assertRaises(AttributeError, msg="member_benefit_rate has not yet been initialised for this group") as cm:
            member_benefits: dict[str, bool] = group.change_benefit_rate(0.1312)

    def test_change_benefit_rate_dtype(self) -> None:
        """
        Test that change_benefit_rate with an invalid data type input will raise the expected error.
        """
        group: grp.Group = grp.Group(member_benefit_rate=0.2)
        with self.assertRaises(TypeError, msg="rate_delta must be a float value") as cm:
            member_benefits: dict[str, bool] = group.change_benefit_rate("zero_point_one")

    def test_change_benefit_rate(self) -> None:
        """
        Test that change_benefit_rate is working as intended.
        """
        group_members: list[str] = ["foo", "bar"]
        group: grp.Group = grp.Group(members=group_members, member_benefit_rate=0.0)
        member_benefits: dict[str, bool] = group.change_benefit_rate(0.4)
        self.assertIsInstance(
            member_benefits,
            dict,
            "Group -- change_benefit_rate is not returning a dictionary value",
        )
        self.assertAlmostEqual(
            group.member_benefit_rate,
            0.4,
            1,
            "Group -- change_benefit_rate is not actually changing the member_benefit_rate attribute",
        )
        self.assertEqual(
            len(member_benefits.keys()),
            2,
            "Group -- change_benefit_rate is not reporting the correct number of group members",
        )
        benefit_count: int = sum(int(flag) for flag in member_benefits.values())
        self.assertEqual(
            benefit_count,
            1,
            "Group -- change_benefit_rate is not producing the correct number of benefited agents for the new member_benefit_rate value",
        )
        for member in member_benefits:
            self.assertIn(
                member,
                group_members,
                "Group -- change_benefit_rate is reporting a benefit status for a non-existent member",
            )

    def test_change_silencing_rate_empty(self) -> None:
        """
        Test that change_silencing_rate on an uninitialised group will raise the expected error.
        """
        group: grp.Group = grp.Group()
        with self.assertRaises(AttributeError, msg="silencing_rate has not yet been initialised for this group") as cm:
            members_silenced: dict[str, bool] = group.change_silencing_rate(0.1312)

    def test_change_silencing_rate_dtype(self) -> None:
        """
        Test that change_silencing_rate with an incorrect data type input will raise the expected error.
        """
        group: grp.Group = grp.Group(silencing_rate=0.1)
        with self.assertRaises(TypeError, msg="rate_delta must be a float value") as cm:
            members_silenced: dict[str, bool] = group.change_silencing_rate("zero_point_two")

    def test_change_silencing_rate(self) -> None:
        """
        Test that change_silencing_rate is working as intended.
        """
        group_members: list[str] = ["foo", "bar"]
        group: grp.Group = grp.Group(members=group_members, silencing_rate=0.0)
        members_silenced: dict[str, bool] = group.change_silencing_rate(0.4)
        self.assertIsInstance(
            members_silenced,
            dict,
            "Group -- a valid call to change_silencing_rate is not returning a dictionary value",
        )
        self.assertAlmostEqual(
            group.silencing_rate,
            0.4,
            1,
            "Group -- a valid call to change_silencing_rate is not changing the silencing_rate attribute",
        )
        self.assertEqual(
            len(members_silenced.keys()),
            2,
            "Group -- a valid call to change_silencing_rate is not reporting a new silencing status for the correct number of agents",
        )
        silenced_count: int = sum(int(flag) for flag in members_silenced.values())
        self.assertEqual(
            silenced_count,
            1,
            "Group -- a valid call to change_silencing_rate is not producing the correct number of silenced agents for the new silencing_rate value",
        )
        for member in members_silenced:
            self.assertIn(
                member,
                group_members,
                "Group -- a valid call to change_silencing_rate is reporting a silencing status for a non-existent member",
            )

    def test_change_aggregate_hierarchy_weighting_empty(self) -> None:
        """
        Test that change_aggregate_hierarchy_weighting on an uninitialised group will raise the expected error.
        """
        group: grp.Group = grp.Group()
        with self.assertRaises(AttributeError, msg="aggregate_hierarchy_weighting has not been initialised for this group") as cm:
            per_agent_delta: float = group.change_aggregate_hierarchy_weighting(0.1312)

    def test_change_aggregate_hierarchy_weighting_dtype(self) -> None:
        """
        Test that change_aggregate_hierarchy_weighting with an invalid data type input will raise the expected error.
        """
        group: grp.Group = grp.Group(aggregate_hierarchy_weighting=0.1)
        with self.assertRaises(TypeError, msg="weighting_delta must be a float") as cm:
            per_agent_delta: float = group.change_aggregate_hierarchy_weighting("zero_point_two")

    def test_change_aggregate_hierarchy_weighting(self) -> None:
        """
        Test that change_aggregate_hierarchy_weighting is working as intended.
        """
        group: grp.Group = grp.Group(members=["foo", "bar"], aggregate_hierarchy_weighting=0.25)
        per_agent_delta: float = group.change_aggregate_hierarchy_weighting(-0.13)
        self.assertIsInstance(
            per_agent_delta,
            float,
            "Group -- a valid call to change_aggregate_hierarchy_weighting is not returning a float value",
        )
        self.assertEqual(
            per_agent_delta,
            -0.13,
            "Group -- a valid call to change_aggregate_hierarchy_weighting is not returning the correct per-agent delta",
        )
        self.assertEqual(
            group.aggregate_hierarchy_weighting,
            0.12,
            "Group -- a valid call to change_aggregate_hierarchy_weighting is not changing the aggregate_hierarchy_weighting attribute",
        )

    def test_set_index_invalid(self) -> None:
        """
        Test that set_index() with an incorrect data type will raise the expected error.
        """
        group: grp.Group = grp.Group()
        with self.assertRaises(TypeError, msg="index must be an integer") as cm:
            group.set_index("six-seven")

    def test_set_index(self) -> None:
        """
        Test that set_index() is working as intended.
        """
        group: grp.Group = grp.Group()
        group.set_index(4)
        self.assertHasAttr(
            group,
            "index",
            "Group -- a valid call to set_index on a new group is not initialising the index attribute",
        )
        self.assertEqual(
            group.index,
            4,
            "Group -- a valid call to set_index is not changing the index to the correct value",
        )

    def test_set_max_size_dtype(self) -> None:
        """
        Test that set_max_size() with an incorrect data type will raise the expected error.
        """
        group: grp.Group = grp.Group()
        with self.assertRaises(TypeError, msg="max_size must be an integer") as cm:
            group.set_max_size("no cap")

    def test_set_max_size_invalid(self) -> None:
        """
        Test that set_max_size() with an invalid max_size value when no_limit=False will raise the expected error.
        """
        group: grp.Group = grp.Group()
        with self.assertRaises(ValueError, msg="max_size must be equal or greater to 1") as cm:
            group.set_max_size(-1312)

    def test_set_max_size_limit(self) -> None:
        """
        Test that set_max_size() with no_limit=False is working as intended.
        """
        group: grp.Group = grp.Group()
        group.set_max_size(1312)
        self.assertEqual(
            group.max_size,
            1312,
            "Group -- a valid call to set_max_size with no_limit=False is not setting the max_size attribute to the correct value",
        )

    def test_set_max_size_nolimit(self) -> None:
        """
        Test that set_max_size() with no_limit=True is working as intended.
        """
        # Initialise the group with an explicit max_size
        group: grp.Group = grp.Group(2)
        group.set_max_size(1111, no_limit=True)
        self.assertEqual(
            group.max_size,
            -1,
            "Group -- a valid call to set_max_size with no_limit=True is not setting the max_size attribute to -1",
        )

    def test_set_hierarchy_invalid(self) -> None:
        """
        Test that set_hierarchy() with an invalid data type will raise the expected error.
        """
        group: grp.Group = grp.Group()
        with self.assertRaises(TypeError, msg="hierarchy must be a string") as cm:
            group.set_hierarchy(False)

    def test_set_hierarchy(self) -> None:
        """
        Test that set_hierarchy() is working as intended.
        """
        group: grp.Group = grp.Group()
        group.set_hierarchy("foobar")
        self.assertHasAttr(
            group,
            "hierarchy",
            "Group -- a valid call to set_hierarchy on a new group is not initialising the hierarchy attribute",
        )
        self.assertEqual(
            group.hierarchy,
            "foobar",
            "Group -- a valid call to set_hierarchy is not changing the group's hierarchy to the correct value",
        )

    def test_set_cohesion_dtype(self) -> None:
        """
        Test that set_cohesion() with an incorrect data type will raise the expected error.
        """
        group: grp.Group = grp.Group()
        with self.assertRaises(TypeError, msg="cohesion must be a string") as cm:
            group.set_cohesion(True)

    def test_set_cohesion_invalid(self) -> None:
        """
        Test that set_cohesion() with an unsupported cohesion type will raise the expected error.
        """
        group: grp.Group = grp.Group()
        with self.assertRaises(ValueError, msg="The cohesion type 'friendly' is not supported -- cannot change the Group's cohesion type") as cm:
            group.set_cohesion("friendly")

    def test_set_cohesion(self) -> None:
        """
        Test that set_cohesion() is working as intended.
        """
        group: grp.Group = grp.Group()
        group.set_cohesion("close")
        self.assertEqual(
            group.cohesion,
            "close",
            "Group -- a valid call to set_cohesion is not setting the cohesion type to the correct value",
        )

    def test_group_at_capacity(self) -> None:
        """
        Test that group_at_capacity() is working as intended.
        """
        group: grp.Group = grp.Group(2)
        at_capacity: bool = group.group_at_capacity()
        self.assertIsInstance(
            at_capacity,
            bool,
            "Group -- group_at_capacity is not returning a boolean value (with limit)",
        )
        self.assertFalse(
            at_capacity,
            "Group -- group_at_capacity is not reporting that the group is not at capacity (with limit)"
        )
        group.members = ["foo", "bar"]
        at_capacity = group.group_at_capacity()
        self.assertTrue(
            at_capacity,
            "Group -- group_at_capacity is not reporting that the group is at capacity (with limit)",
        )

    def test_group_at_capacity_nolimit(self) -> None:
        """
        Test that group_at_capacity() is working as intended when the group has no max size.
        """
        group: grp.Group = grp.Group()
        at_capacity: bool = group.group_at_capacity()
        self.assertIsInstance(
            at_capacity,
            bool,
            "Group -- group_at_capacity is not returning a boolean value (no limit)",
        )
        self.assertFalse(
            at_capacity,
            "Group -- group_at_capacity is not returning False when there is no limit",
        )
        group.members = ["foo", "bar"]
        at_capacity = group.group_at_capacity()
        self.assertFalse(
            at_capacity,
            "Group -- group_at_capacity is not returning False when there is no limit",
        )

    def test_is_radicalised_empty(self) -> None:
        """
        Test that is_radicalised() when the radicalisation_rate has not been initialised will raise the expected error.
        """
        group: grp.Group = grp.Group()
        with self.assertRaises(AttributeError, msg="radicalisation_rate has not been initialised yet for this group") as cm:
            is_radicalised: bool = group.is_radicalised()

    def test_is_radicalised(self) -> None:
        """
        Test that is_radicalised() is working as intended.
        """
        group: grp.Group = grp.Group(radicalisation_rate=0.75)
        is_radicalised: bool = group.is_radicalised()
        self.assertIsInstance(
            is_radicalised,
            bool,
            "Group -- a valid call to is_radicalised() is not returning a boolean value",
        )
        self.assertTrue(
            is_radicalised,
            "Group -- a valid call to is_radicalised() is not reporting that the group is radicalised at the specified threshold",
        )
        high_threshold: bool = group.is_radicalised(threshold=0.95)
        self.assertFalse(
            high_threshold,
            "Group -- a valid call to is_radicalised() is not reporting that the group is not radicalised at the specified threshold",
        )

    def test_is_benefited_empty(self) -> None:
        """
        Test that is_benefited() when the member_benefit_rate has not been initialised will raise the expected error.
        """
        group: grp.Group = grp.Group()
        with self.assertRaises(AttributeError, msg="member_benefit_rate has not yet been initialised for this group") as cm:
            is_benefited: bool = group.is_benefited()

    def test_is_benefited(self) -> None:
        """
        Test that is_benefited() is working as intended.
        """
        group: grp.Group = grp.Group(member_benefit_rate=0.75)
        is_benefited: bool = group.is_benefited()
        self.assertIsInstance(
            is_benefited,
            bool,
            "Group -- a valid call to is_benefited() is not returning a boolean value",
        )
        self.assertTrue(
            is_benefited,
            "Group -- a valid call to is_benefited() is not reporting that the group is radicalised at the specified threshold",
        )
        high_threshold: bool = group.is_benefited(threshold=0.95)
        self.assertFalse(
            high_threshold,
            "Group -- a valid call to is_benefited() is not reporting that the group is not radicalised at the specified threshold",
        )

    def test_is_silenced_empty(self) -> None:
        """
        Test that is_silenced() when the silencing_rate has not been initialised will raise the expected error.
        """
        group: grp.Group = grp.Group()
        with self.assertRaises(AttributeError, msg="silencing_rate has not yet been initialised for this group") as cm:
            is_silenced: bool = group.is_silenced()

    def test_is_silenced(self) -> None:
        """
        Test that is_silenced() is working as intended.
        """
        group: grp.Group = grp.Group(silencing_rate=0.8)
        is_silenced: bool = group.is_silenced()
        self.assertIsInstance(
            is_silenced,
            bool,
            "Group -- a valid call to is_silenced() is not returning a boolean value",
        )
        self.assertTrue(
            is_silenced,
            "Group -- a valid call to is_silenced() is not reporting that the group is silenced at the specified threshold",
        )
        high_threshold: bool = group.is_silenced(threshold=0.95)
        self.assertFalse(
            high_threshold,
            "Group -- a valid call to is_silenced() is not reporting that the group is not silenced at the specified threshold",
        )
