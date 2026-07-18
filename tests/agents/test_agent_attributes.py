from __future__ import annotations

import unittest as ut

import gatoh.agents as agt


class TestAgentAttributes(ut.TestCase):
    def test_no_attributes(self) -> None:
        """
        Test that unassigned variables are None at Agent init.
        """
        empty_agent: agt.Agent = agt.Agent()
        self.assertIsInstance(
            empty_agent, agt.Agent, "Agent -- agent not initialised correctly"
        )
        self.assertNotHasAttr(
            empty_agent,
            "id",
            "Agent -- empty agent is being initialised with an id attribute",
        )
        self.assertNotHasAttr(
            empty_agent,
            "index",
            "Agent -- empty agent is being initialised with an index attribute",
        )
        self.assertNotHasAttr(
            empty_agent,
            "opinion",
            "Agent -- empty agent is being initialised with an opinion attribute",
        )
        self.assertNotHasAttr(
            empty_agent,
            "personal_benefit",
            "Agent -- empty agent is being initialised with a personal_benefit attribute",
        )
        self.assertNotHasAttr(
            empty_agent,
            "social_susceptibility",
            "Agent -- empty agent is being initialised with a social_susceptibility attribute",
        )
        self.assertEqual(
            empty_agent.social_weightings,
            {},
            "Agent -- empty agent social_weightings are not being initialised as an empty dict",
        )
        self.assertEqual(
            empty_agent.is_silenced,
            {},
            "Agent -- empty agent is_silenced are not being initialised as an empty dict",
        )
        self.assertEqual(
            empty_agent.previous_opinion,
            0.0,
            "Agent -- empty agent is not being initialised with previous_opinion of 0.0",
        )
        self.assertEqual(
            empty_agent.personality,
            "neutral",
            "Agent -- empty agent is not being initialised with a neutral personality",
        )
        self.assertFalse(
            empty_agent.radicalised,
            "Agent -- empty agent is not being initialised as non-radicalised",
        )
        self.assertIsNone(
            empty_agent.rw_distributions,
            "Agent -- empty agent is not being initialised with rw_distributions equal to None",
        )
        self.assertIsNone(
            empty_agent.opinion_rw,
            "Agent -- empty agent is not being initialised with opinion_rw equal to None",
        )

    def test_agent_in(self) -> None:
        """
        Test that __in__() correctly sees an Agent in an iterable.
        """
        empty_agent: agt.Agent = agt.Agent()
        agent_iterable: list[agt.Agent] = []
        self.assertFalse(
            empty_agent in agent_iterable,
            "Agent -- __in__() is not returning False when the Agent is not in an iterable",
        )
        agent_iterable.append(empty_agent)
        self.assertTrue(
            empty_agent in agent_iterable,
            "Agent -- __in__() is not returning True when the Agent is in an iterable",
        )

    def test_initial_str(self) -> None:
        """
        Test that the __str__ representation is returning the expected output initially.
        """
        empty_agent: agt.Agent = agt.Agent()
        with self.assertRaises(AttributeError) as cm:
            str_repr: str = empty_agent.__str__()
        str_except = cm.exception
        self.assertEqual(
            str_except,
            AttributeError,
            "Agent -- __str__() with non-existing id and opinion attributes is not raising an AttributeError",
        )

    def test_init_args(self) -> None:
        """
        Test that the *args passed to an Agent __init__ are recognised correctly.
        """
        agent: agt.Agent = agt.Agent(
            "TestAgent",
            {"TestGraph": 0.2},
            0.44,
            True,
            ("impulsive", 0.9),
        )
        self.assertEqual(
            agent.id,
            "TestAgent",
            "Agent -- a string in __init__ *args is not being assigned to the id attribute",
        )
        self.assertEqual(
            agent.social_weightings,
            {"TestGraph": 0.2},
            "Agent -- a dictionary in __init__ *args is not being assigned to the social_weightings attribute",
        )
        self.assertEqual(
            agent.opinion,
            0.44,
            "Agent -- a float in __init__ *args is not being assigned to the opinion attribute",
        )
        self.assertTrue(
            agent.personal_benefit,
            "Agent -- a boolean in __init__ *args is not being assigned to the personal_benefit attribute",
        )
        self.assertEqual(
            agent.personality,
            "impulsive",
            "Agent -- the string variable of the tuple in __init__ *args is not being assigned to the personality attribute",
        )
        self.assertEqual(
            agent.social_susceptibility,
            0.9,
            "Agent -- the float variable of the tuple in __init__ *args is not being assigned to the social_susceptibility attribute",
        )

    def test_init_kwargs_override(self) -> None:
        """
        Test that existing Agent attributes passed in the __init__ **kwargs are correctly overriding the default values.
        """
        agent: agt.Agent = agt.Agent(
            personality="erratic",
            previous_opinion=0.99,
            radicalised=True,
        )
        self.assertEqual(
            agent.personality,
            "erratic",
            "Agent -- explicit personality attribute as a kwarg is not overwriting the default value",
        )
        self.assertEqual(
            agent.previous_opinion,
            0.99,
            "Agent -- explict previous_opinion attribute as a kwarg is not overwriting the default value",
        )
        self.assertTrue(
            agent.radicalised,
            "Agent -- explicit radicalised attribute as a kwarg is not overwriting the default value",
        )

    def test_init_kwargs_unseen(self) -> None:
        """
        Test that unseen **kwargs passed to an Agent __init__ are being assigned correctly.
        """
        agent: agt.Agent = agt.Agent(
            discontent=0.34,
            optimist=True,
            iterations_left={"TestGraph": 40},
            discontent_rw=(0.0, 0.01),
            current_level_of_influence="very strong",
        )
        self.assertHasAttr(
            agent,
            "discontent",
            "Agent -- unseen attribute passed in **kwargs is not being assigned correctly (float)",
        )
        self.assertIsInstance(
            agent.discontent,
            float,
            "Agent -- unseen attribute is not being stored as the correct type (float)",
        )
        self.assertEqual(
            agent.discontent,
            0.34,
            "Agent -- unseen attribute value is not being stored correctly (float)",
        )
        self.assertHasAttr(
            agent,
            "optimist",
            "Agent -- unseen attribute passed in **kwargs is not being assigned correctly (bool)",
        )
        self.assertIsInstance(
            agent.optimist,
            bool,
            "Agent -- unseen attribute is not being stored as the correct type (bool)",
        )
        self.assertTrue(
            agent.optimist,
            "Agent -- unseen attribute value is not being stored correctly (bool)",
        )
        self.assertHasAttr(
            agent,
            "iterations_left",
            "Agent -- unseen attribute passed in **kwargs is not being assigned correctly (dict)",
        )
        self.assertIsInstance(
            agent.iterations_left,
            dict,
            "Agent -- unseen attribute is not being stored as the correct type (dict)",
        )
        self.assertEqual(
            agent.iterations_left,
            {"TestGraph": 40},
            "Agent -- unseen attribute value is not being stored correctly (dict)",
        )
        self.assertHasAttr(
            agent,
            "discontent_rw",
            "Agent -- unseen attribute passed in **kwargs is not being assigned correctly (tuple)",
        )
        self.assertIsInstance(
            agent.discontent_rw,
            tuple,
            "Agent -- unseen attribute is not being stored as the correct type (tuple)",
        )
        self.assertEqual(
            agent.discontent_rw,
            (0.0, 0.01),
            "Agent -- unseen attribute value is not being stored correctly (tuple)",
        )
        self.assertHasAttr(
            agent,
            "current_level_of_influence",
            "Agent -- unseen attribute passed in **kwargs is not being assigned correctly (str)",
        )
        self.assertIsInstance(
            agent.current_level_of_influence,
            str,
            "Agent -- unseen attribute is not being stored as the correct type (str)",
        )
        self.assertEqual(
            agent.current_level_of_influence,
            "very strong",
            "Agent -- unseen attribute value is not being stored correctly (str)",
        )

    def test_init_kwargs_mixed(self) -> None:
        """
        Test that a mix of explicit and unseen kwargs are initialised correctly.
        """
        agent: agt.Agent = agt.Agent(
            opinion=0.24,
            personality="impulsive",
            discontent=0.04,
        )
        self.assertHasAttr(
            agent,
            "opinion",
            "Agent -- explicit opinion attribute in mixed **kwargs is not being assigned correctly",
        )
        self.assertHasAttr(
            agent,
            "discontent",
            "Agent -- unseen attribute in mixed **kwargs is not being assigned correctly",
        )
        self.assertEqual(
            agent.opinion,
            0.24,
            "Agent -- explicit opinion attribute in mixed **kwargs is not storing the value correctly",
        )
        self.assertEqual(
            agent.personality,
            "impulsive",
            "Agent -- explicit personality attribute in mixed **kwargs is not overwriting the default value",
        )
        self.assertEqual(
            agent.discontent,
            0.04,
            "Agent -- unseen attribute in mixed **kwargs is not storing its value correctly",
        )

    def test_init_mixed(self) -> None:
        """
        Test that __init__ with a mix of *args and **kwargs is working correctly.
        """
        agent: agt.Agent = agt.Agent(
            "TestAgent",
            {"TestGraph": 0.05},
            discontent=0.75,
        )
        self.assertHasAttr(
            agent,
            "id",
            "Agent -- id is not being assigned correctly from a mixed args and kwargs agent init",
        )
        self.assertHasAttr(
            agent,
            "discontent",
            "Agent -- unseen attribute is not being assigned correctly from a mixed args and kwargs agent init",
        )
        self.assertEqual(
            agent.id,
            "TestAgent",
            "Agent -- id is not being stored correctly from a mixed args and kwargs agent init",
        )
        self.assertEqual(
            agent.social_weightings,
            {"TestGraph": 0.05},
            "Agent -- social_weightings are not being stored correctly from a mixed args and kwargs agent init",
        )
        self.assertEqual(
            agent.discontent,
            0.75,
            "Agent -- unseen attribute is not being stored correctly from a mixed args and kwargs agent init",
        )

    def test_valid_str_normal(self) -> None:
        """
        Test that __str__ on a non-radicalised Agent with all required attributes returns the expected representation.
        """
        agent: agt.Agent = agt.Agent(
            "TestAgent",
            0.3,
        )
        str_repr: str = agent.__str__()
        self.assertEqual(
            str_repr,
            "Agent TestAgent which is not radicalised with an opinion value of 0.3",
        )

    def test_valid_str_radicalised(self) -> None:
        """
        Test that __str__ on a radicalised Agent with all required attributes returns the expected representation.
        """
        agent: agt.Agent = agt.Agent(
            "TestAgent",
            0.98,
            radicalised=True,
        )
        str_repr: str = agent.__str__()
        self.assertEqual(
            str_repr,
            "Agent TestAgent which is radicalised with an opinion value of 0.98",
        )

    def test_get_attribute_empty(self) -> None:
        """
        Test that get_attribute() on a non-existing Agent attribute raises a warning and returns None.
        """
        empty_agent: agt.Agent = agt.Agent()
        with self.assertWarns(UserWarning) as cm:
            attribute = empty_agent.get_attribute("foobar")
        ga_warning = cm.warning
        self.assertEqual(
            ga_warning.message,
            "WARNING: Attempting to get an Agent attribute (foobar) which doesn't exist.",
            "Agent -- get_attribute() on a non-existing attribute is not raising the expected warning",
        )
        self.assertIsNone(
            attribute,
            "Agent -- get_attribute() on a non-existing attribute is not returning None",
        )
