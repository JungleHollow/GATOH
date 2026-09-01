from __future__ import annotations

import unittest as ut

import gatoh.agents as agt


class TestAgentCreation(ut.TestCase):
    def test_draw_personality(self) -> None:
        """
        Test that the draw_personality utility function is working as intended.
        """
        drawn_personality: str = agt.draw_personality()
        self.assertIsInstance(
            drawn_personality,
            str,
            "Agent -- draw_personality is not returning a string result",
        )
        self.assertIn(
            drawn_personality,
            agt.PERSONALITIES,
            "Agent -- draw_personality is not drawing a valid personality value",
        )

    def test_generate_agent_invalid(self) -> None:
        """
        Test that generate_agent() with an invalid required parameter produces the expected error.
        """
        empty_agent: agt.Agent = agt.Agent()
        with self.assertRaises(TypeError, msg="One or more of the required parameters 'id', 'index', or 'hierarchies' are not of the appropriate data type") as cm:
            _ = empty_agent.generate_agent(
                "TEST0001",  # ID
                "0",  # index
                ["A"],  # hierarchies
            )

    def test_generate_agent_invalid_hierarchy(self) -> None:
        """
        Test that the secondary data type check for generate_agent will produce the expected error when appropriate.
        """
        empty_agent: agt.Agent = agt.Agent()
        with self.assertRaises(TypeError, msg="One or more of the hierarchy names input to Agent.generate_agent are not valid strings") as cm:
            _ = empty_agent.generate_agent(
                "TEST0001",  # ID
                0,  # index
                [1],  # hierarchies
            )

    def test_generate_agent(self) -> None:
        """
        Test that generate_agent() is working correctly.
        """
        empty_agent: agt.Agent = agt.Agent()
        _ = empty_agent.generate_agent(
            "TEST0001",  # ID
            0,  # index
            ["A"],  # hierarchies
        )
        self.assertEqual(
            empty_agent.id,
            "TEST0001",
            "Agent -- generate_agent is not setting the agent id correctly",
        )
        self.assertEqual(
            empty_agent.index,
            0,
            "Agent -- generate_agent is not setting the agent index correctly",
        )
        self.assertEqual(
            empty_agent.personality,
            "neutral",
            "Agent -- generate_agent without the optional personality argument is assigning a non-default personality",
        )
        self.assertIsInstance(
            empty_agent.personal_benefit,
            bool,
            "Agent -- generate_agent without the optional personal benefit argument is not assigning a boolean flag for personal_benefit",
        )
        self.assertIn(
            "A",
            empty_agent.social_weightings,
            "Agent -- generate_agent is not creating the appropriate hierarchy entries in social_weightings",
        )
        self.assertIsInstance(
            empty_agent.social_weightings["A"],
            float,
            "Agent -- generate_agent is not assigning a float value for the hierarchy entries in social_weightings",
        )
        self.assertTrue(
            0.0 <= empty_agent.social_weightings["A"] <= 1.0,
            "Agent -- generate_agent is not generating social_weightings values in the correct range",
        )
        self.assertIn(
            "A",
            empty_agent.is_silenced,
            "Agent -- generate_agent is not creating the appropriate hierarchy entries in is_silenced",
        )
        self.assertIsInstance(
            empty_agent.is_silenced["A"],
            bool,
            "Agent -- generate_agent is not assigning a boolean flag for the hierarchy entries in is_silenced",
        )
        self.assertFalse(
            empty_agent.is_silenced["A"],
            "Agent -- generate_agent is not initiating is_silenced as False",
        )
        self.assertIsNone(
            empty_agent.rw_distributions,
            "Agent -- generate_agent without the optional parameter explicit_rw is creating some value for rw_distributions",
        )
        self.assertIsNone(
            empty_agent.opinion_rw,
            "Agent -- generate_agent without the optional parameter explicit_opinion_rw is creating some value for opinion_rw",
        )
        self.assertIsInstance(
            empty_agent.opinion,
            float,
            "Agent -- generate_agent is assigning a non-float value to the agent opinion",
        )
        self.assertTrue(
            -1.0 <= empty_agent.opinion <= 1.0,
            "Agent -- generate_agent is generating an opinion value outside of the valid range",
        )
        self.assertIsInstance(
            empty_agent.social_susceptibility,
            float,
            "Agent -- generate_agent is assigning a non-float value to social_susceptibility",
        )
        self.assertTrue(
            0.0 <= empty_agent.social_susceptibility <= 1.0,
            "Agent -- generate_agent is generating a social_susceptibility value outside of the valid range",
        )
        self.assertEqual(
            empty_agent.previous_opinion,
            0.0,
            "Agent -- generate_agent is somehow touching the previous_opinion attribute"
        )
        # Given that this is a random generation, also include an edge case check for initialised radicalisation
        if empty_agent.opinion <= -agt.RADICALISATION_INIT_THRESH or agt.RADICALISATION_INIT_THRESH <= empty_agent.opinion:
            self.assertTrue(
                empty_agent.radicalised,
                "Agent -- generate_agent is not flagging the initialised agent as radicalised if the initial opinion is very strong"
            )

    def test_add_attribute_no_args(self) -> None:
        """
        Test that add_attribute with neither 'value', or 'parameters' and 'distribution' keyword arguments will produce the expected error.
        """
        empty_agent: agt.Agent = agt.Agent()
        with self.assertRaises(ValueError, msg="Either explicit 'value' or distribution and valid distribution parameters are expected when adding Agent attributes.") as cm:
            empty_agent.add_attribute("foo")

    def test_add_attribute_noverwrite(self) -> None:
        """
        Test that add_attribute on an existing attribute when overwrite=False produces the expected warning.
        """
        empty_agent: agt.Agent = agt.Agent()
        with self.assertWarns(UserWarning, msg="WARNING: Attempting to overwrite an existing Agent attribute (previous_opinion) without meaning to.") as cm:
            empty_agent.add_attribute("previous_opinion", value=0.1312, overwrite=False)
        self.assertEqual(
            empty_agent.previous_opinion,
            0.0,
            "Agent -- add_attribute when overwrite=False is changing the specified attribute's value",
        )

    def test_add_attribute_value(self) -> None:
        """
        Test that add_attribute with an explicit value is working as intended.
        """
        empty_agent: agt.Agent = agt.Agent()
        empty_agent.add_attribute("foobar", value=True)
        self.assertHasAttr(
            empty_agent,
            "foobar",
            "Agent -- add_attribute on a new attribute with an explicit value did not add the attribute to the agent",
        )
        self.assertIsInstance(
            empty_agent.foobar,
            bool,
            "Agent -- add_attribute on a new attribute with an explicit value did not add the attribute as the expected data type",
        )
        self.assertTrue(
            empty_agent.foobar,
            "Agent -- add_attribute on a new attribute with an explicit value did not store the attribute's value correctly",
        )

    def test_add_attribute_distribution(self) -> None:
        """
        Test that add_attribute with a distribution is working as intended.
        """
        empty_agent: agt.Agent = agt.Agent()
        empty_agent.add_attribute("fatigue", distribution="gaussian")
        self.assertHasAttr(
            empty_agent,
            "fatigue",
            "Agent -- add_attribute on a new attribute with a distribution did not add the attribute to the agent",
        )
        self.assertIsInstance(
            empty_agent.fatigue,
            float,
            "Agent -- add_attribute on a new attribute with a distribution did not assign a float value to the attribute",
        )
        self.assertTrue(
            0.0 <= empty_agent.fatigue <= 1.0,
            "Agent -- add_attribute on a new attribute with a distribution and default parameters did not generate a value in the valid range",
        )
