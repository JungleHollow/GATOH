from __future__ import annotations

import unittest as ut

import gatoh.agents.agents as agt


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
