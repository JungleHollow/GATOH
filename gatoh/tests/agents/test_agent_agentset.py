from __future__ import annotations

import unittest as ut
import random as rd

import gatoh.agents.agents as agt


class TestAgentSet(ut.TestCase):
    def test_init(self) -> None:
        """
        Test that an empty initialisation of an AgentSet is returning the expected result.
        """
        empty_agentset: agt.AgentSet = agt.AgentSet()
        self.assertIsInstance(
            empty_agentset,
            agt.AgentSet,
            "The initialisation of an AgentSet is not returning an AgentSet object",
        )
        self.assertEqual(
            empty_agentset.agents,
            [],
            "The initialisation of an AgentSet is not creating an empty 'agents' list",
        )
        self.assertIsInstance(
            empty_agentset.random,
            rd.Random,
            "The initialisation of an AgentSet is not creating the random generator appropriately",
        )

    def test_add(self) -> None:
        """
        Test that AgentSet.add is working correctly.
        """
        agentset: agt.AgentSet = agt.AgentSet()
        new_agent: agt.Agent = agt.Agent()
        agent_index: int = agentset.add(new_agent)
        self.assertIsInstance(
            agent_index,
            int,
            "AgentSet add() is not returning an integer index when called",
        )
        self.assertEqual(
            agent_index,
            1,
            "AgentSet add() is not reporting the correct index for the newly added Agent",
        )
        self.assertEqual(
            agentset.agents[agent_index - 1],
            new_agent,
            "AgentSet add() is not correctly adding the new Agent object to 'agents'",
        )
        self.assertEqual(
            new_agent.index,
            agent_index,
            "AgentSet add() is not setting the Agent's 'index' attribute correctly",
        )

    def test_len(self) -> None:
        """
        Test that the __len__ override for an AgentSet is working correctly.
        """
        agentset: agt.AgentSet = agt.AgentSet()
        self.assertEqual(
            len(agentset),
            0,
            "The __len__ of an empty AgentSet is not reporting 0",
        )
        empty_agent: agt.Agent = agt.Agent()
        _ = agentset.add(empty_agent)
        self.assertEqual(
            len(agentset),
            1,
            "The __len__ of a non-empty AgentSet is not reporting the correct number of agents",
        )

    def test_in(self) -> None:
        """
        Test that the __in__ override for an AgentSet is working correctly.
        """
        agentset: agt.AgentSet = agt.AgentSet()
        new_agent: agt.Agent = agt.Agent()
        _ = agentset.add(new_agent)
        indirect_call: bool = new_agent in agentset
        direct_call: bool = agentset.__in__(new_agent)
        self.assertEqual(
            indirect_call,
            direct_call,
            "The direct and indirect calls of __in__ for an AgentSet are not reporting the same status",
        )
        self.assertTrue(
            indirect_call,
            "AgentSet __in__ on a valid Agent is not returning True",
        )
        invalid_agent: agt.Agent = agt.Agent()
        self.assertFalse(
            invalid_agent in agentset,
            "AgentSet __in__ on an invalid Agent is not returning False",
        )

    def test_contains(self) -> None:
        """
        Test that the __contains__ override for an AgentSet is working correctly.
        """
        agentset: agt.AgentSet = agt.AgentSet()
        new_agent: agt.Agent = agt.Agent()
        _ = agentset.add(new_agent)
        contains_call: bool = agentset.__contains__(new_agent)
        self.assertTrue(
            contains_call,
            "AgentSet __contains__ on a valid Agent is not returning True",
        )
        invalid_agent: agt.Agent = agt.Agent()
        self.assertFalse(
            agentset.__contains__(invalid_agent),
            "AgentSet __contains__ on an invalid Agent is not returning False",
        )
