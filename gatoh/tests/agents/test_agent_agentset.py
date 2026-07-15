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
            0,
            "AgentSet add() is not reporting the correct index for the newly added Agent",
        )
        self.assertEqual(
            agentset.agents[agent_index],
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

    def test_getitem(self) -> None:
        """
        Test that the __getitem__ override for an AgentSet is working correctly.
        """
        agentset: agt.AgentSet = agt.AgentSet()
        new_agent: agt.Agent = agt.Agent()
        second_agent: agt.Agent = agt.Agent()
        _ = agentset.add(new_agent)
        _ = agentset.add(second_agent)
        get_agent: agt.Agent | list[agt.Agent] = agentset.__getitem__(new_agent.index)
        self.assertIsInstance(
            get_agent,
            agt.Agent,
            "AgentSet __getitem__() on a valid, single index is not returning an Agent object",
        )
        self.assertEqual(
            new_agent,
            get_agent,
            "AgentSet __getitem__() is not returning the expected Agent object"
        )

    def test_update_indices(self) -> None:
        """
        Test that the update_indices() method is working correctly.
        """
        agentset: agt.AgentSet = agt.AgentSet()
        new_agent: agt.Agent = agt.Agent()
        _ = agentset.add(new_agent)
        agentset.agents[0].index = 404
        self.assertEqual(
            agentset.agents[0].index,
            404,
            "Direct setting of agent 'index' attributes in the AgentSet is not persistent",
        )
        agentset.update_indices()
        self.assertEqual(
            agentset.agents[0].index,
            0,
            "AgentSet update_indices() is not correctly updating the indices of contained Agent objects",
        )

    def test_discard_simple(self) -> None:
        """
        Test that the core functionality of discard() is working correctly.
        """
        agentset: agt.AgentSet = agt.AgentSet()
        agent_one: agt.Agent = agt.Agent()
        agent_two: agt.Agent = agt.Agent()
        _ = agentset.add(agent_one)
        discard_one: bool = agentset.discard(agent_one)
        self.assertEqual(
            len(agentset),
            0,
            "AgentSet discard() is not correctly removing Agent objects (simple case)",
        )
        self.assertTrue(
            discard_one,
            "AgentSet discard() is not reporting that an Agent was removed correctly (simple case)",
        )
        discard_two: bool = agentset.discard(agent_two)
        self.assertFalse(
            discard_two,
            "AgentSet discard() is not reporting that an Agent was not removed correctly (simple case)",
        )

    def test_discard_complex(self) -> None:
        """
        Test that a more complex case of discard() is working correctly.
        """
        agentset: agt.AgentSet = agt.AgentSet()
        external_agents: list[agt.Agent] = []
        for _ in range(10):
            new_agent: agt.Agent = agt.Agent()
            external_agents.append(new_agent)
            _ = agentset.add(new_agent)
        self.assertEqual(
            len(agentset),
            10,
            "AgentSet -- adding multiple Agents in a loop is not working correctly",
        )
        discard_call: bool = agentset.discard(external_agents[4])
        self.assertEqual(
            len(agentset),
            9,
            "AgentSet discard() is not correctly removing Agent objects (complex case)",
        )
        self.assertTrue(
            discard_call,
            "AgentSet discard() is not reporting that an Agent was removed correctly (complex case)"
        )
        self.assertNotIn(
            external_agents[4],
            agentset,
            "AgentSet discard() did not remove the expected Agent from the agentset",
        )
        for idx, agent in enumerate(agentset):
            self.assertEqual(
                agent.index,
                idx,
                "AgentSet discard() did not correctly call update_indices() after removing the Agent",
            )

    def test_agent_at_index_invalid(self) -> None:
        """
        Test that agent_at_index() with an invalid index raises the expected warning.
        """
        agentset: agt.AgentSet = agt.AgentSet()
        with self.assertWarns(UserWarning) as cm:
            agent_return: agt.Agent | None = agentset.agent_at_index(44)
        self.assertEqual(
            cm.warning,
            UserWarning,
            "AgentSet -- agent_at_index() with an invalid index is not raising a UserWarning",
        )
        self.assertEqual(
            cm.msg,
            "WARNING: Index 44 is out of bounds for the AgentSet. Only 0 Agents have been created.",
            "AgentSet -- agent_at_index() with an invalid index is not producing the expected warning message",
        )
        self.assertIsNone(
            agent_return,
            "AgentSet -- agent_at_index() with an invalid index is not returning None",
        )

    def test_agent_at_index(self) -> None:
        """
        Test that agent_at_index() with a valid index is working correctly.
        """
        agentset: agt.AgentSet = agt.AgentSet()
        new_agent: agt.Agent = agt.Agent()
        _ = agentset.add(new_agent)
        agent_return: agt.Agent | None = agentset.agent_at_index(0)
        self.assertIsInstance(
            agent_return,
            agt.Agent,
            "AgentSet -- agent_at_index() with a valid index is not returning an Agent object",
        )
        self.assertEqual(
            new_agent,
            agent_return,
            "AgentSet -- agent_at_index() with a valid index is not returning the correct Agent object",
        )

    def test_get_agent_by_id_invalid(self) -> None:
        """
        Test that get_agent_by_id() with an invalid ID raises the expected error.
        """
        agentset: agt.AgentSet = agt.AgentSet()
        with self.assertRaises(KeyError) as cm:
            agent_return: agt.Agent | None = agentset.get_agent_by_id("foo")
        self.assertEqual(
            cm.exception,
            KeyError,
            "AgentSet -- get_agent_by_id() with an invalid ID is not raising a KeyError",
        )
        self.assertEqual(
            cm.msg,
            "The Agent with id 'foo' does not exist in the AgentSet -- unable to return an Agent object.",
            "AgentSet -- get_agent_by_id() with an invalid ID is not producing the expected error message",
        )

    def test_get_agent_by_id(self) -> None:
        """
        Test that get_agent_by_id() with a valid ID is working correctly.
        """
        agentset: agt.AgentSet = agt.AgentSet()
        new_agent: agt.Agent = agt.Agent("bar")
        _ = agentset.add(new_agent)
        agent_return: agt.Agent | None = agentset.get_agent_by_id("bar")
        self.assertIsInstance(
            agent_return,
            agt.Agent,
            "AgentSet -- get_agent_by_id() with a valid ID is not returning an Agent object",
        )
        self.assertEqual(
            new_agent,
            agent_return,
            "AgentSet -- get_agent_by_id() with a valid ID is not returning the correct Agent",
        )

    def test_get_index_invalid(self) -> None:
        """
        Test that get_index() with an invalid Agent raises the expected error.
        """
        agentset: agt.AgentSet = agt.AgentSet()
        new_agent: agt.Agent = agt.Agent("foo")
        with self.assertRaises(KeyError) as cm:
            index_return: int = agentset.get_index(new_agent)
        self.assertEqual(
            cm.exception,
            KeyError,
            "AgentSet -- get_index() with an invalid Agent is not raising a KeyError",
        )
        self.assertEqual(
            cm.msg,
            "The agent foo does not exist in the AgentSet -- unable to return an index.",
            "AgentSet -- get_index() with an invalid Agent is not producing the expected error message",
        )

    def test_get_index(self) -> None:
        """
        Test that get_index() with a valid Agent returns the correct index.
        """
        agentset: agt.AgentSet = agt.AgentSet()
        new_agent: agt.Agent = agt.Agent("foo")
        _ = agentset.add(new_agent)
        index_return: int = agentset.get_index(new_agent)
        self.assertIsInstance(
            index_return,
            int,
            "AgentSet -- get_index() with a valid Agent is not returning an integer index",
        )
        self.assertEqual(
            index_return,
            0,
            "AgentSet -- get_index() with a valid Agent is not returning the correct agent index",
        )

    def test_discard_index_invalid(self) -> None:
        """
        Test that discard_index() with an invalid index is working correctly.
        """
        agentset: agt.AgentSet = agt.AgentSet()
        for i in range(4):
            new_agent: agt.Agent = agt.Agent(f"{i}")
            _ = agentset.add(new_agent)
        index_discarded: bool = agentset.discard_index(4444)
        self.assertIsInstance(
            index_discarded,
            bool,
            "AgentSet -- discard_index() with an invalid index is not returning a boolean",
        )
        self.assertFalse(
            index_discarded,
            "AgentSet -- discard_index() with an invalid index is reporting that an index was removed",
        )
        self.assertEqual(
            len(agentset),
            4,
            "AgentSet -- discard_index() with an invalid index is removing one or more Agents from the AgentSet",
        )

    def test_discard_index(self) -> None:
        """
        Test that discard_index() with a valid index is working correctly.
        """
        agentset: agt.AgentSet = agt.AgentSet()
        external_agents: list[agt.Agent] = []
        for i in range(4):
            new_agent: agt.Agent = agt.Agent(f"{i}")
            external_agents.append(new_agent)
            _ = agentset.add(new_agent)
        index_discarded: bool = agentset.discard_index(2)
        self.assertIsInstance(
            index_discarded,
            bool,
            "AgentSet -- discard_index() with a valid index is not returning a boolean",
        )
        self.assertTrue(
            index_discarded,
            "AgentSet -- discard_index() with a valid index is not reporting that an index was removed",
        )
        self.assertEqual(
            len(agentset),
            3,
            "AgentSet -- discard_index() with a valid index is not actually removing an Agent object",
        )
        self.assertNotIn(
            external_agents[2],
            agentset,
            "AgentSet -- discard_index() with a valid index is not removing the correct Agent object",
        )
        for idx, agent in enumerate(agentset):
            self.assertEqual(
                agent.index,
                idx,
                "AgentSet -- discard_index() with a valid index is not correctly calling update_indices()",
            )

    def test_remove_invalid(self) -> None:
        """
        Test that remove() with an invalid Agent raises the expected error.
        """
        agentset: agt.AgentSet = agt.AgentSet()
        for i in range(4):
            new_agent: agt.Agent = agt.Agent(f"{i}")
            _ = agentset.add(new_agent)
        invalid_agent: agt.Agent = agt.Agent("foo")
        with self.assertRaises(KeyError) as cm:
            agent_removed: bool = agentset.remove(invalid_agent)
        self.assertEqual(
            cm.exception,
            KeyError,
            "AgentSet -- remove() with an invalid Agent is not raising a KeyError",
        )
        self.assertEqual(
            cm.msg,
            "Tried to remove an Agent with id foo that doesn't exist in the AgentSet",
            "AgentSet -- remove() with an invalid Agent is not producing the expected error message",
        )
        self.assertEqual(
            len(agentset),
            4,
            "AgentSet -- remove() with an invalid Agent is removing one or more Agents despite raising an error",
        )

    def test_remove(self) -> None:
        """
        Test that remove() with a valid Agent is working correctly.
        """
        agentset: agt.AgentSet = agt.AgentSet()
        external_agents: list[agt.Agent] = []
        for i in range(4):
            new_agent: agt.Agent = agt.Agent(f"{i}")
            external_agents.append(new_agent)
            _ = agentset.add(new_agent)
        agent_removed: bool = agentset.remove(external_agents[2])
        self.assertIsInstance(
            agent_removed,
            bool,
            "AgentSet -- remove() with a valid Agent is not returning a boolean",
        )
        self.assertTrue(
            agent_removed,
            "AgentSet -- remove() with a valid Agent is not reporting that an Agent was removed",
        )
        self.assertEqual(
            len(agentset),
            3,
            "AgentSet -- remove() with a valid Agent is not actually removing an Agent object from the AgentSet",
        )
        self.assertNotIn(
            external_agents[2],
            agentset,
            "AgentSet -- remove() with a valid Agent is not removing the correct Agent object",
        )
