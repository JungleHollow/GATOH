from __future__ import annotations

import unittest as ut
from typing import override

import gatoh.agents as agt


class TestAgentObjectsEmpty(ut.TestCase):
    @override
    def setUp(self) -> None:
        # Create the empty agent
        self.agent: agt.Agent = agt.Agent()

    def test_update(self) -> None:
        """
        Test that calling update() on an empty Agent produces the expected error.
        """
        opinion_silenced: dict[str, bool] = {}
        negation_ocurred: bool = False
        with self.assertRaises(RuntimeError, msg="The agent for which an update is being attempted has not yet been initialised") as cm:
            self.agent.update(opinion_silenced, negation_ocurred)

    def test_opinion_silencing(self) -> None:
        """
        Test that calling opinion_silencing() on an empty Agent produces the expected error.
        """
        est_op_clim: float = 0.69
        with self.assertRaises(RuntimeError, msg="The agent for which opinion silencing is being determined has not yet been initialised") as cm:
            silencing: tuple[bool, float] = self.agent.opinion_silencing(est_op_clim)

    def test_opinion_negation(self) -> None:
        """
        Test that calling opinion_negation() on an empty Agent produces the expected error.
        """
        hier: str = "foo"
        abs_diff: float = 0.13
        thresh: float = 0.12
        with self.assertRaises(RuntimeError, msg="The agent for which opinion negation is being determined has not yet been initialised") as cm:
            opinion_negation: bool = self.agent.opinion_negation(hier, abs_diff, thresh)

    def test_deradicalisation(self) -> None:
        """
        Test that calling deradicalisation() on an empty Agent produces the expected error.
        """
        hier_changes: list[float] = []
        neighbour_bens: list[bool] = []
        thresh: float = 0.0
        with self.assertRaises(RuntimeError, msg="The agent for which deradicalisation is being determined has not yet been initialised") as cm:
            deradicalisation: bool = self.agent.deradicalisation(hier_changes, neighbour_bens, thresh)

    def test_radicalisation(self) -> None:
        """
        Test that calling radicalisation() on an empty Agent produces the expected error.
        """
        hier_changes: list[float] = []
        neighbour_bens: list[bool] = []
        thresh: float = 0.0
        with self.assertRaises(RuntimeError, msg="The agent for which radicalisation is being determined has not yet been initialised") as cm:
            radicalisation: bool = self.agent.radicalisation(hier_changes, neighbour_bens, thresh)

    def test_stochastic_opinion(self) -> None:
        """
        Test that calling stochastic_opinion() on an empty Agent produces the expected error.
        """
        opinion_rw: tuple[float, float] = (0.0, 0.0)
        with self.assertRaises(RuntimeError, msg="The agent for which the stochastic opinion is being determined has not yet been initialised") as cm:
            self.agent.stochastic_opinion(opinion_rw)

    def test_stochastic_benefit_change(self) -> None:
        """
        Test that calling stochastic_benefit_change() on an empty Agent produces the expected error.
        """
        with self.assertRaises(RuntimeError, msg="The agent for which a stochastic benefit change is being determined has not yet been initialised") as cm:
            self.agent.stochastic_benefit_change()

    @override
    def tearDown(self) -> None:
        del self.agent
