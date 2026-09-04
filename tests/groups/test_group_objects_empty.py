from __future__ import annotations

import unittest as ut
from typing import override

import gatoh.groups as grp


class TestGroupObjectsEmpty(ut.TestCase):
    @override
    def setUp(self) -> None:
        # Create the empty group
        self.group: grp.Group = grp.Group()

    def test_update(self) -> None:
        """
        Test that calling update() on an empty Group will raise the expected error.
        """
        with self.assertRaises(AttributeError, msg="The group for which an update is being attempted has not yet been initialised") as cm:
            update_info: tuple[str, dict[str, bool]] = self.group.update(0.0, False)

    def test_opinion_silencing(self) -> None:
        """
        Test that calling opinion_silencing() on an empty Group will raise the expected error.
        """
        with self.assertRaises(AttributeError, msg="The group for which opinion silencing is being determined has not yet been initialised") as cm:
            silencing_info: tuple[bool, float] = self.group.opinion_silencing(0.0)

    def test_opinion_negation(self) -> None:
        """
        Test that calling opinion_negation() on an empty Group will raise the expected error.
        """
        with self.assertRaises(AttributeError, msg="The group for which opinion negation is being determined has not yet been initialised") as cm:
            negation_ocurred: bool = self.group.opinion_negation(0.0, 0.0)

    def test_stochastic_opinion(self) -> None:
        """
        Test that stochastic_opinion() on an empty Group will raise the expected error.
        """
        with self.assertRaises(RuntimeError, msg="The group for which the stochastic opinion is being determined has not yet been initialised") as cm:
            opinion_info: tuple[str, float] = self.group.stochastic_opinion((0.0, 0.0))

    def test_stochastic_benefit_change(self) -> None:
        """
        Test that stochastic_benefit_change() on an empty Group will raise the expected error.
        """
        with self.assertRaises(RuntimeError, msg="The group for which a stochastic beenfit change is being determined has not yet been initialised") as cm:
            benefit_info: dict[str, bool] = self.group.stochastic_benefit_change()

    def test_stochastic_radicalisation_change(self) -> None:
        """
        Test that stochastic_radicalisation_change() on an empty Group will raise the expected error.
        """
        with self.assertRaises(RuntimeError, msg="The group for which a stochastic radicalisation change is being determined has not yet been initialised") as cm:
            radicalisation_info: dict[str, bool] = self.group.stochastic_radicalisation_change()

    def test_stochastic_silencing_change(self) -> None:
        """
        Test that stochastic_silencing_change() on an empty Group will raise the expected error.
        """
        with self.assertRaises(RuntimeError, msg="The group for which a stochastic silencing change is being determined has not yet been initialised") as cm:
            silencing_info: dict[str, bool] = self.group.stochastic_silencing_change()

    @override
    def tearDown(self) -> None:
        del self.group