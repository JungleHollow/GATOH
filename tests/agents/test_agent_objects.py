from __future__ import annotations

import unittest as ut
from typing import override

import gatoh.agents as agt


class TestAgentObjects(ut.TestCase):
    @override
    def setUp(self) -> None:
        self.agent: agt.Agent = agt.Agent(
            "TEST0000",
            {"A": 0.2},
            0.44,
            True,
            ("social", 0.75),
        )

    def test_update_invalid_arg_one(self) -> None:
        """
        Test that update() with an invalid data type of opinion_silenced produces the expected error.
        """
        invalid_opinion_silenced: list[tuple[str, bool]] = [("A", True)]
        valid_negation_ocurred: bool = True
        with self.assertRaises(TypeError, msg="opinion_silenced must be a <string: boolean> dictionary -- the input is of an incorrect data type") as cm:
            self.agent.update(invalid_opinion_silenced, valid_negation_ocurred)
        self.assertFalse(
            self.agent.is_silenced["A"],
            "Agent -- update with an invalid data type for opinion_silenced is causing the agent's is_silenced to change",
        )
        self.assertAlmostEqual(
            self.agent.opinion,
            0.44,
            5,
            "Agent -- update with an invalid data type for opinion_silenced and negation_ocurred=True is causing the agent's opinion to change",
        )

    def test_update_invalid_arg_two(self) -> None:
        """
        Test that update() with an invalid data type of negation_ocurred produces the expected error.
        """
        valid_opinion_silenced: dict[str, bool] = {"A", True}
        invalid_negation_ocurred: str = "yes"
        with self.assertRaises(TypeError, msg="negation_ocurred must be a boolean -- the input value is not of the correct data type") as cm:
            self.agent.update(valid_opinion_silenced, invalid_negation_ocurred)
        self.assertFalse(
            self.agent.is_silenced["A"],
            "Agent -- update with a valid opinion_silenced but an invalid data type for negation_ocurred is causing the agent's is_silenced to change",
        )
        self.assertAlmostEqual(
            self.agent.opinion,
            0.44,
            5,
            "Agent -- update with an invalid data type for negation_ocurred is causing the agent's opinion to change",
        )

    def test_update_invalid_hierarchy(self) -> None:
        """
        Test that update() with valid argument data types but with a non-existent hierarchy present in opinion_silenced will produce the expected error.
        """
        invalid_hierarchy_silenced: dict[str, bool] = {"foo", True}
        negation_ocurred: bool = True
        with self.assertRaises(KeyError, msg="Hierarchy 'foo' has been passed in opinion_silenced, but does not exist in the agent's rw_distributions") as cm:
            self.agent.update(invalid_hierarchy_silenced, negation_ocurred)
        self.assertFalse(
            self.agent.is_silenced["A"],
            "Agent -- update with a non-existent hierarchy present in opinion_silenced is causing the agent's is_silenced to change",
        )
        self.assertAlmostEqual(
            self.agent.opinion,
            0.44,
            5,
            "Agent -- update with a non-existent hierarchy and a valid negation_ocurred=True is causing the agent's opinion to invert",
        )

    def test_update_silenced(self) -> None:
        """
        Test that update() with only silencing ocurring is working as intended.
        """
        opinion_silenced: dict[str, bool] = {"A", True}
        negation_ocurred: bool = False
        self.agent.update(opinion_silenced, negation_ocurred)
        self.assertTrue(
            self.agent.is_silenced["A"],
            "Agent -- update with only silencing ocurring is not changing the agent's is_silenced flag for a hierarchy",
        )
        self.assertAlmostEqual(
            self.agent.opinion,
            0.44,
            5,
            "Agent -- update with only silencing ocurring is causing the agent's opinion to change",
        )

    def test_update_negation(self) -> None:
        """
        Test that update() with only negation ocurring is working as intended.
        """
        opinion_silenced: dict[str, bool] = {"A": False}
        negation_ocurred: bool = True
        self.agent.update(opinion_silenced, negation_ocurred)
        self.assertFalse(
            self.agent.is_silenced["A"],
            "Agent -- update with only negation ocurring is changing a flag in the agent's is_silenced",
        )
        self.assertAlmostEqual(
            self.agent.opinion,
            -0.44,
            5,
            "Agent -- update with only negation ocurring is not causing the agent's opinion to invert",
        )

    def test_update_all(self) -> None:
        """
        Test that update() with both silencing and negation ocurring is working as intended.
        """
        opinion_silenced: dict[str, bool] = {"A": True}
        negation_ocurred: bool = True
        self.agent.update(opinion_silenced, negation_ocurred)
        self.assertTrue(
            self.agent.is_silenced["A"],
            "Agent -- update with both silencing and negation ocurring is not changing the agent's is_silenced flag for a hierarchy",
        )
        self.assertAlmostEqual(
            self.agent.opinion,
            -0.44,
            5,
            "Agent -- update with both silencing and negation ocurring is not causing the agent's opinion to invert",
        )

    def test_opinion_silencing_true_no_threshold(self) -> None:
        """
        Test that opinion_silencing() with no silencing_threshold passed will correctly report that silencing should occur.
        """
        est_op_climate: float = -0.8
        opinion_silencing: tuple[bool, float] = self.agent.opinion_silencing(est_op_climate)
        # Worked example:
        #   1. No explicit silencing_threshold was passed, so the agent's social_susceptibility of 0.75 is used as the threshold
        #   2. The agent's personality is "social", so the absolute difference is calculated as abs(estimated opinion climate - agent opinion)
        #   3. Therefore: absolute difference = abs(-0.8 - 0.44) = abs(-1.24) = 1.24
        #   4. 1.24 is larger than the threshold of 0.75, so silencing will occur
        self.assertIsInstance(
            opinion_silencing,
            tuple,
            "Agent -- opinion_silencing is not returning a tuple as the result (no threshold - true)",
        )
        self.assertIsInstance(
            opinion_silencing[0],
            bool,
            "Agent -- the first argument returned by opinion_silencing is not a boolean (no threshold - true)",
        )
        self.assertIsInstance(
            opinion_silencing[1],
            float,
            "Agent -- the second argument returned by opinion_silencing is not a float (no threshold - true)",
        )
        self.assertTrue(
            opinion_silencing[0],
            "Agent -- opinion_silencing is not correctly flagging when silencing should have ocurred (no threshold - true)",
        )
        self.assertAlmostEqual(
            opinion_silencing[1],
            1.24,
            5,
            "Agent -- opinion_silencing is not calculating and reporting the correct absolute difference value (no threshold - true)",
        )

    def test_opinion_silencing_false_no_threshold(self) -> None:
        """
        Test that opinion_silencing() with no silencing_threshold passed will correctly report that silencing should not occur.
        """
        est_op_climate: float = 0.69
        opinion_silencing: tuple[bool, float] = self.agent.opinion_silencing(est_op_climate)
        # Worked example:
        #   1. No explicit silencing_threshold was passed, so the agent's social susceptibility of 0.75 is used as the threshold
        #   2. The agent's personality is "social", so the absolute difference is calculated as abs(estimated opinion climate - agent opinion)
        #   3. Therefore: absolute difference = abs(0.69 - 0.44) = abs(0.25) = 0.25
        #   4. 0.25 is smaller than the threshold of 0.75, so silencing will not occur
        self.assertIsInstance(
            opinion_silencing,
            tuple,
            "Agent -- opinion_silencing is not returning a tuple as the result (no threshold - false)",
        )
        self.assertIsInstance(
            opinion_silencing[0],
            bool,
            "Agent -- the first argument returned by opinion_silencing is not a boolean (no threshold - false)",
        )
        self.assertIsInstance(
            opinion_silencing[1],
            float,
            "Agent -- the second argument returned by opinion_silencing is not a float (no threshold - false)",
        )
        self.assertFalse(
            opinion_silencing[0],
            "Agent -- opinion silencing is not correctly flagging when silencing should not have ocurred (no threshold - false)",
        )
        self.assertAlmostEqual(
            opinion_silencing[1],
            0.25,
            5,
            "Agent -- opinion_silencing is not calculating and reporting the correct absolute difference value (no threshold - false)",
        )

    def test_opinion_silencing_true(self) -> None:
        """
        Test that opinion_silencing() with a silencing_threshold will correctly report that silencing should occur.
        """
        est_op_climate: float = 0.69
        silencing_threshold: float = 0.01
        opinion_silencing: tuple[bool, float] = self.agent.opinion_silencing(est_op_climate, silencing_threshold=silencing_threshold)
        # Worked example:
        #   1. An explicit silencing_threshold of 0.01 was passed
        #   2. The agent's personality is "social", so the absolute difference is calculated as abs(estimated opinion climate - agent opinion)
        #   3. Therefore: absolute_difference = abs(0.69 - 0.44) = abs(0.25) = 0.25
        #   4. 0.25 is greater than the threshold of 0.01, so silencing will occur
        self.assertIsInstance(
            opinion_silencing,
            tuple,
            "Agent -- opinion_silencing is not returning a tuple as the result (threshold - true)",
        )
        self.assertIsInstance(
            opinion_silencing[0],
            bool,
            "Agent -- the first argument returned by opinion_silencing is not a boolean (threshold - true)",
        )
        self.assertIsInstance(
            opinion_silencing[1],
            float,
            "Agent -- the second argument returned by opinion_silencing is not a float (threshold - true)",
        )
        self.assertTrue(
            opinion_silencing[0],
            "Agent -- opinion_silencing is not correctly flagging when silencing should have ocurred (threshold - true)",
        )
        self.assertAlmostEqual(
            opinion_silencing[1],
            0.25,
            5,
            "Agent -- opinion_silencing is not calculating and reporting the correct absolute difference value (threshold - true)",
        )

    def test_opinion_silencing_false(self) -> None:
        """
        Test that opinion_silencing() with a silencing_threshold will correctly report that silencing should not occur.
        """
        est_op_climate: float = -0.8
        silencing_threshold: float = 1.99
        opinion_silencing: tuple[bool, float] = self.agent.opinion_silencing(est_op_climate, silencing_threshold=silencing_threshold)
        # Worked example:
        #   1. An explicit silencing_threshold of 1.99 was passed
        #   2. The agent's personality is "social", so the absolute difference is calculated as abs(estimated opinion climate - agent opinion)
        #   3. Therefore: absolute_difference = abs(-0.8 - 0.44) = abs(-1.24) = 1.24
        #   4. 1.24 is smaller than the threshold of 1.99, so silencing will not occur
        self.assertIsInstance(
            opinion_silencing,
            tuple,
            "Agent -- opinion_silencing is not returning a tuple as the result (threshold - false)",
        )
        self.assertIsInstance(
            opinion_silencing[0],
            bool,
            "Agent -- the first argument returned by opinion_silencing is not a boolean (threshold - false)",
        )
        self.assertIsInstance(
            opinion_silencing[1],
            float,
            "Agent -- the second argument returned by opinion_silencing is not a float (threshold - false)",
        )
        self.assertFalse(
            opinion_silencing[0],
            "Agent -- opinion_silencing is not correctly flagging when silencing should not occur (threshold - false)",
        )
        self.assertAlmostEqual(
            opinion_silencing[1],
            1.24,
            5,
            "Agent -- opinion_silencing is not calculating and reporting the correct absolute difference value (threshold - false)",
        )

    def test_opinion_silencing_radicalised(self) -> None:
        """
        Test that opinion_silencing() on a radicalised agent will return the expected result.
        """
        est_op_climate: float = -0.99
        silencing_threshold: float = 0.01,
        self.agent.change_radicalisation(True)
        opinion_silencing: tuple[bool, float] = self.agent.opinion_silencing(est_op_climate, silencing_threshold=silencing_threshold)
        self.assertIsInstance(
            opinion_silencing,
            tuple,
            "Agent -- opinion_silencing is not returning a tuple as the result (radicalised)",
        )
        self.assertIsInstance(
            opinion_silencing[0],
            bool,
            "Agent -- the first argument returned by opinion_silencing is not a boolean (radicalised)",
        )
        self.assertIsInstance(
            opinion_silencing[1],
            float,
            "Agent -- the second argument returned by opinion_silencing is not a float (radicalised)",
        )
        self.assertFalse(
            opinion_silencing[0],
            "Agent -- opinion_silencing is not correctly flagging when silencing should not occur (radicalised)",
        )
        self.assertAlmostEqual(
            opinion_silencing[1],
            0.0,
            5,
            "Agent -- opinion_silencing is not reporting an absolute difference of 0.0 (radicalised)",
        )

    def test_opinion_negation_invalid(self) -> None:
        """
        Test that opinion_negation() with an unseen hierarchy will raise the expected error.
        """
        abs_difference: float = 0.0
        thresh: float = 0.0
        with self.assertRaises(KeyError, msg="The hierarchy 'foo' does not exist in the agent's social_weightings") as cm:
            opinion_negation: bool = self.agent.opinion_negation("foo", abs_difference, thresh)

    def test_opinion_negation_false(self) -> None:
        """
        Test that opinion_negation() will correctly report that negation should not occur.
        """
        abs_difference: float = 0.03
        thresh: float = 1.99
        opinion_negation: bool = self.agent.opinion_negation("A", abs_difference, thresh)
        # Worked example:
        #   1. negation_strength is initialised as the reported absolute difference (0.03)
        #   2. Agent personality is "social", so negation_strength is modified by /= social_susceptibility * social_weightings[hierarchy]
        #   3. Therefore, negation_strength = 0.03 / (0.75 * 0.2) = 0.03 / 0.15 = 0.2
        #   4. The negation_strength of 0.2 is not above the threshold of 1.99, so negation does not occur
        self.assertIsInstance(
            opinion_negation,
            bool,
            "Agent -- opinion_negation is not returning a boolean result (negation=false)",
        )
        self.assertFalse(
            opinion_negation,
            "Agent -- opinion_negation is not correctly reporting that negation should not occur",
        )

    def test_opinion_negation_true(self) -> None:
        """
        Test that opinion_negation() will correctly report that negation should occur.
        """
        abs_difference: float = 1.47
        thresh: float = 6.9
        opinion_negation: bool = self.agent.opinion_negation("A", abs_difference, thresh)
        # Worked example:
        #   1. negation_strength is initialised as the reported absolute difference (1.47)
        #   2. Agent personality is "social", so negation_strength is modified by /= social_susceptibility * social_weightings[hierarchy]
        #   3. Therefore, negation_strength = 1.47 / (0.75 * 0.2) = 1.47 / 0.15 = 9.8
        #   4. The negation_strength of 9.8 is above the threshold of 6.9, so negation does occur
        self.assertIsInstance(
            opinion_negation,
            bool,
            "Agent -- opinion_negation is not returning a boolean result (negation=true)",
        )
        self.assertTrue(
            opinion_negation,
            "Agent -- opinion_negation is not correctly reporting that negation should occur",
        )

    def test_opinion_negation_radicalised(self) -> None:
        """
        Test that opinion_negation() on a radicalised agent will correctly report that negation does not occur.
        """
        abs_difference: float = 1.47
        thresh: float = 6.9
        self.agent.change_radicalisation(True)
        opinion_negation: bool = self.agent.opinion_negation("A", abs_difference, thresh)
        self.assertIsInstance(
            opinion_negation,
            bool,
            "Agent -- opinion_negation is not returning a boolean result (radicalised)",
        )
        self.assertFalse(
            opinion_negation,
            "Agent -- opinion_negation on a radicalised agent is not reporting that negation will not occur",
        )
