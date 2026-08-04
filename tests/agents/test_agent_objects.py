from __future__ import annotations

import unittest as ut
from typing import override

from pandas.core.ops import invalid

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

    def test_deradicalisation_nonradicalised(self) -> None:
        """
        Test that a valid deradicalisation() call on a non-radicalised agent will correctly report that deradicalisation cannot occur.
        """
        hier_changes: list[float] = [0.99]
        neighbour_benefits: list[bool] = [True, True, True, False]
        thresh: float = 0.01
        deradicalisation: bool = self.agent.deradicalisation(hier_changes, neighbour_benefits, thresh)
        self.assertIsInstance(
            deradicalisation,
            bool,
            "Agent -- a valid call to deradicalisation did not return a boolean result (nonradicalised)",
        )
        self.assertFalse(
            deradicalisation,
            "Agent -- a valid call to deradicalisation with a nonradicalised agent did not return False",
        )

    def test_deradicalisation_invalid_changes(self) -> None:
        """
        Test that a call to deradicalisation() with an invalid hierarchy_changes data type will produce the expected error.
        """
        self.agent.change_radicalisation(True)
        invalid_changes: dict[str, float] = {"A": 0.99}
        neighbour_benefits: list[bool] = [True, True, True, False]
        thresh: float = 0.01
        with self.assertRaises(TypeError, msg="hierarchy_changes must be a list") as cm:
            deradicalisation: bool = self.agent.deradicalisation(invalid_changes, neighbour_benefits, thresh)

    def test_deradicalisation_invalid_changes_value(self) -> None:
        """
        Test that a call to deradicalisation() with a hierarchy_changes list containing an invalid value will produce the expected error.
        """
        self.agent.change_radicalisation(True)
        invalid_changes: list[float] = [0.13, "0.12"]
        neighbour_benefits: list[bool] = [True, True, True, False]
        thresh: float = 0.01
        with self.assertRaises(TypeError, msg="One or more of the items in hierarchy_changes is of an invalid data type -- all must be floats") as cm:
            deradicalisation: bool = self.agent.deradicalisation(invalid_changes, neighbour_benefits, thresh)

    def test_deradicalisation_invalid_benefits(self) -> None:
        """
        Test that a call to deradicalisation() with an invalid neighbour_benefits data type will produce the expected error.
        """
        self.agent.change_radicalisation(True)
        hier_changes: list[float] = [0.99]
        invalid_benefits: dict[str, bool] = {"foo": True, "bar": True, "foobar": True, "barfoo": False}
        thresh: float = 0.01
        with self.assertRaises(TypeError, msg="neighbour_benefits must be a list") as cm:
            deradicalisation: bool = self.agent.deradicalisation(hier_changes, invalid_benefits, thresh)

    def test_deradicalisation_invalid_benefits_value(self) -> None:
        """
        Test that a call to deradicalisation() with a neighbour_benefits list containing an invalid value will produce the expected error.
        """
        self.agent.change_radicalisation(True)
        hier_changes: list[float] = [0.99]
        invalid_benefits: list[bool] = [True, True, "True", False]
        thresh: float = 0.01
        with self.assertRaises(TypeError, msg="One or more of the items in neighbour_benefits is of an invalid data type -- all must be boolean") as cm:
            deradicalisation: bool = self.agent.deradicalisation(hier_changes, invalid_benefits, thresh)

    def test_deradicalisation_invalid_thresh(self) -> None:
        """
        Test that a call to deradicalisation() with an invalid threshold data type will produce the expected error.
        """
        self.agent.change_radicalisation(True)
        hier_changes: list[float] = [0.99]
        neighbour_benefits: list[bool] = [True, True, True, False]
        invalid_thresh: str = "0.99"
        with self.assertRaises(TypeError, msg="threshold must be a float") as cm:
            deradicalisation: bool = self.agent.deradicalisation(hier_changes, neighbour_benefits, invalid_thresh)

    def test_deradicalisation_true(self) -> None:
        """
        Test that a valid call to deradicalisation() will correctly report that deradicalisation has ocurred.
        """
        self.agent.change_radicalisation(True)
        hier_changes: list[float] = [-0.99]
        neighbour_benefits: list[bool] = [True, True, True, False]
        thresh: float = 0.01
        deradicalisation: bool = self.agent.deradicalisation(hier_changes, neighbour_benefits, thresh)
        # Worked example:
        #   1. The agent's absolute opinion is abs(0.44) = 0.44
        #   2. Aggregate benefit = count(neighbour_benefits == True) / len(neighbour_benefits) = 3 / 4 = 0.75
        #   3. The agent's personality is "social", so the following process is used to determine deradicalisation:
        #       3.1. For each change in hierarchy_changes, determine the absolute change, and if the change agrees with the agent's opinion
        #       3.2. The absolute change is abs(-0.99) = 0.99; 0.44 and -0.99 are not of the same sign (i.e. the change disagrees with the agent's opinion)
        #       3.3. If the absolute change is greater than the agent's social_susceptibility, and the change disagrees with the opinion, immediately flag deradicalisation
        #   4. In this case, the absolute change (0.99) is greater than social_suceptibility (0.75), and the change and opinion are not of the same sign (i.e. not in agreeance)
        #   5. Therefore, deradicalisation will occur
        self.assertIsInstance(
            deradicalisation,
            bool,
            "Agent -- a valid call to deradicalisation is not returning a boolean result (deradicalisation=true)",
        )
        self.assertTrue(
            deradicalisation,
            "Agent -- a valid call to deradicalisation is not correctly reporting that deradicalisation has ocurred",
        )
        self.assertFalse(
            self.agent.radicalised,
            "Agent -- a valid call to deradicalisation in which deradicalisation has ocurred did not set the agent's radicalised attribute to False",
        )

    def test_deradicalisation_false(self) -> None:
        """
        Test that a valid call to deradicalisation() will correctly report that deradicalisation has not ocurred.
        """
        self.agent.change_radicalisation(True)
        hier_changes: list[float] = [0.99]
        neighbour_benefits: list[bool] = [True, True, True, False]
        thresh: float = 0.01
        deradicalisation: bool = self.agent.deradicalisation(hier_changes, neighbour_benefits, thresh)
        # Worked example:
        #   1. The agent's absolute opinion is abs(0.44) = 0.44
        #   2. Aggregate benefit = count(neighbour_benefits == True) / len(neighbour_benefits) = 3 / 4 = 0.75
        #   3. The agent's personality is "social", so the following process is used to determine deradicalisation:
        #       3.1. For each change in hierarchy_changes, determine the absolute change, and if the change agrees with the agent's opinion
        #       3.2. The absolute change is abs(0.99) = 0.99; 0.44 and 0.99 are of the same sign (i.e. the change agrees with the agent's opinion)
        #       3.3. The absolute change is greater than the threshold, but there is agreeance, so instead the absolute change is summed to a total,
        #             and the sign of the change is summed to a total (+ 1.0 for agreeance, - 1.0 for non-agreeance)
        #   4. No immediate deradicalisation was flagged, so instead the following check is made:
        #       (sum_changes >=social_susceptibility * len(hierarchy_changes) * SOCIAL_THRESHOLD_MODIFIER) and (total_sign <= 0.0)
        #   5. Following: (0.99 >= 0.75 * 1 * 0.5) and (1.0 <= 0) = True and False = False
        #   6. Therefore, deradicalisation will not occur
        self.assertIsInstance(
            deradicalisation,
            bool,
            "Agent -- a valid call to deradicalisation is not returning a boolean result (deradicalisation=false)",
        )
        self.assertFalse(
            deradicalisation,
            "Agent -- a valid call to deradicalisation is not correctly reporting that deradicalisation does not occur",
        )
        self.assertTrue(
            self.agent.radicalised,
            "Agent -- a valid call to deradicalisation in which deradicalisation did not occur has set the agent's radicalised attribute to False",
        )

    def test_radicalisation_radicalised(self) -> None:
        """
        Test that a valid radicalisation() call on a radicalised agent will correctly report that radicalisation cannot occur.
        """
        self.agent.change_radicalisation(True)
        hier_changes: list[float] = [0.99]
        neighbour_benefits: list[bool] = [True, True, True, False]
        thresh: float = 0.01
        radicalisation: bool = self.agent.radicalisation(hier_changes, neighbour_benefits, thresh)
        self.assertIsInstance(
            radicalisation,
            bool,
            "Agent -- a valid call to radicalisation did not return a boolean result (radicalised)",
        )
        self.assertFalse(
            radicalisation,
            "Agent -- a valid call to radicalisation on a radicalised agent did not return False",
        )

    def test_radicalisation_invalid_changes(self) -> None:
        """
        Test that a call to radicalisation() with an invalid hierarchy_changes data type will produce the expected error.
        """
        invalid_changes: dict[str, float] = {"A": 0.99}
        neighbour_benefits: list[bool] = [True, True, True, False]
        thresh: float = 0.01
        with self.assertRaises(TypeError, msg="hierarchy_changes must be a list") as cm:
            radicalisation: bool = self.agent.radicalisation(invalid_changes, neighbour_benefits, thresh)

    def test_radicalisation_invalid_changes_value(self) -> None:
        """
        Test that a call to radicalisation() with a hierarchy_changes list containing an invalid value will produce the expected error.
        """
        invalid_changes: list[float] = [0.13, "0.12"]
        neighbour_benefits: list[bool] = [True, True, True, False]
        thresh: float = 0.01
        with self.assertRaises(TypeError, msg="One or more of the items in hierarchy_changes is of an invalid data type -- all must be floats") as cm:
            radicalisation: bool = self.agent.radicalisation(invalid_changes, neighbour_benefits, thresh)

    def test_radicalisation_invalid_benefits(self) -> None:
        """
        Test that a call to radicalisation() with an invalid neighbour_benefits data type will produce the expected error.
        """
        hier_changes: list[float] = [0.99]
        invalid_benefits: dict[str, bool] = {"foo": True, "bar": True, "foobar": True, "barfoo": False}
        thresh: float = 0.01
        with self.assertRaises(TypeError, msg="neighbour_benefits must be a list") as cm:
            radicalisation: bool = self.agent.radicalisation(hier_changes, invalid_benefits, thresh)

    def test_radicalisation_invalid_benefits_value(self) -> None:
        """
        Test that a call to radicalisation() with a neighbour_benefits list containing an invalid value will produce the expected error.
        """
        hier_changes: list[float] = [0.99]
        invalid_benefits: list[bool] = [True, True, "True", False]
        thresh: float = 0.01
        with self.assertRaises(TypeError, msg="One or more of the items in neighbour_benefits is of an invalid data type -- all must be booleans") as cm:
            radicalisation: bool = self.agent.radicalisation(hier_changes, invalid_benefits, thresh)

    def test_radicalisation_invalid_thresh(self) -> None:
        """
        Test that a call to radicalisation() with an invalid threshold data type will produce the expected error.
        """
        hier_changes: list[float] = [0.99]
        neighbour_benefits: list[bool] = [True, True, True, False]
        thresh: str = "0.01"
        with self.assertRaises(TypeError, msg="threshold must be a float") as cm:
            radicalisation: bool = self.agent.radicalisation(hier_changes, neighbour_benefits, thresh)

    def test_radicalisation_true(self) -> None:
        """
        Test that a valid call to radicalisation() will correctly report that radicalisation occurs.
        """
        hier_changes: list[float] = [0.99]
        neighbour_benefits: list[bool] = [True, True, True, False]
        thresh: float = 0.01
        radicalisation: bool = self.agent.radicalisation(hier_changes, neighbour_benefits, thresh)
        # Worked example:
        #   1. The agent's absolute opinion is abs(0.44) = 0.44
        #   2. Aggregate benefit = count(neighbour_benefits == True) / len(neighbour_benefits) = 3 / 4 = 0.75
        #   3. The agent's personality is "social", so the following process is used to determine radicalisation:
        #       3.1. For each change in hierarchy_changes, determine the absolute change, and if the change agrees with the agent's opinion
        #       3.2. The absolute change is abs(0.99) = 0.99; 0.44 and 0.99 are of the same sign (i.e. the change agrees with the agent's opinion)
        #       3.3. If the absolute change is greater than the agent's social_susceptibility, and the change agrees with the opinion, immediately flag radicalisation
        #   4. In this case, the absolute change (0.99) is greater than social_susceptibility (0.75), and the change and opinion are of the same sign (i.e. in agreeance)
        #   5. Therefore, radicalisation will occur
        self.assertIsInstance(
            radicalisation,
            bool,
            "Agent -- a valid call to radicalisation is not returning a boolean result (radicalisation=true)",
        )
        self.assertTrue(
            radicalisation,
            "Agent -- a valid call to radicalisation is not correctly reporting that radicalisation has ocurred",
        )
        self.assertTrue(
            self.agent.radicalised,
            "Agent -- a valid call to radicalisation in which radicalisation has ocurred did not set the agent's radicalised attribute to True",
        )

    def test_radicalisation_false(self) -> None:
        """
        Test that a valid call to radicalisation() will correctly report that radicalisation has not ocurred.
        """
        hier_changes: list[float] = [-0.99]
        neighbour_benefits: list[bool] = [True, True, True, False]
        thresh: float = 0.01
        radicalisation: bool = self.agent.radicalisation(hier_changes, neighbour_benefits, thresh)
        # Worked example:
        #   1. The agent's absolute opinionis abs(0.44) = 0.44
        #   2. Aggregate benefit = count(neighbour_benefits == True) / len(neighbour_benefits) = 3 / 4 = 0.75
        #   3. The agent's personality is "social", so the following process is used to determine radicalisation:
        #       3.1. For each change in hierarchy_changes, determine the absolute change, and if the change agrees with the agent's opinion
        #       3.2. The absolute change is abs(-0.99) = 0.99; 0.44 and -0.99 are not of the same sign (i.e. the change does not agree with the agent's opinion)
        #       3.3. The absolute change is greater than the threshold, but there is no agreeance, so instead the absolute change is summed to a total,
        #             and the sign of the change is summed to a total (+ 1.0 for agreeance, -1.0 for non-agreeance)
        #   4. No immediate radicalisation was flagged, so instead the following check is made:
        #       (sum_changes >= social_susceptibility * len(hierarchy_changes) * SOCIAL_THRESHOLD_MODIFIER) and (total_sign >= 0.0)
        #   5. Following: (0.99 >= 0.75 * 1 * 0.5) and (-1.0 >= 0.0) = True and False = False
        #   6. Therefore, radicalisation will not occur
        self.assertIsInstance(
            radicalisation,
            bool,
            "Agent -- a valid call to radicalisation is not returning a boolean result (radicalisation=false)",
        )
        self.assertFalse(
            radicalisation,
            "Agent -- a valid call to radicalisation is not correctly reporting that radicalisation does not occur",
        )
        self.assertFalse(
            self.agent.radicalised,
            "Agent -- a valid call to radicalisation in which radicalisation did not occur has set the agent's radicalised attribute to True",
        )

    def test_evolve_hierarchies_invalid_input(self) -> None:
        """
        Test that evolve_hierarchies() with an invalid rw_distributions data type will produce the expected error.
        """
        rw_distribs_invalid: list[tuple[float, float]] = [(0.0, 0.1)]
        with self.assertRaises(TypeError, msg="rw_distributions must be a dictionary") as cm:
            self.agent.evolve_hierarchies(rw_distribs_invalid)

    def test_evolve_hierarchies_invalid_item(self) -> None:
        """
        Test that evolve_hierarchies() with a rw_distributions dict containing an item of an invalid data type will produce the expected error.
        """
        rw_distribs_invalid: dict[str, list[float]] = {"A": [0.0, 0.1]}
        with self.assertRaises(TypeError, msg="One or more items in rw_distributions is of an invalid data type -- all must be tuples") as cm:
            self.agent.evolve_hierarchies(rw_distribs_invalid)

    def test_evolve_hierarchies_invalid_item_value(self) -> None:
        """
        Test that evolve_hierarchies() with a rw_distributions dict containing valid tuple items with some invalid data type within them will produce the expected error.
        """
        rw_distribs_invalid: dict[str, tuple[float, str]] = {"A": (0.0, "0.1")}
        with self.assertRaises(TypeError, msg="One or more tuples in rw_distributions contain invalid data types -- all must be tuples with two float items") as cm:
            self.agent.evolve_hierarchies(rw_distribs_invalid)

    def test_evolve_hierarchies_invalid_hierarchy(self) -> None:
        """
        Test that evolve_hierarchies() with a non-existent hierarchy in the rw_distributions keys will produce the expected error.
        """
        rw_distribs_invalid: dict[str, tuple[float, float]] = {"foobar": (0.13, 0.12)}
        with self.assertRaises(KeyError, msg="One or more hierarchy keys in rw_distributions are not present in the agent's social_weightings") as cm:
            self.agent.evolve_hierarchies(rw_distribs_invalid)

    def test_evolve_hierarchies(self) -> None:
        """
        Test that a valid call to evolve_hierarchies() is working as intended.
        """
        rw_distribs: dict[str, tuple[float, float]] = {"A": (0.0, 0.5)}
        self.agent.evolve_hierarchies(rw_distribs)
        self.assertIsInstance(
            self.agent.social_weightings["A"],
            float,
            "Agent -- a valid call to evolve_hierarchies is causing a non-float value to be set for a social weighting",
        )
        self.assertTrue(
            (self.agent.social_weightings["A"] >= -agt.SOCIAL_WEIGHTINGS_MAX) and (self.agent.social_weightings["A"] <= agt.SOCIAL_WEIGHTINGS_MAX),
            "Agent -- a valid call to evolve_hierarchies is not producing social_weightings values in the valid range",
        )
        self.assertNotAlmostEqual(
            self.agent.social_weightings["A"],
            0.2,
            5,
            "Agent -- a valid call to evolve_hierarchies is not changing the social_weighting value of the hierarchies correctly",
        )

    def test_stochastic_opinion_invalid(self) -> None:
        """
        Test that a call to stochastic_opinion() with an opinion_rw of an invalid data type will produce the expected error.
        """
        invalid_opinion_rw: list[float] = [0.0, 0.1]
        with self.assertRaises(TypeError, msg="opinion_rw must be a tuple") as cm:
            self.agent.stochastic_opinion(invalid_opinion_rw)

    def test_stochastic_opinion_invalid_value(self) -> None:
        """
        Test that a call to stochastic_opinion() with an opinion_rw tuple containing an invalid data type will produce the expected error.
        """
        invalid_opinion_rw: tuple[float, str] = (0.0, "0.1")
        with self.assertRaises(TypeError, msg="One or both of the values in opinion_rw are invalid data types -- both must be floats") as cm:
            self.agent.stochastic_opinion(invalid_opinion_rw)

    def test_stochastic_opinion(self) -> None:
        """
        Test that a valid call to stochastic_opinion() is working as intended.
        """
        opinion_rw: tuple[float, float] = (0.0, 0.5)
        self.agent.stochastic_opinion(opinion_rw)
        self.assertIsInstance(
            self.agent.opinion,
            float,
            "Agent -- a valid call to stochastic_opinion is causing a non-float value to be set for the agent's opinion",
        )
        self.assertTrue(
            (self.agent.opinion >= -agt.OPINION_MAX) and (self.agent.opinion <= agt.OPINION_MAX),
            "Agent -- a valid call to stochastic_opinion is not producing an opinion value in the valid range",
        )
        self.assertNotAlmostEqual(
            self.agent.opinion,
            0.44,
            5,
            "Agent -- a valid call to stochastic_opinion is not changing the agent's opinion value correctly",
        )

    def test_stochastic_personality_change_invalid_prob(self) -> None:
        """
        Test that a call to stochastic_personality_change() with an invalid probability value will produce the expected error.
        """
        personalities_p_invalid: dict[str, float] = {"neutral": "half", "rational": "half"}
        with self.assertRaises(TypeError, msg=f"A non-float probability was supplied in personality_probs when trying to determine a stochastic personality change in agent {self.agent.id}") as cm:
            self.agent.stochastic_personality_change(personality_probs=personalities_p_invalid)

    def test_stochastic_personality_change_invalid_personality(self) -> None:
        """
        Test that a call to stochastic_personality_change() with an unsupported personality specified will produce the expected error.
        """
        personalities_p_invalid: dict[str, float] = {"neutral": 0.9, "rare": 0.1}
        with self.assertRaises(KeyError, msg=f"An unsupported personality was specified in personality_probs when trying to determine a stochastic personality change in agent {self.agent.id}") as cm:
            self.agent.stochastic_personality_change(personality_probs=personalities_p_invalid)

    def test_stochastic_personality_change(self) -> None:
        """
        Test that stochastic_personality_change() is working as intended.
        """
        self.agent.stochastic_personality_change()
        self.assertIn(
            self.agent.personality,
            agt.PERSONALITIES,
            "Agent -- a valid call to stochastic_personality_change without explicit probabilities is not drawing a valid personality type",
        )

    def test_stochastic_personality_change_probs(self) -> None:
        """
        Test that stochastic_personality_change() with explicit probabilities is working as intended.
        """
        personalities_p: dict[str, float] = {"neutral": 0.4, "impulsive": 0.4, "rational": 0.2}
        self.agent.stochastic_personality_change(personality_probs=personalities_p)
        self.assertIn(
            self.agent.personality,
            personalities_p.keys(),
            "Agent -- a valid call to stochastic_personality_change with explicit probabilities is not drawing a specified personality type",
        )

    def test_stochastic_benefit_change_false(self) -> None:
        """
        Test that stochastic_benefit_change() when personal_benefit was False will result in a personal_benefit of True.
        """
        self.agent.set_benefit(False)
        self.agent.stochastic_benefit_change()
        self.assertIsInstance(
            self.agent.personal_benefit,
            bool,
            "Agent -- stochastic_benefit_change set the personal_benefit attribute to a non-boolean value",
        )
        self.assertTrue(
            self.agent.personal_benefit,
            "Agent -- stochastic_benefit_change when the initial personal_benefit was False did not result in a final value of True",
        )

    def test_stochastic_benefit_change_true(self) -> None:
        """
        Test that stochastic_benefit_change() when personal_benefit was True will result in a personal_benefit of False.
        """
        self.agent.set_benefit(True)
        self.agent.stochastic_benefit_change()
        self.assertIsInstance(
            self.agent.personal_benefit,
            bool,
            "Agent -- stochastic_benefit_change set the personal_benefit attribute to a non-boolean value",
        )
        self.assertFalse(
            self.agent.personal_benefit,
            "Agent -- stochastic_benefit_change when the initial personal_benefit was True did not result in a final value of False",
        )

    def test_stochastic_radicalisation_change_false(self) -> None:
        """
        Test that stochastic_radicalisation_change() when radicalised was False will result in a radicalised of True.
        """
        self.agent.change_radicalisation(False)
        self.agent.stochastic_radicalisation_change()
        self.assertIsInstance(
            self.agent.radicalised,
            bool,
            "Agent -- stochastic_radicalisation_change set the radicalised attribute to a non-boolean value",
        )
        self.assertTrue(
            self.agent.radicalised,
            "Agent -- stochastic_radicalisation_change when the initial radicalised was False did not result in a final value of True",
        )

    def test_stochastic_radicalisation_change_true(self) -> None:
        """
        Test that stochastic_radicalisation_change() when radicalised was True will result in a radicalised of False.
        """
        self.agent.change_radicalisation(True)
        self.agent.stochastic_radicalisation_change()
        self.assertIsInstance(
            self.agent.radicalised,
            bool,
            "Agent -- stochastic_radicalisation_change set the radicalised attribute to a non-boolean value",
        )
        self.assertFalse(
            self.agent.radicalised,
            "Agent -- stochastic_radicalisation_change when the initial radicalised was True did not result in a final value of False",
        )
