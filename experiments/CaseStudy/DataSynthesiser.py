from typing import Any

import argparse
import copy
import json
import random
import warnings

import polars as pl
import numpy as np


class DataSynthesiser:
    """
    A class that is used to generate synthetic data entries based on the real results seen
    in the case study surveys.

    :param response_file: The path to the .csv file in which the survey results are written.
    :type response_file: str
    :param output_path: The path to a .csv file in which the generated synthetic data should be written.
    :type output_path: str
    :param survey_values: The path to a .json file which codifies the values of every possible survey response.
    :type survey_values: str
    :param community_code: The community ID code to assign to the generated synthetic entries.
    :type community_code: str, optional
    :param social_graphs: The names of the social hierarchies which will be present in the generated data.
    :type social_graphs: list[str], optional
    """
    def __init__(
        self,
        response_file: str,
        output_path: str,
        survey_values: str,
        community_code: str = "FALSE",
        social_graphs: list[str] | None = None,
    ) -> None:
        self.response_file: str = response_file
        self.output_path: str = output_path
        self.values_path: str = survey_values
        self.community_code: str = community_code

        self.social_graphs: list[str]
        if social_graphs is None:
            self.social_graphs = ["Age", "Family", "Friends", "Religion", "Cultural"]
        else:
            self.social_graphs = social_graphs

        self.survey_values: dict[str, Any]
        with open(self.values_path, "r") as file:
            self.survey_values = json.load(file)

        self.response_distribution: dict[str, Any]
        with open(self.response_file, "r") as file:
            self.response_distribution = json.load(file)

        self.num_questions: int = len(self.response_distribution.keys())

        self.num_synthetic_entries: int = 0

        self.output_dataframe: pl.DataFrame
        self.output_dict: dict[str, Any] = {"AgentId": []}

        self.output_relationships: dict[str, Any] = {
            hierarchy: [] for hierarchy in self.social_graphs
        }  # Will be lists of (start_node, end_node, weight) triples

        for question in self.response_distribution:
            self.output_dict[question] = []

    def generate_n_entries(self, n: int = 100) -> None:
        """
        Using the responses recorded in self.response_distribution, create n randomly distributed
        data entries.

        :param n: The number of randomly distributed data entries to create.
        :type n: int, optional
        """
        for _ in range(n):
            self.num_synthetic_entries += 1

            agent_id: str = f"{self.community_code}{self.num_synthetic_entries:05}"
            self.output_dict["AgentId"].append(agent_id)

            is_religious: bool = False
            participates_community: bool = False

            for question, responses in self.response_distribution.items():
                choices: list[str] = list(responses.keys())
                weights: list[int] = list(responses.values())
                generated_response: str = ""

                match question:
                    case "Q11":
                        generated_response = random.choices(
                            choices, weights=weights, k=1
                        )[0]
                        if generated_response == "Yes":
                            is_religious = True
                    case "Q12":
                        if is_religious:
                            generated_response = random.choices(
                                choices[:-1], weights=weights[:-1], k=1
                            )[0]
                        else:
                            generated_response = "Not important"
                    case "Q27":
                        generated_response = random.choices(
                            choices, weights=weights, k=1
                        )[0]
                        if generated_response == "Yes":
                            participates_community = True
                    case "Q30":
                        if participates_community:
                            generated_response = random.choices(
                                choices[:2], weights=weights[:2], k=1
                            )[0]
                        else:
                            generated_response = random.choices(
                                choices[2:], weights=weights[2:], k=1
                            )[0]
                    case _:
                        generated_response = random.choices(
                            choices, weights=weights, k=1
                        )[0]

                self.output_dict[question].append(generated_response)
        self.create_dataframe()
        self.generate_relationships()
        return None

    def generate_relationships(self) -> None:
        """
        Randomly generate the social hierarchy relationships for the agents based on assumptions and other responses
        """

        for i in self.output_dataframe.iter_rows(named=True):
            for hierarchy in self.social_graphs:
                match hierarchy:
                    case "Age":
                        for j in self.output_dataframe.iter_rows(named=True):
                            if (
                                j["AgentId"] == i["AgentId"]
                            ):  # Cannot have relationships start and end at the same node...
                                continue
                            else:
                                self.gen_age_relation(i, j)
                    case "Family":
                        continue  # Handled externally
                    case "Friends":
                        continue  # Handled externally
                    case "Religion":
                        if i["Q11"] == "No":  # The start_node agent is not religious
                            continue
                        else:
                            for j in self.output_dataframe.iter_rows(named=True):
                                if (
                                    j["Q11"] == "No"
                                ):  # The end_node agent is not religious
                                    continue
                                elif i["AgentId"] == j["AgentId"]:
                                    continue
                                else:
                                    self.gen_religious_relation(i, j)
                    case "Cultural":
                        for j in self.output_dataframe.iter_rows(named=True):
                            if i["AgentId"] == j["AgentId"]:
                                continue
                            else:
                                self.gen_cultural_relation(i, j)
                    case _:
                        warnings.warn(
                            "Undefined social hierarchy was passed to the data synthesiser",
                            UserWarning,
                        )

        self.gen_family_relationships()
        self.gen_friend_relationships()
        return None

    def gen_age_relation(self, start_row: dict[str, Any], end_row: dict[str, Any]) -> None:
        """
        Helper function to simplify generate_relationships();
        handles the "Age" social hierarchy

        :param start_row: The dataframe row of the start_node agent for which the relationship is being generated.
        :type start_row: dict[str, Any]
        :param end_row: The dataframe row of the end_node agent for which the relationship is being generated.
        :type end_row: dict[str, Any]
        """
        weight: float = 1.0 - (
            0.25
            * abs(
                self.survey_values["Q1"][start_row["Q1"]]
                - self.survey_values["Q1"][end_row["Q1"]]
            )
        )

        # Add some stochastic noise to the generated weight
        weight = np.random.normal(weight, 0.2 * weight)

        # Ensure the weight is still within range
        if weight < 0.0:
            weight = 0.0
        elif weight > 1.0:
            weight = 1.0

        # Now check if the weight goes negative (implying a dislike)
        # (>= 1.95 should result in only around 5% of weights being negative)
        negative_check: bool = abs(np.random.normal(0, 1)) >= 1.95

        if negative_check:
            weight *= -1.0

        self.output_relationships["Age"].append(
            (start_row["AgentId"], end_row["AgentId"], weight)
        )
        return None

    def gen_religious_relation(self, start_row: dict[str, Any], end_row: dict[str, Any]) -> None:
        """
        Helper function to simplify generate_relationships();
        handles the "Religion" social hierarchy

        Simply aggregates a starting agent's "religious score" and uses this to uniformly apply
        a relationship weight to all other religious agents in the community.

        The start_node and end_node both have to be religious in the first place; which has
        been checked for in generate_relationships() prior to calling this function.

        :param start_row: The dataframe row of the start_node agent for which the relationship is being generated.
        :type start_row: dict[str, Any]
        :param end_row: The dataframe row of the end_node agent for which the relationship is being generated.
        :type end_row: dict[str, Any]
        """
        weight: float = (
            self.survey_values["Q12"][start_row["Q12"]]
            + self.survey_values["Q13"][start_row["Q13"]]
            + self.survey_values["Q14"][start_row["Q14"]]
            + self.survey_values["Q15"][start_row["Q15"]]
            + self.survey_values["Q16"][start_row["Q16"]]
        ) / 11.0  # 11.0 is the maximum possible score for these questions

        # Add some stochastic noise to the generated weight
        weight = np.random.normal(weight, 0.1 * weight)

        # Ensure the weight is still within range
        if weight < 0.0:
            weight = 0.0
        elif weight > 1.0:
            weight = 1.0

        # Now check if the weight goes negative (implying a dislike)
        # (>= 1.95 should result in only around 5% of weights being negative)
        negative_check: bool = abs(np.random.normal(0, 1)) >= 1.95

        if negative_check:
            weight *= -1.0

        self.output_relationships["Religion"].append(
            (start_row["AgentId"], end_row["AgentId"], weight)
        )
        return None

    def gen_cultural_relation(self, start_row: dict[str, Any], end_row: dict[str, Any]) -> None:
        """
        Helper function to simplify generate_relationships();
        handles the "Cultural" social hierarchy.

        :param start_row: The dataframe row of the start_node agent for which the relationship is being generated.
        :type start_row: dict[str, Any]
        :param end_row: The dataframe row of the end_node agent for which the relationship is being generated.
        :type end_row: dict[str, Any]
        """
        weight: float = (
            self.survey_values["Q20"][start_row["Q20"]]
            + self.survey_values["Q30"][start_row["Q30"]]
            + self.survey_values["Q23"][start_row["Q23"]]
            + self.survey_values["Q24"][start_row["Q24"]]
            + self.survey_values["Q25"][start_row["Q25"]]
            + self.survey_values["Q26"][start_row["Q26"]]
        ) / 16.0  # 16.0 is the maximum possible score for these questions

        # Add some stochastic noise to the generated weight
        weight = np.random.normal(weight, 0.1 * weight)

        # Check that the weight is still within range
        if weight < 0.0:
            weight = 0.0
        elif weight > 1.0:
            weight = 1.0

        # Now check if the weight goes negative (implying a dislike)
        # (>= 1.95 should result in only around 5% of weights being negative)
        negative_check: bool = abs(np.random.normal(0, 1)) >= 1.95

        if negative_check:
            weight *= -1.0

        self.output_relationships["Cultural"].append(
            (start_row["AgentId"], end_row["AgentId"], weight)
        )
        return None

    def gen_family_relationships(self) -> None:
        """
        Helper function to simplify generate_relationships();
        handles the "Family" social hierarchy.

        Set aside an initial 10% of the population to treat as having no family to account for
        'outsiders' that simply work in the town, or those whose families have moved away, etc.

        The remainder of the population is randomly clustered into groups of sizes ranging from
        3 to 12 to represent families in small communities.

        NOTE: With the current approach, interconnections between family clusters (i.e. through
        a marriage joining these families) are not accounted for.
        """
        agent_df: pl.DataFrame = copy.deepcopy(self.output_dataframe.select("AgentId"))
        no_family: list[Any] = list(
            agent_df.sample(fraction=0.1, with_replacement=False)["AgentId"]
        )  # Leave 10% of the total pop as having no family
        agent_df = agent_df.filter(~pl.col("AgentId").is_in(no_family))

        current_sample: list[Any]  # Define but don't initialise the variable yet

        while (
            len(agent_df["AgentId"]) >= 3
        ):  # Any remainders will also be treated as having no family
            family_size: int = random.randint(
                3, 12
            )  # Families can range from 3 to 12 people
            current_sample = list(
                agent_df.sample(n=family_size, with_replacement=False)["AgentId"]
            )
            for i in current_sample:
                for j in current_sample:
                    if i == j:
                        continue
                    # Should create a random weight in [0, 1], with mean 0.5
                    weight: float = random.random()

                    # Check if the weight goes negative (implying a dislike)
                    # (>= 1.95 should result in only around 5% of the weights becoming negative)
                    negative_check: bool = abs(np.random.normal(0, 1)) >= 1.95

                    if negative_check:
                        weight *= -1.0

                    self.output_relationships["Family"].append((i, j, weight))
            agent_df = agent_df.filter(~pl.col("AgentId").is_in(current_sample))
        return None

    def gen_friend_relationships(self) -> None:
        """
        Helper function to simplify generate_relationships();
        handles the "Friends" social hierarchy.

        Randomly creates relationships from each agent to a random susbset of 10% of all other agents.
        This means that unidirectional and unbalanced friendships can exist between the agents in the graph.
        """
        agent_df: pl.DataFrame = copy.deepcopy(self.output_dataframe.select("AgentId"))
        agent_list: list[Any] = list(agent_df["AgentId"])
        for i in agent_list:
            random_sample: list[Any] = list(
                agent_df.sample(fraction=0.1, with_replacement=False)["AgentId"]
            )
            for j in random_sample:
                if i == j:  # Case where 'i' was included in the random sample
                    continue
                weight: float = random.random()

                # Check if the weight goes negative (implying a dislike)
                # (>= 1.95 should result in only around 5% of weights becoming negative)
                negative_check: bool = abs(np.random.normal(0, 1)) >= 1.95

                if negative_check:
                    weight *= -1.0

                self.output_relationships["Friends"].append((i, j, weight))
        return None

    def create_dataframe(self) -> None:
        """
        Create a DataFrame from self.output_dict.
        """
        self.output_dataframe = pl.DataFrame(self.output_dict)
        return None

    def write_csv(self) -> None:
        """
        Writes the generated dataframe to a .csv file.
        """
        self.output_dataframe.write_csv(self.output_path)
        return None


class SynthesiserArgParser:
    def __init__(self) -> None:
        self.parser: argparse.ArgumentParser = argparse.ArgumentParser(
            prog="GATOH DataSynthesiser",
            description="Create synthetic data from observed distributions for use with the GATOH library",
        )
        _ = self.parser.add_argument("response_file", type=str)
        _ = self.parser.add_argument("output_path", type=str)
        _ = self.parser.add_argument("-c", "--community_code", default="FALSE", type=str)
        _ = self.parser.add_argument("-n", "--num_entries", default=100, type=int)
        _ = self.parser.add_argument(
            "-s",
            "--social_graphs",
            default=["Age", "Family", "Friends", "Religion", "Cultural"],
            type=list,
        )
        self.main()
        return None

    def main(self) -> None:
        """
        Initialises and handles the argument parser and processing loop for the data synthesiser.
        """
        args: argparse.Namespace = self.parser.parse_args()
        data_synthesiser: DataSynthesiser = DataSynthesiser(
            args.response_file,
            args.output_path,
            args.community_code,
            args.social_graphs,
        )
        data_synthesiser.generate_n_entries(args.num_entries)
        data_synthesiser.write_csv()


if __name__ == "__main__":
    parser: SynthesiserArgParser = SynthesiserArgParser()
