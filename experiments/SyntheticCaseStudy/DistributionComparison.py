from __future__ import annotations

import csv
import os
from copy import deepcopy

from scipy.stats import ttest_ind
from scipy.stats._result_classes import TtestResult


# Declare all relevant global variables here
SAVEDIR_ROOT: str = "./experiments/SyntheticCaseStudy"

# The number of repetitions that each community was run for where relevant
REPETITIONS: int = 5

# The social hierarchies that were modelled across the experiments
HIERARCHIES: list[str] = [
    # "Age",
    # "Gender",
    "Friends",
    "Family",
    "Cultural",
    "Religious",
    "Geographical",
    "Social",
]

# Define the base save files for each scenario
NONMN_SAVEFILES: dict[str, list[str]] = {
    "Neutral": [f"{SAVEDIR_ROOT}/Neutral/Results/NONMN/NONMN_model_variables.csv"],
    "OngoingNegative": [f"{SAVEDIR_ROOT}/OngoingNegative/Results/NONMN/NONMN_model_variables.csv"],
    "InitialNegative": [f"{SAVEDIR_ROOT}/InitialNegative/Results/NONMN/NONMN_model_variables.csv"],
    "MixedEvents": [f"{SAVEDIR_ROOT}/MixedEvents/Results/NONMN/NONMN_model_variables.csv"],
}
MINNG_SAVEFILES: dict[str, list[str]] = {
    "Neutral": [f"{SAVEDIR_ROOT}/Neutral/Results/MINNG/MINNG_model_variables.csv"],
    "OngoingNegative": [f"{SAVEDIR_ROOT}/OngoingNegative/Results/MINNG/MINNG_model_variables.csv"],
    "InitialNegative": [f"{SAVEDIR_ROOT}/InitialNegative/Results/MINNG/MINNG_model_variables.csv"],
    "MixedEvents": [f"{SAVEDIR_ROOT}/MixedEvents/Results/MINNG/MINNG_model_variables.csv"],
}
# Then loop to add all repetitions
for i in range(1, REPETITIONS):
    NONMN_SAVEFILES["OngoingNegative"].append(f"{SAVEDIR_ROOT}/OngoingNegative/Results/NONMN{i + 1}/NONMN{i + 1}_model_variables.csv")
    NONMN_SAVEFILES["InitialNegative"].append(f"{SAVEDIR_ROOT}/InitialNegative/Results/NONMN{i + 1}/NONMN{i + 1}_model_variables.csv")
    NONMN_SAVEFILES["MixedEvents"].append(f"{SAVEDIR_ROOT}/MixedEvents/Results/NONMN{i + 1}/NONMN{i + 1}_model_variables.csv")
    MINNG_SAVEFILES["OngoingNegative"].append(f"{SAVEDIR_ROOT}/OngoingNegative/Results/MINNG{i + 1}/MINNG{i + 1}_model_variables.csv")
    MINNG_SAVEFILES["InitialNegative"].append(f"{SAVEDIR_ROOT}/InitialNegative/Results/MINNG{i + 1}/MINNG{i + 1}_model_variables.csv")
    MINNG_SAVEFILES["MixedEvents"].append(f"{SAVEDIR_ROOT}/MixedEvents/Results/MINNG{i + 1}/MINNG{i + 1}_model_variables.csv")


class DistributionComparator:
    """
    A class that simply takes in paths to existing GATOH experiment results .csv files and then performs pair-wise comparisons
    for these distributions.

    This class will appropriately handle cases in which experiments have multiple repetitions, aggregating the results for each
    repetition in an experiment as needed.

    :param model_hierarchies: The social hierarchies that were modelled within each experiment.
    :type model_hierarchies: list[str]
    :param saves: An arbitrary number of collections of saves which will have pair-wise comparisons performed on the distributions within each collection.
    :type saves: tuple[dict[str, list[str]], ...]
    """
    def __init__(self, model_hierarchies: list[str], **saves):
        self.hierarchies: list[str] = model_hierarchies
        self.saves: dict[str, dict[str, list[str]]] = saves
        self.opinions_data: dict[str, dict[str, list[float]]] = {}
        self.polarisations_data: dict[str, dict[str, dict[str, list[float]]]] = {}
        self.ttest_results: dict[str, dict[str, dict[str, TtestResult]]] = {}
        # The names of all columns which will be written to the output .csv file
        self.fieldnames: list[str] = [
            "Collection",  # The collection of save files that the recorded row originates from
            "Type",  # The type of statistic/test that is being recorded in this row
            "ID",  # Some type of identifier outlining the involved distribution(s), and hierarchies where relevant
            "Statistic",  # The actual calculated statistic or result of the test
            "P-Value",  # The p-value associated with the test (where applicable)
        ]

    def load_data(self) -> None:
        """
        Uses the save paths provided during initialisation to appropriately load the .csv GATOH result files,
        and aggregate the data where needed.
        """
        for collection, saves in self.saves.items():
            self.opinions_data.setdefault(collection, {})
            self.polarisations_data.setdefault(collection, {})

            for experiment, data_paths in saves.items():
                self.opinions_data[collection].setdefault(experiment, [])
                self.polarisations_data[collection].setdefault(experiment, {})

                for data_path in data_paths:
                    # Open the CSV file and process it
                    with open(data_path, "r", newline="") as csv_file:
                        csv_reader: csv.DictReader[str] = csv.DictReader(csv_file)

                        for row in csv_reader:
                            self.opinions_data[collection][experiment].append(float(row["aggregate_opinions"]))

                            for hierarchy in self.hierarchies:
                                self.polarisations_data[collection][experiment].setdefault(hierarchy, []).append(float(row[f"layer_polarisations_{hierarchy}"]))
        return None

    def save_results(self, save_path: str) -> None:
        """
        Save all results that have been calculated by this class into a .csv file at the specified save path.

        :param save_path: A path to a .csv file in which all calculated results should be written.
        :type save_path: str
        """
        with open(save_path, "w", newline="") as csv_file:
            csv_writer: csv.DictWriter[str] = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            csv_writer.writeheader()

            row_dict: dict[str, str | int | float | TtestResult]

            # First, write the ttest results
            for collection, ttests in self.ttest_results.items():
                for ttest, pairings in ttests.items():
                    test_type: str = f"TTest -- {ttest}"
                    for pairing, result in pairings.items():
                        row_dict = {
                            "Collection": collection,
                            "Type": test_type,
                            "ID": pairing,
                            "Statistic": float(result[0]),
                            "P-Value": float(result[1]),
                        }
                        csv_writer.writerow(row_dict)

            # TODO: Any future statistics/results have to be added to this function ...

        return None

    def perform_ttest(self, base_experiment: str | None = None) -> None:
        """
        Perform a pair-wise Welch's t-test across all experiments in a collection of results.

        If a base experiment is specified, then only tests between the base experiment and all other experiments
        will be carried out.

        :param base_experiment: The name of the baseline experiment against which all other experiments are being compared.
        :type base_experiment: str, optional
        """
        for collection, experiments in self.opinions_data.items():
            self.ttest_results.setdefault(collection, {})
            self.ttest_results[collection].setdefault("opinions", {})

            if base_experiment is None:
                for i in range(len(experiments)):
                    for j in range(i, len(experiments)):
                        self.ttest_results[collection]["opinions"][f"{list(experiments.keys())[i]}-{list(experiments.keys())[j]}"] = ttest_ind(
                            list(experiments.values())[i],
                            list(experiments.values())[j],
                            equal_var=False,
                        )
            else:
                for experiment, values in experiments.items():
                    if experiment == base_experiment:
                        continue

                    self.ttest_results[collection]["opinions"][f"{base_experiment}-{experiment}"] = ttest_ind(
                        experiments[base_experiment],
                        values,
                        equal_var=False,
                    )

        for collection, experiments in self.polarisations_data.items():
            self.ttest_results[collection].setdefault("polarisations", {})

            if base_experiment is None:
                for i in range(len(experiments)):
                    for j in range(i, len(experiments)):
                        for hierarchy in self.hierarchies:
                            self.ttest_results[collection]["polarisations"][f"{list(experiments.keys())[i]}-{list(experiments.keys())[j]}-{hierarchy}"] = ttest_ind(
                                list(experiments.values())[i][hierarchy],
                                list(experiments.values())[j][hierarchy],
                                equal_var=False,
                            )
            else:
                for experiment, values in experiments.items():
                    if experiment == base_experiment:
                        continue

                    for hierarchy in self.hierarchies:
                        self.ttest_results[collection]["polarisations"][f"{base_experiment}-{experiment}-{hierarchy}"] = ttest_ind(
                            experiments[base_experiment][hierarchy],
                            values[hierarchy],
                            equal_var=False,
                        )
        return None


if __name__ == "__main__":
    comparator: DistributionComparator = DistributionComparator(HIERARCHIES, NONMN=NONMN_SAVEFILES, MINNG=MINNG_SAVEFILES)
    comparator.load_data()
    comparator.perform_ttest(base_experiment="Neutral")
    comparator.save_results("./experiments/SyntheticCaseStudy/scenario_distributions_comparison.csv")
