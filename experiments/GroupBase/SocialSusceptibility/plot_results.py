import csv

from gatoh.utils import plot_graph


if __name__ == "__main__":
    ROOT_DIR: str = "./experiments/GroupBase/SocialSusceptibility"

    DATAFILES: dict[str, str] = {
        "ZERO": f"{ROOT_DIR}/ZERO_model_variables.csv",
        "POINT-TWO": f"{ROOT_DIR}/POINT-TWO_model_variables.csv",
        "POINT-FOUR": f"{ROOT_DIR}/POINT-FOUR_model_variables.csv",
        "POINT-SIX": f"{ROOT_DIR}/POINT-SIX_model_variables.csv",
        "POINT-EIGHT": f"{ROOT_DIR}/POINT-EIGHT_model_variables.csv",
        "ONE": f"{ROOT_DIR}/ONE_model_variables.csv",
    }

    aggregate_opinions: dict[str, list[float]] = {
        "ZERO": [],
        "POINT-TWO": [],
        "POINT-FOUR": [],
        "POINT-SIX": [],
        "POINT-EIGHT": [],
        "ONE": [],
    }
    radicalised_agents: dict[str, list[int | float]] = {
        "ZERO": [],
        "POINT-TWO": [],
        "POINT-FOUR": [],
        "POINT-SIX": [],
        "POINT-EIGHT": [],
        "ONE": [],
    }
    radicalised_groups: dict[str, list[int | float]] = {
        "ZERO": [],
        "POINT-TWO": [],
        "POINT-FOUR": [],
        "POINT-SIX": [],
        "POINT-EIGHT": [],
        "ONE": [],
    }
    polarisations: dict[str, list[float]] = {
        "ZERO": [],
        "POINT-TWO": [],
        "POINT-FOUR": [],
        "POINT-SIX": [],
        "POINT-EIGHT": [],
        "ONE": [],
    }

    iterations: dict[str, list[float]] = {
        "ZERO": [i + 1 for i in range(100)],
        "POINT-TWO": [i + 1 for i in range(100)],
        "POINT-FOUR": [i + 1 for i in range(100)],
        "POINT-SIX": [i + 1 for i in range(100)],
        "POINT-EIGHT": [i + 1 for i in range(100)],
        "ONE": [i + 1 for i in range(100)],
    }

    for model_instance, data_file in DATAFILES.items():
        with open(data_file, "r", newline="") as csv_file:
            csv_reader: csv.DictReader[str] = csv.DictReader(csv_file)
            for row in csv_reader:
                aggregate_opinion: float = float(row["aggregate_opinions"])
                radicalised_agent: int = int(row["radicalised_agents"])
                radicalised_group: int = int(row["radicalised_groups"])
                # All hierarchies contain the full population, so polarisation in one layer is representative of all
                polarisation: float = float(row["layer_polarisations_A"])

                aggregate_opinions[model_instance].append(aggregate_opinion)
                radicalised_agents[model_instance].append(radicalised_agent)
                radicalised_groups[model_instance].append(radicalised_group)
                polarisations[model_instance].append(polarisation)

    # Plot the aggregate opinions
    plot_graph(
        iterations,
        aggregate_opinions,
        x_label="Iterations",
        y_label="Network Aggregate Opinions",
        title="Network Aggregate Opinions over Iterations",
        save_path=f"{ROOT_DIR}/GroupSocialSusceptibility_AggOps.png",
    )

    # Plot the radicalised agents
    plot_graph(
        iterations,
        radicalised_agents,
        x_label="Iterations",
        y_label="Number of Radicalised Agents",
        title="Number of Radicalised Agents over Iterations",
        save_path=f"{ROOT_DIR}/GroupSocialSusceptibility_RadicalAgents.png",
    )

    # Plot the radicalised groups
    plot_graph(
        iterations,
        radicalised_groups,
        x_label="Iterations",
        y_label="Number of Radicalised Groups",
        title="Number of Radicalised Groups over Iterations",
        save_path=f"{ROOT_DIR}/GroupSocialSusceptibility_RadicalGroups.png",
    )

    # Plot the polarisations
    plot_graph(
        iterations,
        polarisations,
        x_label="Iterations",
        y_label="Network Polarisation",
        title="Network Polarisation over Iterations",
        save_path=f"{ROOT_DIR}/GroupSocialSusceptibility_Polarisations.png",
    )
