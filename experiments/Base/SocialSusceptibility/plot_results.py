import csv

from gatoh.utils import plot_graph

if __name__ == "__main__":
    DATAFILES: dict[str, str] = {
        "ZERO": "./experiments/Base/SocialSusceptibility/ZERO_model_variables.csv",
        "POINT-TWO": "./experiments/Base/SocialSusceptibility/POINT-TWO_model_variables.csv",
        "POINT-FOUR": "./experiments/Base/SocialSusceptibility/POINT-FOUR_model_variables.csv",
        "POINT-SIX": "./experiments/Base/SocialSusceptibility/POINT-SIX_model_variables.csv",
        "POINT-EIGHT": "./experiments/Base/SocialSusceptibility/POINT-EIGHT_model_variables.csv",
        "ONE": "./experiments/Base/SocialSusceptibility/ONE_model_variables.csv",
    }

    aggregate_opinions: dict[str, list[float]] = {
        "ZERO": [],
        "POINT-TWO": [],
        "POINT-FOUR": [],
        "POINT-SIX": [],
        "POINT-EIGHT": [],
        "ONE": [],
    }
    radicalised_agents: dict[str, list[int]] = {
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

    iterations: dict[str, list[int]] = {
        "ZERO": [i + 1 for i in range(100)],
        "POINT-TWO": [i + 1 for i in range(100)],
        "POINT-FOUR": [i + 1 for i in range(100)],
        "POINT-SIX": [i + 1 for i in range(100)],
        "POINT-EIGHT": [i + 1 for i in range(100)],
        "ONE": [i + 1 for i in range(100)],
    }

    for model_instance, data_file in DATAFILES.items():
        with open(data_file, "r", newline="") as csv_file:
            csv_reader: csv.DictReader = csv.DictReader(csv_file)
            for row in csv_reader:
                aggregate_opinion: float = float(row["aggregate_opinions"])
                radicalised_agent: int = int(row["radicalised_agents"])
                # All hierarchies contain full population, so polarisation in one layer is representative of all
                polarisation: float = float(row["layer_polarisations_A"])

                aggregate_opinions[model_instance].append(aggregate_opinion)
                radicalised_agents[model_instance].append(radicalised_agent)
                polarisations[model_instance].append(polarisation)

    # Plot the aggregate opinions
    plot_graph(
        iterations,
        aggregate_opinions,
        x_label="Iterations",
        y_label="Network Aggregate Opinions",
        title="Network Aggregate Opinions over Iterations",
        save_path="./experiments/Base/SocialSusceptibility/SocialSusceptibility_AggOps.png",
    )

    # Plot the radicalised agents
    plot_graph(
        iterations,
        radicalised_agents,
        x_label="Iterations",
        y_label="Number of Radicalised Agents",
        title="Number of Radicalised Agents over Iterations",
        save_path="./experiments/Base/SocialSusceptibility/SocialSusceptibility_RadicalAgents.png",
    )

    # Plot the polarisations
    plot_graph(
        iterations,
        polarisations,
        x_label="Iterations",
        y_label="Network Polarisation",
        title="Network Polarisation over Iterations",
        save_path="./experiments/Base/SocialSusceptibility/SocialSusceptibility_Polarisations.png",
    )
