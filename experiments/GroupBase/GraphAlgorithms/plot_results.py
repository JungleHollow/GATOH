import csv

from gatoh.utils import plot_graph

if __name__ == "__main__":
    ROOT_DIR: str = "./experiments/GroupBase/GraphAlgorithms"

    DATAFILES: dict[str, str] = {
        "blockmodel": f"{ROOT_DIR}/blockmodel_model_variables.csv",
        "random": f"{ROOT_DIR}/random_model_variables.csv",
        "scale-free": f"{ROOT_DIR}/scale-free_model_variables.csv",
        "small-world": f"{ROOT_DIR}/small-world_model_variables.csv",
    }

    aggregate_opinions: dict[str, list[float]] = {
        "blockmodel": [],
        "random": [],
        "scale-free": [],
        "small-world": [],
    }
    radicalised_agents: dict[str, list[int | float]] = {
        "blockmodel": [],
        "random": [],
        "scale-free": [],
        "small-world": [],
    }
    polarisations: dict[str, list[float]] = {
        "blockmodel": [],
        "random": [],
        "scale-free": [],
        "small-world": [],
    }

    iterations: dict[str, list[int | float]] = {
        "blockmodel": [i + 1 for i in range(100)],
        "random": [i + 1 for i in range(100)],
        "scale-free": [i + 1 for i in range(100)],
        "small-world": [i + 1 for i in range(100)],
    }

    for model_type, data_file in DATAFILES.items():
        with open(data_file, "r", newline="") as csv_file:
            csv_reader: csv.DictReader[str] = csv.DictReader(csv_file)
            for row in csv_reader:
                aggregate_opinion: float = float(row["aggregate_opinions"])
                radicalised_agent: int = int(row["radicalised_agents"])
                # All hierarchies contain the full population, so polarisation in one layer is representative of all
                polarisation: float = float(row["layer_polarisations_family"])

                aggregate_opinions[model_type].append(aggregate_opinion)
                radicalised_agents[model_type].append(radicalised_agent)
                polarisations[model_type].append(polarisation)

    # Plot the aggregate opinions
    plot_graph(
        iterations,
        aggregate_opinions,
        x_label="Iterations",
        y_label="Network Aggregate Opinions",
        title="Network Aggregate Opinions over Iterations",
        save_path=f"{ROOT_DIR}/GraphAlgorithms_AggOps.png",
    )

    # Plot the radicalised agents
    plot_graph(
        iterations,
        radicalised_agents,
        x_label="Iterations",
        y_label="Number of Radicalisation Events",
        title="Number of Radicalisation Events over Iterations",
        save_path=f"{ROOT_DIR}/GraphAlgorithms_RadicalAgents.png",
    )

    # Plot the polarisations
    plot_graph(
        iterations,
        polarisations,
        x_label="Iterations",
        y_label="Network Polarisation",
        title="Network Polarisation over Iterations",
        save_path=f"{ROOT_DIR}/GraphAlgorithms_Polarisations.png",
    )
