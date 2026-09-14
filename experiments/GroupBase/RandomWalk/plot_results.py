import csv

from gatoh.utils import plot_graph

if __name__ == "__main__":
    ROOT_DIR: str = "./experiments/GroupBase/RandomWalk"

    DATAFILES: dict[str, str] = {
        "BASE": f"{ROOT_DIR}/BASE_model_variables.csv",
        "RELS": f"{ROOT_DIR}/RELS_model_variables.csv",
        "HIER": f"{ROOT_DIR}/HIER_model_variables.csv",
        "BOTH": f"{ROOT_DIR}/BOTH_model_variables.csv",
    }

    aggregate_opinions: dict[str, list[float]] = {
        "BASE": [],
        "RELS": [],
        "HIER": [],
        "BOTH": [],
    }
    radicalised_agents: dict[str, list[int | float]] = {
        "BASE": [],
        "RELS": [],
        "HIER": [],
        "BOTH": [],
    }
    radicalised_groups: dict[str, list[int | float]] = {
        "BASE": [],
        "RELS": [],
        "HIER": [],
        "BOTH": [],
    }
    polarisations: dict[str, list[float]] = {
        "BASE": [],
        "RELS": [],
        "HIER": [],
        "BOTH": [],
    }

    iterations: dict[str, list[int | float]] = {
        "BASE": [i + 1 for i in range(100)],
        "RELS": [i + 1 for i in range(100)],
        "HIER": [i + 1 for i in range(100)],
        "BOTH": [i + 1 for i in range(100)],
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
        save_path=f"{ROOT_DIR}/RandomWalk_AggOps.png",
    )

    # Plot the radicalised agents
    plot_graph(
        iterations,
        radicalised_agents,
        x_label="Iterations",
        y_label="Number of Radicalised Agents",
        title="Number of Radicalised Agents over Iterations",
        save_path=f"{ROOT_DIR}/RandomWalk_RadicalAgents.png",
    )

    # Plot the radicalised groups
    plot_graph(
        iterations,
        radicalised_groups,
        x_label="Iterations",
        y_label="Number of Radicalised Groups",
        title="Number of Radicalised Groups over Iterations",
        save_path=f"{ROOT_DIR}/RandomWalk_RadicalGroups.png",
    )

    # Plot the polarisations
    plot_graph(
        iterations,
        polarisations,
        x_label="Iterations",
        y_label="Network Polarisation",
        title="Network Polarisation over Iterations",
        save_path=f"{ROOT_DIR}/RandomWalk_Polarisations.png"
    )
