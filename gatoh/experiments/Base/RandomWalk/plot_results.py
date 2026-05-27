import csv

from gatoh.utils import plot_graph

if __name__ == "__main__":
    DATAFILES: dict[str, str] = {
        "BASE": "./experiments/Base/RandomWalk/BASE_model_variables.csv",
        "RELS": "./experiments/Base/RandomWalk/RELS_model_variables.csv",
        "HIER": "./experiments/Base/RandomWalk/HIER_model_variables.csv",
        "BOTH": "./experiments/Base/RandomWalk/BOTH_model_variables.csv",
    }

    aggregate_opinions: dict[str, list[float]] = {
        "BASE": [],
        "RELS": [],
        "HIER": [],
        "BOTH": [],
    }
    radicalised_agents: dict[str, list[int]] = {
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

    iterations: dict[str, list[int]] = {
        "BASE": [i + 1 for i in range(100)],
        "RELS": [i + 1 for i in range(100)],
        "HIER": [i + 1 for i in range(100)],
        "BOTH": [i + 1 for i in range(100)],
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
        save_path="./experiments/Base/RandomWalk/RandomWalk_AggOps.png",
    )

    # Plot the radicalised agents
    plot_graph(
        iterations,
        radicalised_agents,
        x_label="Iterations",
        y_label="Number of Radicalised Agents",
        title="Number of Radicalised Agents over Iterations",
        save_path="./experiments/Base/RandomWalk/RandomWalk_RadicalAgents.png",
    )

    # Plot the polarisations
    plot_graph(
        iterations,
        polarisations,
        x_label="Iterations",
        y_label="Network Polarisation",
        title="Network Polarisation over Iterations",
        save_path="./experiments/Base/RandomWalk/RandomWalk_Polarisations.png",
    )
