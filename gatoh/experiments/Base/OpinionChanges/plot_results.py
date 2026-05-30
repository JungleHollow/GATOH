import csv

from gatoh.utils.utils import plot_graph

if __name__ == "__main__":
    ROOT_DIR: str = "./gatoh/experiments/Base/OpinionChanges"
    LOGGED_SAVEDIRS: str = (
        "./gatoh/experiments/Base/OpinionChanges/OpinionChanges_logged_savedirs.csv"
    )
    SAVEDIRS: dict[str, str] = {}

    MODEL_HIERARCHIES: list[str] = ["A", "B", "C", "D", "E", "F"]

    graph_groups: dict[str, list[tuple[str, str]]] = {}

    iterations: dict[str, dict[str, list[int]]] = {}

    aggregate_opinions: dict[str, dict[str, list[float]]] = {}
    radicalised_agents: dict[str, dict[str, list[int]]] = {}
    polarisations: dict[str, dict[str, list[float]]] = {}

    with open(LOGGED_SAVEDIRS, "r", newline="") as csv_file:
        csv_reader: csv.DictReader = csv.DictReader(csv_file)
        for row in csv_reader:
            SAVEDIRS[row["model_name"]] = row["model_savedir"]

    # Batch the models so that repeats with the same change iteration can be graphed together
    for model, save_dir in SAVEDIRS.items():
        change_iteration: str = model.split("_")[0]  # Extract only the slice "ITER-000"
        if change_iteration not in graph_groups.keys():
            graph_groups[change_iteration] = []

            # It is assumed that the other dicts will also be uninitialised
            iterations[change_iteration] = {}
            aggregate_opinions[change_iteration] = {}
            radicalised_agents[change_iteration] = {}
            polarisations[change_iteration] = {}
        if model not in aggregate_opinions[change_iteration].keys():
            aggregate_opinions[change_iteration][model] = []

            # It is assumed that the other dicts will also be uninitialised
            iterations[change_iteration][model] = [i + 1 for i in range(100)]
            radicalised_agents[change_iteration][model] = []
            polarisations[change_iteration][model] = []

        graph_groups[change_iteration].append((model, save_dir))

    # Load the data
    for graph_group, data_files in graph_groups.items():
        for data_file in data_files:
            with open(data_file[1], "r", newline="") as csv_file:
                csv_reader: csv.DictReader = csv.DictReader(csv_file)

                # In this experiment, Agent membership is not equal, so polarisations must be averaged
                network_polarisations: list[float] = []

                for row in csv_reader:
                    aggregate_opinion: float = float(row["aggregate_opinions"])
                    radicalised_agent: int = int(row["radicalised_agents"])

                    polarisation: float = 0.0
                    for hierarchy in MODEL_HIERARCHIES:
                        polarisation += float(row[f"layer_polarisations_{hierarchy}"])
                    polarisation /= len(MODEL_HIERARCHIES)

                    aggregate_opinions[graph_group][data_file[0]].append(
                        aggregate_opinion
                    )
                    radicalised_agents[graph_group][data_file[0]].append(
                        radicalised_agent
                    )
                    polarisations[graph_group][data_file[0]].append(polarisation)

    # Create the aggregate opinion plots for each graph group
    for graph_group, instances in aggregate_opinions.items():
        graph_group_save_path: str = f"./gatoh/experiments/Base/SocialSusceptibility/SocialSusceptibility_{graph_group}_AggOps.png"

        change_iteration_int: int = int(graph_group.split("-")[1])

        plot_graph(
            iterations[graph_group],
            instances,
            x_label="Iterations",
            y_label="Network Aggregate Opinions",
            title="Network Aggregate Opinions over Iterations",
            save_path=graph_group_save_path,
            vertical_x=change_iteration_int,
            vertical_name="Change Iteration",
        )

    # Create the radicalised agents plots for each graph group
    for graph_group, instances in radicalised_agents.items():
        graph_group_save_path: str = f"./gatoh/experiments/Base/SocialSusceptibility/SocialSusceptibility_{graph_group}_RadicalAgents.png"

        change_iteration_int: int = int(graph_group.split("-")[1])

        plot_graph(
            iterations[graph_group],
            instances,
            x_label="Iterations",
            y_label="Number of Radicalised Agents",
            title="Number of Radicalised Agents over Iterations",
            save_path=graph_group_save_path,
            vertical_x=change_iteration_int,
            vertical_name="Change Iteration",
        )

    # Create the polarisation plots for each graph group
    for graph_group, instances in polarisations.items():
        graph_group_save_path: str = f"./gatoh/experiments/Base/SocialSusceptibility/SocialSusceptibility_{graph_group}_Polarisations.png"

        change_iteration_int: int = int(graph_group.split("-")[1])

        plot_graph(
            iterations[graph_group],
            instances,
            x_label="Iterations",
            y_label="Network Polarisation",
            title="Network Polarisation over Iterations",
            save_path=graph_group_save_path,
            vertical_x=change_iteration_int,
            vertical_name="Change Iteration",
        )
