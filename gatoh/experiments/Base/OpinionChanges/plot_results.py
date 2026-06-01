import csv
from copy import deepcopy

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
    radicalised_agents: dict[str, dict[str, list[float]]] = {}
    polarisations: dict[str, dict[str, list[float]]] = {}

    csv_reader: csv.DictReader

    with open(LOGGED_SAVEDIRS, "r", newline="") as csv_file:
        csv_reader = csv.DictReader(csv_file)
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
                csv_reader = csv.DictReader(csv_file)

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

    # Calculate averages for each graph group
    for graph_group in aggregate_opinions.keys():
        agg_opps_avgs: list[float] = []
        radical_agts_avgs: list[float] = []
        polarisations_avgs: list[float] = []

        for i in range(100):
            agg_opps_sum: float = 0.0
            radical_agts_total: float = 0.0
            polarisations_sum: float = 0.0

            for instance in aggregate_opinions[graph_group].keys():
                agg_opps_sum += aggregate_opinions[graph_group][instance][i]
                radical_agts_total += radicalised_agents[graph_group][instance][i]
                polarisations_sum += polarisations[graph_group][instance][i]

            num_instances: int = len(list(aggregate_opinions[graph_group].keys()))

            agg_opps_avgs.append(float(agg_opps_sum / num_instances))
            radical_agts_avgs.append(float(radical_agts_total / num_instances))
            polarisations_avgs.append(float(polarisations_sum / num_instances))

        aggregate_opinions[graph_group]["Average"] = deepcopy(agg_opps_avgs)
        radicalised_agents[graph_group]["Average"] = deepcopy(radical_agts_avgs)
        polarisations[graph_group]["Average"] = deepcopy(polarisations_avgs)

        # Remember to update the iterations dict
        iterations[graph_group]["Average"] = [i + 1 for i in range(100)]

        # Manual garbage collection
        del agg_opps_avgs, radical_agts_avgs, polarisations_avgs

    # Create the aggregate opinion plots for each graph group
    graph_group_save_path: str
    change_iteration_int: int

    for graph_group, instances in aggregate_opinions.items():
        graph_group_save_path = f"./gatoh/experiments/Base/OpinionChanges/plots/OpinionChanges_{graph_group}_AggOps.png"

        change_iteration_int = int(graph_group.split("-")[1])

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
        graph_group_save_path = f"./gatoh/experiments/Base/OpinionChanges/plots/OpinionChanges_{graph_group}_RadicalAgents.png"

        change_iteration_int = int(graph_group.split("-")[1])

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
        graph_group_save_path = f"./gatoh/experiments/Base/OpinionChanges/plots/OpinionChanges_{graph_group}_Polarisations.png"

        change_iteration_int = int(graph_group.split("-")[1])

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

    # Create experiment-level plots with only the averages from each graph group
    group_averages: dict[str, dict[str, list[float]]] = {
        "agg_opps": {},
        "rad_agents": {},
        "polarisations": {},
    }
    group_iterations: dict[str, list[float]] = {}

    # First extract only the average information to the above dicts
    for graph_group in aggregate_opinions.keys():
        group_averages["agg_opps"][graph_group] = deepcopy(
            aggregate_opinions[graph_group]["Average"]
        )
        group_averages["rad_agents"][graph_group] = deepcopy(
            radicalised_agents[graph_group]["Average"]
        )
        group_averages["polarisations"][graph_group] = deepcopy(
            polarisations[graph_group]["Average"]
        )
        group_iterations[graph_group] = [i + 1 for i in range(100)]

    # Then plot the averages to the experiment's root directory
    plot_graph(
        group_iterations,
        group_averages["agg_opps"],
        x_label="Average Aggregate Opinions",
        y_label="Iterations",
        title="Average Aggregate Opinions over Iterations",
        save_path="./gatoh/experiments/Base/OpinionChanges/OpinionChanges_AggOpps.png",
    )
    plot_graph(
        group_iterations,
        group_averages["rad_agents"],
        x_label="Average Number of Radicalised Agents",
        y_label="Iterations",
        title="Average Number of Radicalised Agents over Iterations",
        save_path="./gatoh/experiments/Base/OpinionChanges/OpinionChanges_RadicalAgents.png",
    )
    plot_graph(
        group_iterations,
        group_averages["polarisations"],
        x_label="Average Network Polarisation",
        y_label="Iterations",
        title="Average Network Polarisation over Iterations",
        save_path="./gatoh/experiments/Base/OpinionChanges/OpinionChanges_Polarisations.png",
    )
