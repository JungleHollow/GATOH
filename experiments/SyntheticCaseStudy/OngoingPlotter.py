from __future__ import annotations

from SCSPlotter import AnalysisResults


if __name__ == "__main__":
    ROOT_DIR: str = "./experiments/SyntheticCaseStudy/OngoingNegative"

    SAVEDIR_ROOT: str = f"{ROOT_DIR}/Results"
    VISUALISATION_ROOT: str = f"{SAVEDIR_ROOT}/Visualisations"

    SAVEDIRS: dict[str, str] = {
        "NONMN": f"{SAVEDIR_ROOT}/NONMN",
        "MINNG": f"{SAVEDIR_ROOT}/MINNG",
    }
    SAVEFILES: dict[str, str] = {
        "NONMN": f"{SAVEDIRS['NONMN']}/NONMN_model_variables.csv",
        "MINNG": f"{SAVEDIRS['MINNG']}/MINNG_model_variables.csv",
    }

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

    COMMUNITIES: list[str] = ["MINNG", "NONMN"]

    results: AnalysisResults = AnalysisResults(
        ROOT_DIR,
        100,
        HIERARCHIES,
        1,
        COMMUNITIES,
    )

    for model_name, save_file in SAVEFILES.items():
        results.import_model_parameters(model_name, save_file)

    results.calculate_all_stats()
    results.create_all_plots()
