from __future__ import annotations

from SCSPlotter import AnalysisResults


if __name__ == "__main__":
    ROOT_DIR: str = "./experiments/SyntheticCaseStudy/MixedEvents"

    SAVEDIR_ROOT: str = f"{ROOT_DIR}/Results"
    VISUALISATION_ROOT: str = f"{SAVEDIR_ROOT}/Visualisations"

    SAVEDIRS: dict[str, str] = {
        "NONMN": f"{SAVEDIR_ROOT}/NONMN",
        "NONMN2": f"{SAVEDIR_ROOT}/NONMN2",
        "NONMN3": f"{SAVEDIR_ROOT}/NONMN3",
        "NONMN4": f"{SAVEDIR_ROOT}/NONMN4",
        "NONMN5": f"{SAVEDIR_ROOT}/NONMN5",
        "MINNG": f"{SAVEDIR_ROOT}/MINNG",
        "MINNG2": f"{SAVEDIR_ROOT}/MINNG2",
        "MINNG3": f"{SAVEDIR_ROOT}/MINNG3",
        "MINNG4": f"{SAVEDIR_ROOT}/MINNG4",
        "MINNG5": f"{SAVEDIR_ROOT}/MINNG5",
    }
    SAVEFILES: dict[str, str] = {
        "NONMN": f"{SAVEDIRS['NONMN']}/NONMN_model_variables.csv",
        "NONMN2": f"{SAVEDIRS['NONMN2']}/NONMN2_model_variables.csv",
        "NONMN3": f"{SAVEDIRS['NONMN3']}/NONMN3_model_variables.csv",
        "NONMN4": f"{SAVEDIRS['NONMN4']}/NONMN4_model_variables.csv",
        "NONMN5": f"{SAVEDIRS['NONMN5']}/NONMN5_model_variables.csv",
        "MINNG": f"{SAVEDIRS['MINNG']}/MINNG_model_variables.csv",
        "MINNG2": f"{SAVEDIRS['MINNG2']}/MINNG2_model_variables.csv",
        "MINNG3": f"{SAVEDIRS['MINNG3']}/MINNG3_model_variables.csv",
        "MINNG4": f"{SAVEDIRS['MINNG4']}/MINNG4_model_variables.csv",
        "MINNG5": f"{SAVEDIRS['MINNG5']}/MINNG5_model_variables.csv",
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
        5,
        COMMUNITIES,
    )

    for model_name, save_file in SAVEFILES.items():
        results.import_model_parameters(model_name, save_file)

    results.calculate_all_stats()
    results.create_all_plots()
