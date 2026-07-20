from typing import TypedDict

import polars as pl

QUESTION_RESPONSES: list[str] = [
    "18-25",
    "26-35",
    "36-45",
    "46-60",
    "60+",
    "Female",
    "Male",
    "Other",
    "No response",
    "<5",
    "5-10",
    "11-20",
    "20+",
    "Yes",
    "No",
    "A lot",
    "Moderately",
    "A little",
    "Not at all",
    "Improved",
    "Maintained",
    "Worsened",
    "Significant",
    "Slight",
    "None",
    "Always",
    "Sometimes",
    "Rarely",
    "Never",
    "Many",
    "Some",
    "A few",
    "None",
    "Very positive",
    "Positive",
    "Negative",
    "Very negative",
    "Yes",
    "No",
    "Very important",
    "Important",
    "Somewhat important",
    "Not important",
    "A lot",
    "Moderately",
    "A little",
    "Not at all",
    "In favour",
    "Against",
    "Neutral",
    "No opinion",
    "Unsure",
    "Always",
    "Sometimes",
    "Rarely",
    "Never",
    "Yes",
    "No",
    "Depends",
    "Environmental impact",
    "Economic benefits",
    "Community opinion",
    "Religious beliefs",
    "Personal experience",
    "Totally compatible",
    "Partially compatible",
    "A little compatible",
    "Incompatible",
    "Yes",
    "No",
    "Unsure",
    "Yes",
    "No",
    "Maybe",
    "Yes",
    "No",
    "Football",
    "Basketball",
    "Volleyball",
    "Athletics",
    "Other",
    "Local team",
    "National team",
    "International team",
    "Does not follow football",
    "Always",
    "Sometimes",
    "Rarely",
    "Never",
    "Family",
    "Friends",
    "Neighbours",
    "Alone",
    "A lot",
    "Moderately",
    "A little",
    "Not at all",
    "Yes",
    "No",
    "Local festivities",
    "Religious events",
    "Sports events",
    "Community meetings",
    "None",
    "Traditional music",
    "Popular music",
    "Religious music",
    "International music",
    "Varied",
    "Frequently",
    "Occasionally",
    "Rarely",
    "Never",
]


class OutputDict(TypedDict):
    """
    A helper class for typechecking of TableReader.output_dict.

    Each question number has been manually included to allow for the possibility of
    different data types per question in the future.
    """
    AgentId: list[str]
    Q01: list[str]
    Q02: list[str]
    Q03: list[str]
    Q04: list[str]
    Q05: list[str]
    Q06: list[str]
    Q07: list[str]
    Q08: list[str]
    Q09: list[str]
    Q10: list[str]
    Q11: list[str]
    Q12: list[str]
    Q13: list[str]
    Q14: list[str]
    Q15: list[str]
    Q16: list[str]
    Q17: list[str]
    Q18: list[str]
    Q19: list[str]
    Q20: list[str]
    Q21: list[str]
    Q22: list[str]
    Q23: list[str]
    Q24: list[str]
    Q25: list[str]
    Q26: list[str]
    Q27: list[str]
    Q28: list[str]
    Q29: list[str]
    Q30: list[str]



class TableReader:
    """
    A class that reads the case study survey data and translates it into English.

    :param filename: The path to the original case study data .csv file for a community.
    :type filename: str
    :param community_code: The unique identifier that will be assigned to the community to which the data belongs.
    :type community_code: str
    """
    def __init__(self, filename: str, community_code: str):
        self.filename: str = filename
        self.community_code: str = community_code

        self.dataframe: pl.DataFrame
        with open(self.filename, "r") as file:
            self.dataframe = pl.read_csv(file, has_header=False)

        self.output_dict: OutputDict = {
            "AgentId": [],
            "Q01": [],
            "Q02": [],
            "Q03": [],
            "Q04": [],
            "Q05": [],
            "Q06": [],
            "Q07": [],
            "Q08": [],
            "Q09": [],
            "Q10": [],
            "Q11": [],
            "Q12": [],
            "Q13": [],
            "Q14": [],
            "Q15": [],
            "Q16": [],
            "Q17": [],
            "Q18": [],
            "Q19": [],
            "Q20": [],
            "Q21": [],
            "Q22": [],
            "Q23": [],
            "Q24": [],
            "Q25": [],
            "Q26": [],
            "Q27": [],
            "Q28": [],
            "Q29": [],
            "Q30": [],
        }
        self.output_dataframe: pl.DataFrame

        for i in range(1, 31):
            self.output_dict[f"Q{i:02}"] = []

    def parse_values(self):
        """
        Simply iterates through the csv file's values and appropriately converts them to their English counterparts.
        """
        for row_idx, row in enumerate(self.dataframe.iter_rows()):
            agent_id: str = f"{self.community_code}{row_idx + 1:04}"
            self.output_dict["AgentId"].append(agent_id)

            current_q: int = 1
            for col_idx, col in enumerate(row):
                if col == 1 or col == "1":
                    response: str = QUESTION_RESPONSES[col_idx]
                    self.output_dict[f"Q{current_q:02}"].append(response)
                    current_q += 1
                else:
                    continue

    def write_out(self, output_path: str):
        """
        Writes the converted .csv file to a new path.

        :param output_path: The new path to which the translated .csv file should be written.
        :type output_path: str
        """
        self.output_dataframe = pl.DataFrame(self.output_dict)
        self.output_dataframe.write_csv(output_path)


if __name__ == "__main__":
    table_reader: TableReader = TableReader(
        "./data/CaseStudy/no_mineria_tabla.csv",
        "NONMN",
    )
    table_reader.parse_values()
    table_reader.write_out(
        "./data/NONMN/NonMining.csv"
    )

    table_reader_two: TableReader = TableReader(
        "./data/CaseStudy/mineria_tabla.csv",
        "MINNG",
    )
    table_reader_two.parse_values()
    table_reader_two.write_out(
        "./data/MINNG/Mining.csv"
    )
