import polars as pl

QUESTION_RESPONSES = [
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


class TableReader:
    def __init__(self, filename: str, community_code: str):
        self.filename: str = filename
        self.community_code: str = community_code

        self.dataframe: pl.DataFrame
        with open(self.filename, "r") as file:
            self.dataframe = pl.read_csv(file, has_header=False)

        self.output_dict: dict = {"AgentId": []}
        self.output_dataframe: pl.DataFrame

        for i in range(1, 31):
            self.output_dict[f"Q{i:02}"] = []

    def parse_values(self):
        for row_idx, row in enumerate(self.dataframe.iter_rows()):
            agent_id: str = f"{self.community_code}{row_idx + 1:05}"
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
        self.output_dataframe = pl.DataFrame(self.output_dict)
        self.output_dataframe.write_csv(output_path)


if __name__ == "__main__":
    table_reader: TableReader = TableReader(
        "/home/manuelms/Desktop/Uni Work/Masters Thesis/Papers Submissions/chocarpamba_tabla.csv",
        "CHPBA",
    )
    table_reader.parse_values()
    table_reader.write_out(
        "/home/manuelms/Desktop/Uni Work/Masters Thesis/ABMOS/data/Chocarpamba.csv"
    )
