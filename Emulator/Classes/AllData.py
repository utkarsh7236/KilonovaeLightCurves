from collections import defaultdict
import pandas as pd
from pathlib import Path


class AllData:
    """Load the data from the set path and prepare it in a useable format.
    """

    def __init__(self):
        path = Path(__file__).parent.resolve()
        self.folder_name = str(path) + "/bns_m3_3comp"  # Change folder name of data as required.
        self.parent = path

    def load_path(self, path_to_dir):
        """ User defined path
        """
        self.folder_path = path_to_dir
        self.path = path_to_dir + self.folder_name
        return None

    def load_raw_data(self):
        """ Loads raw data from given path. Implimentation may different for windows and mac/linux users.
        >>> data = AllData()
        >>> data.load_path(str(Path(__file__).parent.resolve()))
        >>> data.load_raw_data()
        >>> print(data.raw_data.file_name.iloc[0])
        nph1.0e+06_mejdyn0.001_mejwind0.130_phi45.txt
        >>> data.raw_data.file_name.iloc[192] == "nph1.0e+06_mejdyn0.001_mejwind0.090_phi0.txt"
        True
        >>> data.raw_data.file_name.iloc[192] == "nph1.0e+06_mejdyn0.005_mejwind0.110_phi0.txt"
        False
        >>> data.raw_data.file_name.iloc[192] == "nph1.0e+06_mejdyn0.005_mejwind0.110_phi0.txt"
        False
        """
        resd = defaultdict(list)
        folder_path = Path(str(self.parent.parent.resolve()) + "/bns_m3_3comp")

        for file in folder_path.iterdir():
            with open(file, "r") as file_open:
                resd["file_name"].append(file.name)
        temp_df = pd.DataFrame(resd)
        self.raw_data = temp_df[temp_df.file_name != ".DS_Store"].reset_index(drop=True)
        return None

    def process(self):
        """ Processes the data to a readable reference dataframe.
        >>> data = AllData()
        >>> data.load_path(str(Path(__file__).parent.resolve()))
        >>> data.load_raw_data()
        >>> data.process()
        >>> print(data.reference_data.mejwind.iloc[68])
        0.03
        >>> print(data.reference_data.mejdyn.iloc[173])
        0.02
        >>> data.reference_data.phi.iloc[55] == 75
        False
        >>> data.reference_data.phi.iloc[57] == 75
        False
        >>> data.reference_data.phi.iloc[56] == 75
        True
        """
        split_series = self.raw_data.file_name.apply(lambda x: x.split('_'))
        temp_df = split_series.apply(pd.Series)
        temp_df["file_name"] = self.raw_data.file_name
        temp_df.columns = ["nph", "mejdyn", "mejwind", "phi", "filename"]
        temp_df["mejdyn"] = temp_df["mejdyn"].str.extract("(\d*\.?\d+)", expand=True)
        temp_df["mejwind"] = temp_df["mejwind"].str.extract("(\d*\.?\d+)", expand=True)
        temp_df["phi"] = temp_df["phi"].str.extract("(\d*\.?\d+)", expand=True)
        temp_df["nph"] = temp_df["nph"].apply(lambda x: float(x[3:]))
        temp_df[["mejdyn", "mejwind", "phi"]] = temp_df[["mejdyn", "mejwind", "phi"]].apply(pd.to_numeric)
        self.reference_data = temp_df.reset_index(drop=True)
        return None

    def save_reference(self):
        """ Saves the reference data into a file for future use.
        """
        try:
            self.reference_data.to_csv("reference.csv", index=False)
            print("[STATUS] Reference Saved")
        except AssertionError:
            print("[ERROR] Reference Unsaved")

    def load_reference(self, name):
        """ Loads the saved dataframe to save on computing time.
        >>> data = AllData()
        >>> data.load_reference("reference.csv")
        >>> print(data.reference_data.mejwind.iloc[68])
        0.03
        >>> print(data.reference_data.mejdyn.iloc[173])
        0.02
        >>> data.reference_data.phi.iloc[55] == 75
        False
        >>> data.reference_data.phi.iloc[57] == 75
        False
        >>> data.reference_data.phi.iloc[56] == 75
        True
        """
        self.reference_data = pd.read_csv(name)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
