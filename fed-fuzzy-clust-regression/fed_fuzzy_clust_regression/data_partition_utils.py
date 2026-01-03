import pandas as pd
from datasets import Dataset, DatasetDict


def get_column_names_for_dat_file(file_path: str) -> tuple:
    """
    Extracts the column names from a .dat file and returns them as a list
    :param file_path: Path to the .dat file
    :return: Tuple of list of column names and target column name
    """
    # open the file and read the first lines one by one to get column names
    column_names = []
    target_column = None
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('@attribute'):
                # Extract the column name from the line
                parts = line.split()
                if len(parts) >= 2:
                    column_name = parts[1]
                    column_names.append(column_name)
            elif line.startswith('@outputs') or line.startswith('@output'):
                parts = line.split()
                target_column = parts[1]
            elif line.startswith('@data'):
                # Stop reading when we reach the data section
                break
    return column_names, target_column


def read_dat_file(file_path: str) -> tuple:
    """
    Reads a .dat file and returns a Pandas DataFrame and the target column name.
    :param file_path: Path to the .dat file
    :return: Tuple of Pandas DataFrame and target column name
    """

    column_names, target_column = get_column_names_for_dat_file(file_path)
    df = pd.read_csv(file_path, header=None, names=column_names, sep=",", engine="python",
                     skiprows=len(column_names) + 4)
    return df, target_column


def add_artificial_label_column(df: pd.DataFrame, target_column: str, q: int = 10) -> pd.DataFrame:
    """
    Adds an artificial label column based on the target column using quantile cuts.
    The column name is "label".
    :param df: Dataframe containing the data. Specifically, the target column must be present.
    :param target_column: Name of the target column on which to perform quantile cuts.
    :param q: Number of quantile cuts to perform (=number of classes).
    :return: Dataframe with an additional label column.
    """
    label_col_name = "label"
    df[label_col_name] = pd.qcut(df[target_column], q, labels=False, duplicates="drop")

    return df

def transform_pandas_to_dataset_with_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 43,
                                           info_dict: dict = None) -> DatasetDict:
    """
    Transforms a Pandas DataFrame into a HuggingFace Dataset and splits it into train and test sets.
    :param info_dict:
    :param df: Dataframe to be transformed.
    :return: DatasetDict containing the train and test set.
    """
    ds = Dataset.from_pandas(df)
    if info_dict is not None:
        ds.info.metadata = info_dict
    # Define train/test split
    ds = ds.train_test_split(test_size=test_size, seed=seed)

    return ds


def create_dataset_from_dat_file(path: str, test_size: float = 0.2, seed: int = 43) -> DatasetDict:
    """
    Creates a HuggingFace Dataset from a .dat file. Splits the data into train and test sets.
    :param path:
    :param test_size:
    :param seed:
    :return:
    """
    df_data, target_column = read_dat_file(path)
    # make sure the target column always has the same name
    df_data.rename(columns={target_column: "target"}, inplace=True)

    # add artificial label column for distributing data
    df_data = add_artificial_label_column(df_data, target_column="target")
    ds = transform_pandas_to_dataset_with_split(df_data, test_size, seed, info_dict=None)

    return ds
