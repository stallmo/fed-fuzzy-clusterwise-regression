# python
import pandas as pd
import numpy as np
import pytest
from datasets import DatasetDict

from fed_fuzzy_clust_regression.data_partition_utils import (
    get_column_names_for_dat_file,
    read_dat_file,
    add_artificial_label_column,
    transform_pandas_to_dataset_with_split,
    create_dataset_from_dat_file,
)


def _write_sample_dat(path, n_rows=20):
    """
    Write a sample .dat file with n_rows rows.
    """
    lines = [
        "@relation test",
        "@attribute f1 numeric",
        "@attribute f2 numeric",
        "@attribute target numeric",
        "@outputs target",
        "@comment generated for tests",
        "@data",
    ]
    data_lines = []
    rng = np.random.RandomState(0)
    for _ in range(n_rows):
        f1 = float(rng.normal())
        f2 = float(rng.normal())
        target = float(rng.normal())
        data_lines.append(f"{f1},{f2},{target}")
    content = "\n".join(lines + data_lines) + "\n"
    path.write_text(content)


def test_get_column_names_for_dat_file(tmp_path):
    p = tmp_path / "sample.dat"
    _write_sample_dat(p, n_rows=3)
    columns, target = get_column_names_for_dat_file(str(p))
    assert isinstance(columns, list)
    assert columns == ["f1", "f2", "target"]
    assert target == "target"


def test_read_dat_file_parses_data_correctly(tmp_path):
    p = tmp_path / "sample_read.dat"
    _write_sample_dat(p, n_rows=5)
    df, target = read_dat_file(str(p))
    # DataFrame should have the columns discovered by header parsing
    assert list(df.columns) == ["f1", "f2", "target"]
    assert target == "target"
    # number of rows equals written rows
    assert len(df) == 5
    # numeric types
    assert pd.api.types.is_numeric_dtype(df["f1"])
    assert pd.api.types.is_numeric_dtype(df["target"])


def test_add_artificial_label_column_creates_q_bins():
    # Create a simple dataframe with distinct numeric target values
    df = pd.DataFrame({
        "f1": np.arange(100).astype(float),
        "target": np.linspace(0, 1, 100)
    })
    q = 10
    df_labeled = add_artificial_label_column(df.copy(), target_column="target", q=q)
    assert "label" in df_labeled.columns
    labels = df_labeled["label"].values
    # labels should be integers in range 0..q-1
    assert labels.min() >= 0
    assert labels.max() <= q - 1
    # all q labels should appear (for evenly distributed linspace input)
    assert set(np.unique(labels)) == set(range(q))


def test_transform_pandas_to_dataset_with_split_returns_datasetdict():
    df = pd.DataFrame({
        "f1": np.arange(20).astype(float),
        "target": np.arange(20).astype(float)
    })
    ds = transform_pandas_to_dataset_with_split(df, test_size=0.25, seed=123)
    assert isinstance(ds, DatasetDict)
    assert set(ds.keys()) == {"train", "test"}
    total = ds["train"].num_rows + ds["test"].num_rows
    assert total == len(df)
    # label column not present by default in this df
    assert "target" in ds["train"].column_names


def test_create_dataset_from_dat_file_end_to_end(tmp_path):
    # End-to-end: write .dat, create dataset, check splits and label column
    p = tmp_path / "sample_full.dat"
    # create 40 rows to have meaningful train/test split
    _write_sample_dat(p, n_rows=40)
    ds = create_dataset_from_dat_file(str(p), test_size=0.2, seed=42)
    assert isinstance(ds, DatasetDict)
    assert set(ds.keys()) == {"train", "test"}
    total = ds["train"].num_rows + ds["test"].num_rows
    assert total == 40
    # ensure the split is correct
    assert ds["train"].num_rows == 32
    assert ds["test"].num_rows == 8
    # artificial label column should exist in the underlying pandas -> dataset
    # datasets columns are same across splits
    assert "label" in ds["train"].column_names
    # values of label are integers and within expected range
    lab_vals = np.unique(np.array(ds["train"]["label"]))
    assert lab_vals.min() >= 0
    assert lab_vals.max() < 10  # default q in implementation is 10
