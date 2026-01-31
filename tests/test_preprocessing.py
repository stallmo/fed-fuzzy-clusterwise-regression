# Add package path to sys.path so tests can import fed_fuzzy_clust_regression
import os
import sys

import numpy as np
import pandas as pd
import pytest

from fed_fuzzy_clust_regression.preprocessing import (
    QuantileClipper,
    DataFrameStandardScaler,
    get_preprocessing_pipeline,
)


def test_quantile_clipper_dataframe_clips_and_preserves_non_numeric():
    df = pd.DataFrame({
        "a": [0.0, 10.0, 20.0, 30.0],
        "b": [1.0, 2.0, 3.0, 4.0],
        "c": ["x", "y", "z", "w"],
    })

    qc = QuantileClipper(lower=0.25, upper=0.75, include_non_numeric=True)
    qc.fit(df)

    # quantiles computed from the fit
    q_lower_a = qc.lower_["a"]
    q_upper_a = qc.upper_["a"]

    out = qc.transform(df)
    # DataFrame input should return a DataFrame
    assert isinstance(out, pd.DataFrame)
    # numeric column "a" should be clipped between lower and upper
    assert out["a"].min() >= q_lower_a - 1e-8
    assert out["a"].max() <= q_upper_a + 1e-8
    # non-numeric column should be unchanged (values equal)
    assert out["c"].tolist() == df["c"].tolist()


def test_quantile_clipper_numpy_array_and_feature_names():
    X = np.array([[0.0, 1.0], [10.0, 2.0], [20.0, 3.0]])
    qc = QuantileClipper(lower=0.0, upper=1.0)
    qc.fit(X)

    # fitting on ndarray should create default column names
    feature_names = qc.get_feature_names_out()
    assert list(feature_names) == ["col_0", "col_1"]

    out = qc.transform(X)
    # array input should return a numpy array
    assert isinstance(out, np.ndarray)
    assert out.shape == X.shape


def test_quantile_clipper_no_numeric_raises():
    df = pd.DataFrame({"a": ["x", "y"], "b": ["u", "v"]})
    qc = QuantileClipper()
    with pytest.raises(ValueError):
        qc.fit(df)


def test_dataframe_standard_scaler_dataframe_roundtrip():
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [10.0, 11.0, 12.0]})
    scaler = DataFrameStandardScaler()
    scaler.fit(df)

    out = scaler.transform(df)
    assert isinstance(out, pd.DataFrame)
    # columns preserved
    assert out.columns.tolist() == df.columns.tolist()
    # inverse_transform recovers approximate original values
    inv = scaler.inverse_transform(out)
    assert np.allclose(inv.values, df.values)


def test_dataframe_standard_scaler_array_input_returns_dataframe_with_columns():
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [10.0, 11.0, 12.0]})
    scaler = DataFrameStandardScaler()
    scaler.fit(df)

    arr = np.array([[1.0, 11.0], [0.5, 10.5]])
    out = scaler.transform(arr)
    assert isinstance(out, pd.DataFrame)
    assert out.columns.tolist() == ["x", "y"]


def test_get_preprocessing_pipeline_returns_pipeline_and_steps():
    pipeline = get_preprocessing_pipeline(0.01, 0.99, include_non_numeric=True)
    from sklearn.pipeline import Pipeline

    assert isinstance(pipeline, Pipeline)

    # should contain quantile_clipper and a scaler step
    step_names = [name for name, _ in pipeline.steps]
    assert "quantile_clipper" in step_names
    assert any("scaler" in name or "minmax" in name for name in step_names)
