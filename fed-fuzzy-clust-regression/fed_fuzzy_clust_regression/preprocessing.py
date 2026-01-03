from typing import Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.validation import check_is_fitted

class QuantileClipper(BaseEstimator, TransformerMixin):
    """
    Clips extreme values column-wise based on lower and upper quantiles.

    :param lower: Lower quantile to clip values to.
    :param upper: Upper quantile to clip values to.
    :param include_non_numeric: If True, non-numeric columns are passed through unchanged.
    """
    def __init__(self, lower: float = 0.01, upper: float = 0.99, include_non_numeric: bool = True):
        self.lower = float(lower)
        self.upper = float(upper)
        self.include_non_numeric = bool(include_non_numeric)

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None):
        # keep original column names / generate defaults for arrays
        if isinstance(X, pd.DataFrame):
            self._columns = X.columns.tolist()
            df = X
        else:
            X = np.asarray(X)
            self._columns = [f"col_{i}" for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=self._columns)

        # numeric columns only
        self._numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(self._numeric_columns) == 0:
            raise ValueError("No numeric columns found to compute quantiles.")

        # compute quantiles per numeric column
        q_lower = df[self._numeric_columns].quantile(self.lower)
        q_upper = df[self._numeric_columns].quantile(self.upper)

        self.lower_ = q_lower.to_dict()
        self.upper_ = q_upper.to_dict()

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        check_is_fitted(self, "lower_")

        input_was_df = isinstance(X, pd.DataFrame)
        if input_was_df:
            df = X.copy()
        else:
            X = np.asarray(X)
            df = pd.DataFrame(X, columns=self._columns)

        # clip only numeric columns; non-numeric left unchanged
        for col in self._numeric_columns:
            low = self.lower_.get(col, -np.inf)
            high = self.upper_.get(col, np.inf)
            df[col] = df[col].clip(lower=low, upper=high)

        return df if input_was_df else df.values

    def inverse_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        # clipping is not invertible; behave as passthrough
        input_was_df = isinstance(X, pd.DataFrame)
        if input_was_df:
            return X.copy()
        return np.asarray(X)

    def get_feature_names_out(self, input_features: Optional[list] = None):
        check_is_fitted(self, "lower_")
        return np.array(self._columns)

class DataFrameStandardScaler(BaseEstimator, TransformerMixin):
    """
    Wrapper around sklearn's StandardScaler that works with pandas DataFrames.
    """
    def __init__(self, **scaler_kwargs):
        self.scaler = StandardScaler(**scaler_kwargs)

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self._columns = X.columns.tolist()
        else:
            X = np.asarray(X)
            self._columns = [f"col_{i}" for i in range(X.shape[1])]
        self.scaler.fit(X, y)
        return self

    def transform(self, X):
        check_is_fitted(self.scaler, "scale_")
        is_df = isinstance(X, pd.DataFrame)
        arr = X.values if is_df else np.asarray(X)
        out = self.scaler.transform(arr)
        if is_df:
            return pd.DataFrame(out, index=X.index, columns=self._columns)
        return pd.DataFrame(out, columns=self._columns)

    def inverse_transform(self, X):
        check_is_fitted(self.scaler, "scale_")
        is_df = isinstance(X, pd.DataFrame)
        arr = X.values if is_df else np.asarray(X)
        out = self.scaler.inverse_transform(arr)
        if is_df:
            return pd.DataFrame(out, index=X.index, columns=self._columns)
        return pd.DataFrame(out, columns=self._columns)

    def get_feature_names_out(self):
        check_is_fitted(self.scaler, "scale_")
        return np.array(self._columns)

def get_preprocessing_pipeline(q_lower: float, q_upper: float, include_non_numeric: bool = True):
    """
    Creates a preprocessing pipeline that clips extreme values and scales them.
    :param q_lower:
    :param q_upper:
    :param include_non_numeric:
    :return:
    """
    pipeline = Pipeline([
        ("quantile_clipper", QuantileClipper(lower=q_lower, upper=q_upper, include_non_numeric=include_non_numeric)),
        #("standard_scaler", DataFrameStandardScaler())
        ("minmax_scaler", MinMaxScaler(feature_range=(0, 1)))
    ])
    return pipeline