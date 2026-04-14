"""Feature preprocessing utilities: imputation, encoding, oversampling."""
import numpy as np
import pandas as pd


def _fit_median_imputer(X: pd.DataFrame) -> dict[str, float]:
    medians: dict[str, float] = {}
    for col in X.columns:
        numeric_series = pd.to_numeric(X[col], errors="coerce")
        if numeric_series.notna().any():
            medians[col] = float(numeric_series.median())
    return medians


def _apply_median_imputer(X: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    out = X.copy()
    for col in out.columns:
        numeric_series = pd.to_numeric(out[col], errors="coerce")
        if col in medians:
            out[col] = numeric_series.fillna(medians[col])
        else:
            out[col] = numeric_series
    return out.fillna(0.0)


def _oversample_minority(X: pd.DataFrame, y: pd.Series, max_factor: int = 3) -> tuple[pd.DataFrame, pd.Series]:
    counts = y.value_counts()
    if counts.empty or len(counts) < 2:
        return X, y

    majority = int(counts.max())
    parts_X = [X]
    parts_y = [y]
    for label, count in counts.items():
        if count <= 0:
            continue
        factor = min(max_factor, max(1, majority // int(count)))
        if factor > 1:
            mask = y == label
            parts_X.append(pd.concat([X.loc[mask]] * (factor - 1), ignore_index=True))
            parts_y.append(pd.concat([y.loc[mask]] * (factor - 1), ignore_index=True))

    return pd.concat(parts_X, ignore_index=True), pd.concat(parts_y, ignore_index=True)


def _preprocess_features(df: pd.DataFrame, training_columns: list[str] | None = None) -> pd.DataFrame:
    """Normalise and one-hot-encode feature columns.

    When *training_columns* is provided (prediction mode), the result is
    aligned to exactly those columns (missing cols filled with 0).
    """
    X = df.copy()
    for col in X.columns:
        X[col] = X[col].apply(
            lambda v: ", ".join(str(i) for i in v) if isinstance(v, list) else v
        )
    for col in X.select_dtypes(include="object").columns:
        X[col] = (
            X[col]
            .astype(str)
            .str.lower()
            .str.strip()
            .str.replace(r'\s+', '_', regex=True)
        )
        X[col] = X[col].replace(
            {'nan': None, 'not_applicable': None, 'n/a': None, 'none': None}
        )
    X = pd.get_dummies(X)
    X = X.loc[:, ~X.columns.duplicated()]

    if training_columns is not None:
        for col in training_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[training_columns]

    return X
