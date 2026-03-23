"""Phase 3: RuleKit model training, and Phase 5: batch prediction."""
import json
import logging

import numpy as np
import pandas as pd
from rulekit.regression import RuleRegressor
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


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


def train_model(pipeline, target_column: str) -> dict:
    """Train a RuleKit regression model from stored training_X + training_Y_df.

    Updates pipeline state in-place and persists to disk.
    Returns a result dict suitable for JSON serialisation.
    """
    if pipeline.training_X is None:
        raise Exception("Phase 2 (feature extraction) must be completed first.")

    if pipeline.training_Y_df is None:
        raise Exception(
            "Missing dataset_Y (CSV with labels). "
            "It must be included in the training ZIP."
        )

    df_gt = pipeline.training_Y_df
    join_col = df_gt.columns[0]
    df_gt = df_gt.rename(columns={join_col: "media_name"})

    pipeline.training_X["media_name"] = (
        pipeline.training_X["media_name"].astype(str).str.strip()
    )
    df_gt["media_name"] = df_gt["media_name"].astype(str).str.strip()

    df_merged = pd.merge(pipeline.training_X, df_gt, on="media_name", how="inner")

    if df_merged.empty:
        raise Exception(
            "No data remained after joining dataset_X with dataset_Y. "
            "Check that the file names in the CSV match the media names (without extension)."
        )

    feature_cols = [c for c in pipeline.feature_spec if c in df_merged.columns]
    if not feature_cols:
        raise Exception("None of the features were found in the data.")

    X = _preprocess_features(df_merged[feature_cols])

    if target_column in df_merged.columns:
        y = df_merged[target_column]
    else:
        found = [c for c in df_merged.columns if c.lower() == target_column.lower()]
        if found:
            y = df_merged[found[0]]
        else:
            raise Exception(
                f"Column '{target_column}' not found in CSV. "
                f"Available columns: {list(df_gt.columns)}"
            )

    pipeline.model = RuleRegressor()
    pipeline.model.fit(X, y)

    pipeline.rules = (
        [str(rule) for rule in pipeline.model.model.rules]
        if hasattr(pipeline.model, 'model')
        else []
    )
    pipeline.mse = round(float(mean_squared_error(y, pipeline.model.predict(X))), 4)
    pipeline.is_trained = True
    pipeline.target_variable = target_column
    pipeline._training_columns = list(X.columns)

    pipeline.save_state()

    return {
        "status": "success",
        "mse": pipeline.mse,
        "rules_count": len(pipeline.rules),
        "rules": pipeline.rules,
        "feature_spec": pipeline.feature_spec,
        "training_data_X": json.loads(pipeline.training_X.to_json(orient="records")),
    }


def predict_batch(pipeline, testing_Y_df: pd.DataFrame | None = None) -> dict:
    """Predict for all objects in testing_X, optionally comparing with testing_Y_df."""
    if not pipeline.is_trained:
        raise Exception("Model is not trained. Complete Phase 3 first.")
    if pipeline.testing_X is None or pipeline.testing_X.empty:
        raise Exception("Missing testing dataset_X. Complete Phase 4 first.")

    feature_cols = [c for c in pipeline.feature_spec if c in pipeline.testing_X.columns]
    X_test = _preprocess_features(
        pipeline.testing_X[feature_cols],
        training_columns=pipeline._training_columns,
    )

    predictions = pipeline.model.predict(X_test)

    actual_values: dict[str, float] = {}
    if testing_Y_df is not None:
        join_col = testing_Y_df.columns[0]
        testing_Y_df = testing_Y_df.rename(columns={join_col: "media_name"})
        testing_Y_df["media_name"] = testing_Y_df["media_name"].astype(str).str.strip()
        target_col = None
        for c in testing_Y_df.columns:
            if c.lower() == pipeline.target_variable.lower() or c == pipeline.target_variable:
                target_col = c
                break
        if target_col is None:
            numeric_cols = [
                c for c in testing_Y_df.columns
                if c != "media_name" and pd.api.types.is_numeric_dtype(testing_Y_df[c])
            ]
            if numeric_cols:
                target_col = numeric_cols[-1]
        if target_col:
            for _, row in testing_Y_df.iterrows():
                actual_values[str(row["media_name"]).strip()] = float(row[target_col])

    results = []
    pred_list: list[float] = []
    actual_list: list[float] = []

    for i, row in pipeline.testing_X.iterrows():
        pred_score = float(predictions[i])
        media_name = str(row.get("media_name", f"object_{i}"))
        rule = pipeline.rules[0] if pipeline.rules else "Default rule"

        item = {
            "media_name": media_name,
            "predicted_score": round(pred_score, 4),
            "rule_applied": rule,
            "extracted_features": {
                k: row[k] for k in pipeline.feature_spec if k in row
            },
        }

        if media_name in actual_values:
            item["actual_score"] = round(actual_values[media_name], 4)
            pred_list.append(pred_score)
            actual_list.append(actual_values[media_name])

        results.append(item)

    metrics = None
    if pred_list and actual_list:
        pred_arr = np.array(pred_list)
        actual_arr = np.array(actual_list)
        mse = float(np.mean((pred_arr - actual_arr) ** 2))
        mae = float(np.mean(np.abs(pred_arr - actual_arr)))
        if len(pred_arr) > 1 and np.std(pred_arr) > 0 and np.std(actual_arr) > 0:
            correlation = float(np.corrcoef(pred_arr, actual_arr)[0, 1])
        else:
            correlation = None
        metrics = {
            "mse": round(mse, 6),
            "mae": round(mae, 6),
            "correlation": round(correlation, 4) if correlation is not None else None,
            "matched_count": len(pred_list),
            "total_count": len(results),
        }

    return {"predictions": results, "metrics": metrics}
