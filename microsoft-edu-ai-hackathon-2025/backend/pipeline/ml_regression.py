"""Regression training branch — RuleKit only."""
from __future__ import annotations

import json
import logging

import pandas as pd
from rulekit.regression import RuleRegressor
from sklearn.metrics import mean_squared_error

from pipeline.ml_preprocessing import _apply_median_imputer, _fit_median_imputer
from pipeline.ml_rules import _count_rule_features, _extract_rules

logger = logging.getLogger(__name__)


def _train_regression_branch(pipeline, X, y_raw, target_column, _cb, warnings: list) -> dict:
    """Train RuleKit regressor and update pipeline state in-place."""
    y = pd.to_numeric(y_raw, errors="coerce")
    valid_mask = ~y.isna()
    if not valid_mask.any():
        raise Exception("Target column is not numeric. For non-numeric labels switch to classification mode.")
    X, y = X.loc[valid_mask], y.loc[valid_mask]
    reg_medians = _fit_median_imputer(X)
    X = _apply_median_imputer(X, reg_medians)

    _cb(25, "Training RuleKit (rule induction)...")
    rulekit_model = RuleRegressor()
    rulekit_model.fit(X, y)
    rules = _extract_rules(rulekit_model)
    rulekit_pred = rulekit_model.predict(X)
    mse = round(float(mean_squared_error(y, rulekit_pred)), 6)

    rulekit_importance = _count_rule_features(rules, list(X.columns))

    pipeline.model = rulekit_model
    pipeline.rules = rules
    pipeline.mse = mse
    pipeline.cv_mse = None
    pipeline.cv_std = None
    pipeline.cv_mae = None
    pipeline.train_accuracy = pipeline.train_balanced_accuracy = pipeline.train_f1_macro = pipeline.train_mcc = None
    pipeline.cv_accuracy = pipeline.cv_balanced_accuracy = pipeline.cv_f1_macro = None
    pipeline.cv_precision_macro = pipeline.cv_recall_macro = pipeline.cv_mcc = None
    pipeline.cv_folds = None
    pipeline.feature_importance = {"rulekit": rulekit_importance}
    pipeline.is_trained = True
    pipeline.target_variable = target_column
    pipeline.predictions = pipeline.prediction_metrics = None
    pipeline._training_columns = list(X.columns)
    pipeline._scaler_mean = [reg_medians.get(col, 0.0) for col in pipeline._training_columns]
    pipeline._scaler_scale = []
    pipeline._label_classes = []
    pipeline.warnings = warnings

    pipeline.save_state()
    _cb(98, "Saving model...")
    return {
        "status": "success",
        "target_mode": "regression",
        "mse": mse,
        "rules_count": len(rules),
        "rules": rules,
        "feature_spec": pipeline.feature_spec,
        "feature_importance": {"rulekit": rulekit_importance},
        "warnings": warnings,
        "training_data_X": json.loads(pipeline.training_X.to_json(orient="records")),
    }
