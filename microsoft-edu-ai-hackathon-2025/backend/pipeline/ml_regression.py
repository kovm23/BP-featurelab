"""Regression training branch — RuleKit + XGBoost ensemble.

Mirrors `ml_classification.py` for the regression half of the training pipeline.
Extracted from `ml_training.py` to keep that module focused on orchestration.
"""
from __future__ import annotations

import json
import logging

import pandas as pd
from rulekit.regression import RuleRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from config import (
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_N_ESTIMATORS,
    XGB_RANDOM_STATE,
)
from pipeline.ml_classification import _run_cross_validation
from pipeline.ml_preprocessing import _apply_median_imputer, _fit_median_imputer
from pipeline.ml_rules import _count_rule_features, _extract_rules

logger = logging.getLogger(__name__)

RULEKIT_WEIGHT = 0.4
XGB_WEIGHT = 0.6


def _train_regression_branch(pipeline, X, y_raw, target_column, _cb, warnings: list) -> dict:
    """Train RuleKit + XGBoost ensemble regressor and update pipeline state in-place."""
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
    rulekit_mse = round(float(mean_squared_error(y, rulekit_pred)), 6)

    _cb(60, "Training XGBoost...")
    xgb_model = XGBRegressor(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        random_state=XGB_RANDOM_STATE,
        verbosity=0,
    )
    xgb_model.fit(X, y)
    xgb_pred = xgb_model.predict(X)
    xgb_mse = round(float(mean_squared_error(y, xgb_pred)), 6)

    _cb(75, "Ensemble prediction...")
    ensemble_pred = RULEKIT_WEIGHT * rulekit_pred + XGB_WEIGHT * xgb_pred
    ensemble_mse = round(float(mean_squared_error(y, ensemble_pred)), 6)

    _cb(80, "K-fold cross-validation...")
    cv_results = _run_cross_validation(X, y, rulekit_weight=RULEKIT_WEIGHT, xgb_weight=XGB_WEIGHT)
    logger.info("CV results: %s", cv_results)
    if cv_results.get("note"):
        warnings.append(cv_results["note"])

    xgb_importance = {}
    if hasattr(xgb_model, "feature_importances_"):
        xgb_importance = dict(sorted(
            {f: round(float(imp), 4) for f, imp in zip(X.columns, xgb_model.feature_importances_) if imp > 0.001}.items(),
            key=lambda x: -x[1],
        ))
    rulekit_importance = _count_rule_features(rules, list(X.columns))

    if cv_results.get("cv_mse") is not None and cv_results["cv_mse"] > 2 * ensemble_mse:
        warnings.append(
            f"Possible overfitting: CV MSE ({cv_results['cv_mse']:.6f}) "
            f"is much higher than training MSE ({ensemble_mse:.6f}). "
            "Consider adding more training data."
        )

    pipeline.model = rulekit_model
    pipeline.xgb_model = xgb_model
    pipeline.rules = rules
    pipeline.mse = ensemble_mse
    pipeline.rulekit_mse = rulekit_mse
    pipeline.xgb_mse = xgb_mse
    pipeline.cv_mse = cv_results.get("cv_mse")
    pipeline.cv_std = cv_results.get("cv_std")
    pipeline.cv_mae = cv_results.get("cv_mae")
    pipeline.train_accuracy = pipeline.train_balanced_accuracy = pipeline.train_f1_macro = pipeline.train_mcc = None
    pipeline.cv_accuracy = pipeline.cv_balanced_accuracy = pipeline.cv_f1_macro = None
    pipeline.cv_precision_macro = pipeline.cv_recall_macro = pipeline.cv_mcc = None
    pipeline.cv_folds = cv_results.get("n_folds")
    pipeline.feature_importance = {"xgboost": xgb_importance, "rulekit": rulekit_importance}
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
        "mse": ensemble_mse,
        "rulekit_mse": rulekit_mse,
        "xgb_mse": xgb_mse,
        **{k: cv_results.get(k) for k in ("cv_mse", "cv_std", "cv_mae")},
        "cv_folds": cv_results.get("n_folds"),
        "rules_count": len(rules),
        "rules": rules,
        "feature_spec": pipeline.feature_spec,
        "feature_importance": {"xgboost": xgb_importance, "rulekit": rulekit_importance},
        "warnings": warnings,
        "training_data_X": json.loads(pipeline.training_X.to_json(orient="records")),
    }
