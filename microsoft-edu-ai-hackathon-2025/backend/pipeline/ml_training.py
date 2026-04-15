"""Phase 3: Model training, and Phase 5: batch prediction.

Training pipeline:
  1. Preprocess features (one-hot encoding, null handling)
  2. Regression: RuleKit + XGBoost ensemble
  3. Classification: RuleKit classifier with interpretable rules
  4. Cross-validation for realistic error estimate
  5. Feature importance from XGBoost + rule frequency from RuleKit
"""
import json
import logging
import os

import numpy as np
import pandas as pd
from rulekit.classification import RuleClassifier
from rulekit.regression import RuleRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_squared_error,
)
from xgboost import XGBRegressor

from pipeline.ml_classification import (
    _compute_classification_metrics,
    _load_actual_values,
    _resolve_positive_label,
    _run_cross_validation,
    _run_cross_validation_classification,
    _rulekit_classification_predict,
    _validate_classification_target,
)
from pipeline.ml_preprocessing import (
    _apply_median_imputer,
    _fit_median_imputer,
    _oversample_minority,
    _preprocess_features,
)
from pipeline.ml_rules import _count_rule_features, _extract_rules, _find_covering_rule
from utils.csv_utils import normalize_media_name
from config import XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE, XGB_RANDOM_STATE

logger = logging.getLogger(__name__)

RULEKIT_WEIGHT = 0.4
XGB_WEIGHT = 0.6
CLASSIFICATION_POSITIVE_THRESHOLD = float(os.getenv("CLASSIFICATION_POSITIVE_THRESHOLD", "0.45"))


# ---------------------------------------------------------------------------
# Training — private branch helpers
# ---------------------------------------------------------------------------

def _make_progress_cb(progress_cb):
    """Wrap an optional progress callback so it can be called safely."""
    def _cb(pct: int, msg: str) -> None:
        if progress_cb:
            try:
                progress_cb(pct, msg)
            except Exception as e:
                logger.debug("progress_cb failed: %s", e)
    return _cb


def _train_classification_branch(pipeline, X, y_raw, target_column, _cb, warnings: list) -> dict:
    """Train RuleKit classifier and update pipeline state in-place."""
    y = _validate_classification_target(y_raw)
    X = X.loc[y.index]
    medians = _fit_median_imputer(X)
    X = _apply_median_imputer(X, medians)
    label_classes = [str(c) for c in pd.Index(pd.unique(y)).tolist()]
    positive_label = _resolve_positive_label(label_classes, y_reference=y)
    X_fit, y_fit = _oversample_minority(X, y)

    _cb(25, "Training RuleKit classifier...")
    rulekit_model = RuleClassifier()
    rulekit_model.fit(X_fit, y_fit)

    _cb(60, "RuleKit classification predictions...")
    y_pred, _ = _rulekit_classification_predict(
        rulekit_model, X,
        positive_label=positive_label,
        positive_threshold=CLASSIFICATION_POSITIVE_THRESHOLD,
        label_classes=label_classes,
    )
    rules = _extract_rules(rulekit_model)
    rulekit_importance = _count_rule_features(rules, list(X.columns))

    _cb(80, "K-fold classification validation...")
    cv_results = _run_cross_validation_classification(
        X, y, positive_threshold=CLASSIFICATION_POSITIVE_THRESHOLD
    )
    if cv_results.get("note"):
        warnings.append(cv_results["note"])
    if positive_label:
        warnings.append(
            f"Classification threshold policy active: "
            f"positive_label='{positive_label}', threshold={CLASSIFICATION_POSITIVE_THRESHOLD:.2f}"
        )

    pipeline.model = rulekit_model
    pipeline.xgb_model = None
    pipeline.rules = rules
    pipeline.mse = pipeline.rulekit_mse = pipeline.xgb_mse = None
    pipeline.cv_mse = pipeline.cv_std = pipeline.cv_mae = None
    pipeline.train_accuracy = round(float(accuracy_score(y, y_pred)), 6)
    pipeline.train_balanced_accuracy = round(float(balanced_accuracy_score(y, y_pred)), 6)
    pipeline.train_f1_macro = round(float(f1_score(y, y_pred, average="macro", zero_division=0)), 6)
    pipeline.train_mcc = round(float(matthews_corrcoef(y, y_pred)), 6)
    pipeline.cv_accuracy = cv_results.get("cv_accuracy")
    pipeline.cv_balanced_accuracy = cv_results.get("cv_balanced_accuracy")
    pipeline.cv_f1_macro = cv_results.get("cv_f1_macro")
    pipeline.cv_precision_macro = cv_results.get("cv_precision_macro")
    pipeline.cv_recall_macro = cv_results.get("cv_recall_macro")
    pipeline.cv_mcc = cv_results.get("cv_mcc")
    pipeline.cv_folds = cv_results.get("n_folds")
    pipeline.warnings = warnings
    pipeline.feature_importance = {"rulekit": rulekit_importance}
    pipeline.is_trained = True
    pipeline.target_variable = target_column
    pipeline.predictions = pipeline.prediction_metrics = None
    pipeline._training_columns = list(X.columns)
    pipeline._scaler_mean = [medians.get(col, 0.0) for col in pipeline._training_columns]
    pipeline._scaler_scale = []
    pipeline._label_classes = label_classes
    pipeline._positive_label = positive_label

    pipeline.save_state()
    _cb(98, "Saving model...")
    return {
        "status": "success",
        "target_mode": "classification",
        "train_accuracy": pipeline.train_accuracy,
        "train_balanced_accuracy": pipeline.train_balanced_accuracy,
        "train_f1_macro": pipeline.train_f1_macro,
        "train_mcc": pipeline.train_mcc,
        **{k: cv_results.get(k) for k in ("cv_accuracy", "cv_balanced_accuracy", "cv_f1_macro",
                                            "cv_precision_macro", "cv_recall_macro", "cv_mcc")},
        "cv_folds": cv_results.get("n_folds"),
        "rules_count": len(rules),
        "rules": rules,
        "feature_spec": pipeline.feature_spec,
        "feature_importance": {"rulekit": rulekit_importance},
        "warnings": warnings,
        "training_data_X": json.loads(pipeline.training_X.to_json(orient="records")),
    }


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


# ---------------------------------------------------------------------------
# Training — public entry point
# ---------------------------------------------------------------------------

def train_model(pipeline, target_column: str, progress_cb=None) -> dict:
    """Train a model from stored training_X + training_Y_df.

    *progress_cb(pct: int, msg: str)* is called at key stages (optional).
    Updates pipeline state in-place and persists to disk.
    Returns a result dict suitable for JSON serialisation.
    """
    if pipeline.training_X is None:
        raise Exception("Phase 2 (feature extraction) must be completed first.")
    if pipeline.training_Y_df is None:
        raise Exception("Missing dataset_Y (CSV with labels). It must be included in the training ZIP.")

    df_gt = pipeline.training_Y_df.copy()
    df_gt = df_gt.rename(columns={df_gt.columns[0]: "media_name"})

    df_x = pipeline.training_X.copy()
    df_x["media_name"] = df_x["media_name"].astype(str).str.strip()
    df_gt["media_name"] = df_gt["media_name"].astype(str).str.strip()
    df_x["_media_join_key"] = df_x["media_name"].apply(normalize_media_name)
    df_gt["_media_join_key"] = df_gt["media_name"].apply(normalize_media_name)

    df_merged = pd.merge(df_x, df_gt, on="_media_join_key", how="inner", suffixes=("_x", "_y"))
    if "media_name_x" in df_merged.columns:
        df_merged["media_name"] = df_merged["media_name_x"]
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
        y_raw = df_merged[target_column]
    else:
        found = [c for c in df_merged.columns if c.lower() == target_column.lower()]
        if found:
            y_raw = df_merged[found[0]]
        else:
            raise Exception(
                f"Column '{target_column}' not found in CSV. "
                f"Available columns: {list(df_gt.columns)}"
            )

    _cb = _make_progress_cb(progress_cb)
    target_mode = getattr(pipeline, "target_mode", "regression")
    _cb(15, "Features ready (no scaling)...")
    warnings: list = []

    if target_mode == "classification":
        return _train_classification_branch(pipeline, X, y_raw, target_column, _cb, warnings)
    return _train_regression_branch(pipeline, X, y_raw, target_column, _cb, warnings)


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

def predict_batch(pipeline, testing_Y_df: pd.DataFrame | None = None, progress_cb=None) -> dict:
    """Predict for all objects in testing_X using the trained model.

    Optionally compares with testing_Y_df for evaluation metrics.
    """
    def _cb(pct: int, msg: str) -> None:
        if progress_cb:
            try:
                progress_cb(pct, msg)
            except Exception as e:
                logger.debug("progress_cb failed: %s", e)

    if not pipeline.is_trained:
        raise Exception("Model is not trained. Complete Phase 3 first.")
    if pipeline.testing_X is None or pipeline.testing_X.empty:
        raise Exception("Missing testing dataset_X. Complete Phase 4 first.")

    _cb(10, "Preprocessing testing features...")
    feature_cols = [c for c in pipeline.feature_spec if c in pipeline.testing_X.columns]
    X_test = _preprocess_features(
        pipeline.testing_X[feature_cols],
        training_columns=pipeline._training_columns,
    )

    target_mode = getattr(pipeline, "target_mode", "regression")

    # ------------------------------------------------------------------ classification
    if target_mode == "classification":
        _cb(25, "RuleKit classification predictions...")
        rulekit_classifier = getattr(pipeline, "model", None)
        xgb_classifier = getattr(pipeline, "xgb_model", None)
        legacy_xgb = rulekit_classifier is None and xgb_classifier is not None
        if rulekit_classifier is None and xgb_classifier is None:
            raise Exception("Classification model is missing. Train Phase 3 again.")

        if legacy_xgb:
            y_pred = np.asarray([str(l) for l in xgb_classifier.predict(X_test)], dtype=object)
            y_pred_proba = None
        else:
            medians = dict(zip(pipeline._training_columns, pipeline._scaler_mean))
            X_test = _apply_median_imputer(X_test, medians)
            positive_label = (
                getattr(pipeline, "_positive_label", None)
                or _resolve_positive_label(getattr(pipeline, "_label_classes", []) or [])
            )
            y_pred, y_pred_proba = _rulekit_classification_predict(
                rulekit_classifier, X_test,
                positive_label=positive_label,
                positive_threshold=CLASSIFICATION_POSITIVE_THRESHOLD,
                label_classes=getattr(pipeline, "_label_classes", None) or None,
            )

        coverage_matrix = None
        if rulekit_classifier is not None and hasattr(rulekit_classifier, "get_coverage_matrix") and pipeline.rules:
            try:
                coverage_matrix = rulekit_classifier.get_coverage_matrix(X_test)
            except Exception as e:
                logger.warning("get_coverage_matrix failed: %s", e)

        actual_values = _load_actual_values(testing_Y_df, pipeline.target_variable, numeric=False)

        results, pred_list, actual_list = [], [], []
        total_items = len(pipeline.testing_X)
        _cb(60, f"Building results (0/{total_items})...")
        report_every = max(1, total_items // 10)

        for row_num, (i, row) in enumerate(pipeline.testing_X.iterrows()):
            if row_num % report_every == 0:
                _cb(60 + int(((row_num + 1) / total_items) * 28),
                    f"Building results ({row_num + 1}/{total_items})...")

            media_name = str(row.get("media_name", f"object_{i}"))
            pred_label = str(y_pred[row_num])
            confidence = float(np.max(y_pred_proba[row_num])) if y_pred_proba is not None else None

            rule_applied = "RuleKit (no single rule match)"
            if coverage_matrix is not None and row_num < len(coverage_matrix):
                covered = np.where(np.asarray(coverage_matrix[row_num]).astype(int) > 0)[0]
                if len(covered) > 0 and covered[0] < len(pipeline.rules):
                    rule_applied = pipeline.rules[int(covered[0])]
            elif pipeline.rules:
                rule_applied = _find_covering_rule(row, pipeline.rules)
            elif legacy_xgb:
                rule_applied = "Legacy XGBoost classification model"

            item = {
                "media_name": media_name,
                "predicted_label": pred_label,
                "confidence": round(confidence, 4) if confidence is not None else None,
                "rule_applied": rule_applied,
                "extracted_features": {k: row[k] for k in pipeline.feature_spec if k in row},
            }
            media_key = normalize_media_name(media_name)
            if media_key in actual_values:
                item["actual_label"] = actual_values[media_key]
                pred_list.append(pred_label)
                actual_list.append(actual_values[media_key])
            results.append(item)

        _cb(92, "Computing evaluation metrics...")
        metrics = _compute_classification_metrics(actual_list, pred_list, results)
        pipeline.predictions = results
        pipeline.prediction_metrics = metrics
        pipeline.save_state()
        return {"predictions": results, "metrics": metrics}

    # ------------------------------------------------------------------ regression
    _cb(25, "RuleKit predictions...")
    reg_medians = dict(zip(pipeline._training_columns, pipeline._scaler_mean))
    X_test = _apply_median_imputer(X_test, reg_medians)
    rulekit_pred = pipeline.model.predict(X_test)

    if hasattr(pipeline, "xgb_model") and pipeline.xgb_model is not None:
        _cb(40, "XGBoost predictions...")
        xgb_pred = pipeline.xgb_model.predict(X_test)
        _cb(55, "Ensemble combination (RuleKit + XGBoost)...")
        predictions = RULEKIT_WEIGHT * rulekit_pred + XGB_WEIGHT * xgb_pred
    else:
        predictions = rulekit_pred

    actual_values = _load_actual_values(testing_Y_df, pipeline.target_variable, numeric=True)

    results, pred_list, actual_list = [], [], []
    total_items = len(pipeline.testing_X)
    _cb(60, f"Building results (0/{total_items})...")
    report_every = max(1, total_items // 10)

    for row_num, (i, row) in enumerate(pipeline.testing_X.iterrows()):
        if row_num % report_every == 0:
            _cb(60 + int(((row_num + 1) / total_items) * 28),
                f"Building results ({row_num + 1}/{total_items})...")

        pred_score = float(predictions[row_num])
        media_name = str(row.get("media_name", f"object_{i}"))
        media_key = normalize_media_name(media_name)
        rule = _find_covering_rule(row, pipeline.rules) if pipeline.rules else "Default rule"

        item = {
            "media_name": media_name,
            "predicted_score": round(pred_score, 4),
            "rule_applied": rule,
            "extracted_features": {k: row[k] for k in pipeline.feature_spec if k in row},
        }
        if media_key in actual_values:
            item["actual_score"] = round(actual_values[media_key], 4)
            pred_list.append(pred_score)
            actual_list.append(actual_values[media_key])
        results.append(item)

    _cb(92, "Computing evaluation metrics...")
    metrics = None
    if pred_list and actual_list:
        pred_arr, actual_arr = np.array(pred_list), np.array(actual_list)
        mse = float(np.mean((pred_arr - actual_arr) ** 2))
        mae = float(np.mean(np.abs(pred_arr - actual_arr)))
        correlation = (
            float(np.corrcoef(pred_arr, actual_arr)[0, 1])
            if len(pred_arr) > 1 and np.std(pred_arr) > 0 and np.std(actual_arr) > 0
            else None
        )
        metrics = {
            "mode": "regression",
            "mse": round(mse, 6),
            "mae": round(mae, 6),
            "correlation": round(correlation, 4) if correlation is not None else None,
            "matched_count": len(pred_list),
            "total_count": len(results),
        }

    pipeline.predictions = results
    pipeline.prediction_metrics = metrics
    pipeline.save_state()
    return {"predictions": results, "metrics": metrics}
