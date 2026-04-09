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
import re

import numpy as np
import pandas as pd
from rulekit.classification import RuleClassifier
from rulekit.regression import RuleRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from xgboost import XGBRegressor
from utils.csv_utils import normalize_media_name

logger = logging.getLogger(__name__)

# Ensemble weights (RuleKit is interpretable, XGBoost is more accurate)
RULEKIT_WEIGHT = 0.4
XGB_WEIGHT = 0.6
CLASSIFICATION_POSITIVE_LABEL = os.getenv("CLASSIFICATION_POSITIVE_LABEL", "").strip()
CLASSIFICATION_POSITIVE_THRESHOLD = float(os.getenv("CLASSIFICATION_POSITIVE_THRESHOLD", "0.45"))
CV_MAX_FOLDS = max(2, int(os.getenv("CV_MAX_FOLDS", "3")))


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

    X_bal = pd.concat(parts_X, ignore_index=True)
    y_bal = pd.concat(parts_y, ignore_index=True)
    return X_bal, y_bal


def _resolve_positive_label(labels: list[str], y_reference: pd.Series | None = None) -> str | None:
    if not labels or len(labels) != 2:
        return None
    if CLASSIFICATION_POSITIVE_LABEL and CLASSIFICATION_POSITIVE_LABEL in labels:
        return CLASSIFICATION_POSITIVE_LABEL
    if y_reference is not None and not y_reference.empty:
        counts = y_reference.astype(str).value_counts().to_dict()
        return min(labels, key=lambda lbl: counts.get(lbl, 0))
    for preferred in ("1", "true", "positive", "malignant", "yes"):
        if preferred in labels:
            return preferred
    return labels[-1]
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


def _find_covering_rule(row: pd.Series, rules: list[str]) -> str:
    """Find the first RuleKit rule whose IF-conditions are satisfied by row values.

    Rules have the format: IF feature >= val AND feature <= val THEN ...
    Returns the full rule string or a fallback label.
    """
    for rule_str in rules:
        # Extract the condition part (before THEN)
        match = re.match(r"IF\s+(.+?)\s+THEN", rule_str, re.IGNORECASE)
        if not match:
            continue
        conditions_str = match.group(1)
        # Split on AND
        conditions = re.split(r"\s+AND\s+", conditions_str, flags=re.IGNORECASE)
        all_met = True
        for cond in conditions:
            cond = cond.strip()
            # Parse "feature op value"
            m = re.match(r"(.+?)\s*(>=|<=|>|<|=)\s*(.+)", cond)
            if not m:
                all_met = False
                break
            feat, op, val_str = m.group(1).strip(), m.group(2), m.group(3).strip()
            if feat not in row.index:
                all_met = False
                break
            try:
                row_val = float(row[feat])
                threshold = float(val_str)
            except (ValueError, TypeError):
                all_met = False
                break
            if op == ">=" and not (row_val >= threshold):
                all_met = False
            elif op == "<=" and not (row_val <= threshold):
                all_met = False
            elif op == ">" and not (row_val > threshold):
                all_met = False
            elif op == "<" and not (row_val < threshold):
                all_met = False
            elif op == "=" and not (abs(row_val - threshold) < 1e-9):
                all_met = False
            if not all_met:
                break
        if all_met:
            return rule_str
    return "RuleKit (no single rule match)"


def _count_rule_features(rules: list[str], feature_names: list[str]) -> dict:
    """Count how often each feature appears in RuleKit rules."""
    counts = {}
    for feat in feature_names:
        count = 0
        for rule in rules:
            if feat in rule:
                count += 1
        if count > 0:
            counts[feat] = count
    total = sum(counts.values()) or 1
    return {k: round(v / total, 4) for k, v in sorted(counts.items(), key=lambda x: -x[1])}


def _run_cross_validation(X: pd.DataFrame, y: pd.Series, n_splits: int = CV_MAX_FOLDS) -> dict:
    """Run K-fold CV with the ensemble approach. Returns CV metrics."""
    n_splits = min(n_splits, len(X))
    if n_splits < 2:
        return {"cv_mse": None, "cv_std": None, "cv_mae": None, "note": "Too few samples for CV"}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_mse = []
    fold_mae = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        medians = _fit_median_imputer(X_train)
        X_train = _apply_median_imputer(X_train, medians)
        X_val = _apply_median_imputer(X_val, medians)

        # RuleKit
        rk = RuleRegressor()
        try:
            rk.fit(X_train, y_train)
            rk_pred = rk.predict(X_val)
        except Exception as e:
            logger.warning("RuleRegressor CV fold failed, using mean fallback: %s", e)
            rk_pred = np.full(len(X_val), y_train.mean())

        # XGBoost
        xgb = XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42, verbosity=0,
        )
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_val)

        # Ensemble
        ensemble_pred = RULEKIT_WEIGHT * rk_pred + XGB_WEIGHT * xgb_pred

        fold_mse.append(float(mean_squared_error(y_val, ensemble_pred)))
        fold_mae.append(float(mean_absolute_error(y_val, ensemble_pred)))

    return {
        "cv_mse": round(float(np.mean(fold_mse)), 6),
        "cv_std": round(float(np.std(fold_mse)), 6),
        "cv_mae": round(float(np.mean(fold_mae)), 6),
        "n_folds": n_splits,
    }


def _run_cross_validation_classification(
    X: pd.DataFrame, y_labels: pd.Series, n_splits: int = CV_MAX_FOLDS
) -> dict:
    """Run stratified CV for classification with RuleKit only."""
    class_counts = y_labels.value_counts()
    positive_counts = class_counts[class_counts > 0]
    max_splits = min(n_splits, len(X), int(positive_counts.min())) if len(positive_counts) else 0
    if max_splits < 2:
        return {
            "cv_accuracy": None,
            "cv_balanced_accuracy": None,
            "cv_f1_macro": None,
            "cv_precision_macro": None,
            "cv_recall_macro": None,
            "cv_mcc": None,
            "n_folds": max_splits,
            "note": "Too few samples per class for stratified CV",
        }

    kf = StratifiedKFold(n_splits=max_splits, shuffle=True, random_state=42)
    fold_acc = []
    fold_bal_acc = []
    fold_f1 = []
    fold_prec = []
    fold_rec = []
    fold_mcc = []
    positive_label = _resolve_positive_label([str(c) for c in pd.Index(pd.unique(y_labels)).tolist()], y_reference=y_labels)

    for train_idx, val_idx in kf.split(X, y_labels):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_labels.iloc[train_idx], y_labels.iloc[val_idx]

        medians = _fit_median_imputer(X_train)
        X_train = _apply_median_imputer(X_train, medians)
        X_val = _apply_median_imputer(X_val, medians)
        X_train, y_train = _oversample_minority(X_train, y_train)

        rk_clf = RuleClassifier()
        rk_clf.fit(X_train, y_train)
        y_pred, _ = _rulekit_classification_predict(
            rk_clf,
            X_val,
            positive_label=positive_label,
            positive_threshold=CLASSIFICATION_POSITIVE_THRESHOLD,
        )

        fold_acc.append(float(accuracy_score(y_val, y_pred)))
        fold_bal_acc.append(float(balanced_accuracy_score(y_val, y_pred)))
        fold_f1.append(float(f1_score(y_val, y_pred, average="macro", zero_division=0)))
        fold_prec.append(float(precision_score(y_val, y_pred, average="macro", zero_division=0)))
        fold_rec.append(float(recall_score(y_val, y_pred, average="macro", zero_division=0)))
        fold_mcc.append(float(matthews_corrcoef(y_val, y_pred)))

    return {
        "cv_accuracy": round(float(np.mean(fold_acc)), 6),
        "cv_balanced_accuracy": round(float(np.mean(fold_bal_acc)), 6),
        "cv_f1_macro": round(float(np.mean(fold_f1)), 6),
        "cv_precision_macro": round(float(np.mean(fold_prec)), 6),
        "cv_recall_macro": round(float(np.mean(fold_rec)), 6),
        "cv_mcc": round(float(np.mean(fold_mcc)), 6),
        "n_folds": max_splits,
    }


def _ensure_probability_matrix(proba: np.ndarray, class_count: int) -> np.ndarray:
    arr = np.asarray(proba, dtype=float)
    if arr.ndim == 1:
        if class_count == 2:
            return np.column_stack([1.0 - arr, arr])
        return arr.reshape(-1, 1)
    return arr


def _align_probability_columns(
    proba: np.ndarray,
    source_labels: list[str],
    target_labels: list[str],
) -> np.ndarray:
    arr = _ensure_probability_matrix(proba, len(source_labels))
    aligned = np.zeros((arr.shape[0], len(target_labels)), dtype=float)
    source_map = {str(label): idx for idx, label in enumerate(source_labels)}
    for target_idx, label in enumerate(target_labels):
        src_idx = source_map.get(str(label))
        if src_idx is not None and src_idx < arr.shape[1]:
            aligned[:, target_idx] = arr[:, src_idx]

    row_sums = aligned.sum(axis=1, keepdims=True)
    nonzero = row_sums.squeeze(axis=1) > 0
    if np.any(nonzero):
        aligned[nonzero] = aligned[nonzero] / row_sums[nonzero]
    return aligned


def _hard_predictions_to_probability(pred_labels: np.ndarray, label_classes: list[str]) -> np.ndarray:
    proba = np.zeros((len(pred_labels), len(label_classes)), dtype=float)
    label_to_idx = {str(label): idx for idx, label in enumerate(label_classes)}
    for row_idx, label in enumerate(pred_labels):
        cls_idx = label_to_idx.get(str(label))
        if cls_idx is not None:
            proba[row_idx, cls_idx] = 1.0
    return proba


def _proba_to_labels(proba: np.ndarray, label_classes: list[str]) -> np.ndarray:
    arr = np.asarray(proba, dtype=float)
    return np.array(
        [label_classes[int(idx)] for idx in np.argmax(arr, axis=1)],
        dtype=object,
    )


def _rulekit_classification_predict(
    rulekit_model,
    X: pd.DataFrame,
    positive_label: str | None = None,
    positive_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if rulekit_model is None:
        raise Exception("Classification RuleKit model is missing.")

    pred_labels = np.asarray(rulekit_model.predict(X), dtype=object)
    pred_proba = None
    if hasattr(rulekit_model, "predict_proba"):
        raw_proba = rulekit_model.predict_proba(X)
        source_labels = [
            str(label)
            for label in getattr(rulekit_model, "label_unique_values", pd.Index(pd.unique(pred_labels)).tolist())
        ]
        target_labels = source_labels or [str(label) for label in pd.Index(pd.unique(pred_labels)).tolist()]
        if target_labels:
            pred_proba = _align_probability_columns(raw_proba, source_labels, target_labels)
            pred_labels = _proba_to_labels(pred_proba, target_labels)
            if (
                positive_label
                and positive_threshold is not None
                and len(target_labels) == 2
                and positive_label in target_labels
            ):
                pos_idx = target_labels.index(positive_label)
                neg_label = target_labels[1 - pos_idx]
                pred_labels = np.where(
                    pred_proba[:, pos_idx] >= float(positive_threshold),
                    positive_label,
                    neg_label,
                ).astype(object)
    return pred_labels, pred_proba


def _resolve_eval_target_column(testing_Y_df: pd.DataFrame, target_variable: str) -> str | None:
    """Find the exact evaluation target column or fail loudly.

    Silent fallbacks can produce plausible-looking but incorrect metrics, so when
    evaluation labels are provided we require the trained target column to exist.
    """
    if testing_Y_df is None or testing_Y_df.empty:
        return None

    for col in testing_Y_df.columns:
        if col == "media_name":
            continue
        if col == target_variable or col.lower() == target_variable.lower():
            return col

    available = [c for c in testing_Y_df.columns if c != "media_name"]
    raise Exception(
        "Evaluation labels CSV must contain the same target column used for training: "
        f"'{target_variable}'. Available columns: {available}"
    )


def _validate_classification_target(y_raw: pd.Series) -> pd.Series:
    """Validate that the selected target behaves like a categorical label."""
    y = (
        y_raw.astype(str)
        .str.strip()
        .replace({"": None, "nan": None, "none": None, "null": None})
        .dropna()
    )
    if y.empty:
        raise Exception("Classification target column is empty after removing missing labels.")

    unique_count = int(y.nunique())
    if unique_count < 2:
        raise Exception("Classification target must contain at least 2 distinct classes.")

    unique_ratio = unique_count / max(len(y), 1)
    value_counts = y.value_counts()
    singleton_ratio = float((value_counts == 1).mean()) if not value_counts.empty else 0.0
    tiny_class_ratio = float((value_counts <= 2).mean()) if not value_counts.empty else 0.0
    numeric_candidate = pd.to_numeric(y, errors="coerce")
    numeric_ratio = float(numeric_candidate.notna().mean()) if len(y) else 0.0
    if numeric_ratio > 0.95 and unique_count > 10 and unique_ratio > 0.3:
        raise Exception(
            "Classification mode expects repeated class labels, but the selected target column "
            f"looks continuous or high-cardinality ({unique_count} unique values across {len(y)} rows). "
            "Switch to regression mode or choose a categorical label column."
        )
    if unique_count > 20 and unique_ratio > 0.3 and (
        singleton_ratio > 0.5 or tiny_class_ratio > 0.8
    ):
        raise Exception(
            "Classification mode expects a small, repeated set of class labels, but the selected "
            f"target column looks identifier-like or too sparse ({unique_count} unique values across "
            f"{len(y)} rows, singleton class ratio {singleton_ratio:.2f}). "
            "Choose a categorical label column with repeated classes."
        )

    return y


def train_model(pipeline, target_column: str, progress_cb=None) -> dict:
    """Train a model from stored training_X + training_Y_df.

    *progress_cb(pct: int, msg: str)* is called at key stages (optional).
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

    df_gt = pipeline.training_Y_df.copy()
    join_col = df_gt.columns[0]
    df_gt = df_gt.rename(columns={join_col: "media_name"})

    df_x = pipeline.training_X.copy()
    df_x["media_name"] = df_x["media_name"].astype(str).str.strip()
    df_gt["media_name"] = df_gt["media_name"].astype(str).str.strip()

    df_x["_media_join_key"] = df_x["media_name"].apply(normalize_media_name)
    df_gt["_media_join_key"] = df_gt["media_name"].apply(normalize_media_name)

    df_merged = pd.merge(
        df_x,
        df_gt,
        on="_media_join_key",
        how="inner",
        suffixes=("_x", "_y"),
    )
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

    def _cb(pct: int, msg: str) -> None:
        if progress_cb:
            try:
                progress_cb(pct, msg)
            except Exception as e:
                logger.debug("progress_cb failed: %s", e)

    target_mode = getattr(pipeline, "target_mode", "regression")

    # --- Feature matrix ready (no scaling by request) ---
    _cb(15, "Příznaky připraveny (bez škálování)...")

    warnings = []

    if target_mode == "classification":
        y = _validate_classification_target(y_raw)
        X = X.loc[y.index]
        medians = _fit_median_imputer(X)
        X = _apply_median_imputer(X, medians)
        label_classes = [str(c) for c in pd.Index(pd.unique(y)).tolist()]
        positive_label = _resolve_positive_label(label_classes, y_reference=y)
        X_fit, y_fit = _oversample_minority(X, y)

        _cb(25, "Trénuji RuleKit klasifikátor...")
        rulekit_model = RuleClassifier()
        rulekit_model.fit(X_fit, y_fit)
        _cb(60, "RuleKit klasifikační predikce...")
        y_pred, _ = _rulekit_classification_predict(
            rulekit_model,
            X,
            positive_label=positive_label,
            positive_threshold=CLASSIFICATION_POSITIVE_THRESHOLD,
        )

        train_accuracy = round(float(accuracy_score(y, y_pred)), 6)
        train_balanced_accuracy = round(float(balanced_accuracy_score(y, y_pred)), 6)
        train_f1_macro = round(float(f1_score(y, y_pred, average="macro", zero_division=0)), 6)
        train_mcc = round(float(matthews_corrcoef(y, y_pred)), 6)

        _cb(80, "K-fold validace klasifikace...")
        cv_results = _run_cross_validation_classification(X, y)
        if cv_results.get("note"):
            warnings.append(cv_results["note"])

        rules = (
            [str(rule) for rule in rulekit_model.model.rules]
            if hasattr(rulekit_model, "model")
            else []
        )
        rulekit_importance = _count_rule_features(rules, list(X.columns))

        pipeline.model = rulekit_model
        pipeline.xgb_model = None
        pipeline.rules = rules
        pipeline.mse = None
        pipeline.rulekit_mse = None
        pipeline.xgb_mse = None
        pipeline.cv_mse = None
        pipeline.cv_std = None
        pipeline.cv_mae = None
        pipeline.train_accuracy = train_accuracy
        pipeline.train_balanced_accuracy = train_balanced_accuracy
        pipeline.train_f1_macro = train_f1_macro
        pipeline.train_mcc = train_mcc
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
        pipeline.predictions = None
        pipeline.prediction_metrics = None
        pipeline._training_columns = list(X.columns)
        pipeline._scaler_mean = [medians.get(col, 0.0) for col in pipeline._training_columns]
        pipeline._scaler_scale = []
        pipeline._label_classes = label_classes

        if positive_label:
            warnings.append(
                "Classification threshold policy active: "
                f"positive_label='{positive_label}', threshold={CLASSIFICATION_POSITIVE_THRESHOLD:.2f}"
            )

        pipeline.save_state()
        _cb(98, "Ukládám model...")

        return {
            "status": "success",
            "target_mode": "classification",
            "train_accuracy": train_accuracy,
            "train_balanced_accuracy": train_balanced_accuracy,
            "train_f1_macro": train_f1_macro,
            "train_mcc": train_mcc,
            "cv_accuracy": cv_results.get("cv_accuracy"),
            "cv_balanced_accuracy": cv_results.get("cv_balanced_accuracy"),
            "cv_f1_macro": cv_results.get("cv_f1_macro"),
            "cv_precision_macro": cv_results.get("cv_precision_macro"),
            "cv_recall_macro": cv_results.get("cv_recall_macro"),
            "cv_mcc": cv_results.get("cv_mcc"),
            "cv_folds": cv_results.get("n_folds"),
            "rules_count": len(rules),
            "rules": rules,
            "feature_spec": pipeline.feature_spec,
            "feature_importance": {"rulekit": rulekit_importance},
            "warnings": warnings,
            "training_data_X": json.loads(pipeline.training_X.to_json(orient="records")),
        }

    # ---------------- Regression branch ----------------
    y = pd.to_numeric(y_raw, errors="coerce")
    valid_mask = ~y.isna()
    if not valid_mask.any():
        raise Exception("Target column is not numeric. For non-numeric labels switch to classification mode.")
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]
    reg_medians = _fit_median_imputer(X)
    X = _apply_median_imputer(X, reg_medians)

    _cb(25, "Trénuji RuleKit (indukce pravidel)...")
    logger.info("Training RuleKit model...")
    rulekit_model = RuleRegressor()
    rulekit_model.fit(X, y)

    rules = (
        [str(rule) for rule in rulekit_model.model.rules]
        if hasattr(rulekit_model, "model")
        else []
    )
    rulekit_pred = rulekit_model.predict(X)
    rulekit_mse = round(float(mean_squared_error(y, rulekit_pred)), 6)

    _cb(60, "Trénuji XGBoost...")
    logger.info("Training XGBoost model...")
    xgb_model = XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=42, verbosity=0,
    )
    xgb_model.fit(X, y)
    xgb_pred = xgb_model.predict(X)
    xgb_mse = round(float(mean_squared_error(y, xgb_pred)), 6)

    _cb(75, "Ensemble predikce...")
    ensemble_pred = RULEKIT_WEIGHT * rulekit_pred + XGB_WEIGHT * xgb_pred
    ensemble_mse = round(float(mean_squared_error(y, ensemble_pred)), 6)

    _cb(80, "K-fold křížová validace...")
    logger.info("Running %d-fold cross-validation...", min(5, len(X)))
    cv_results = _run_cross_validation(X, y)
    logger.info("CV results: %s", cv_results)
    if cv_results.get("note"):
        warnings.append(cv_results["note"])

    xgb_importance = {}
    if hasattr(xgb_model, "feature_importances_"):
        for fname, imp in zip(X.columns, xgb_model.feature_importances_):
            if imp > 0.001:
                xgb_importance[fname] = round(float(imp), 4)
        xgb_importance = dict(sorted(xgb_importance.items(), key=lambda x: -x[1]))

    rulekit_importance = _count_rule_features(rules, list(X.columns))

    pipeline.model = rulekit_model
    pipeline.xgb_model = xgb_model
    pipeline.rules = rules
    pipeline.mse = ensemble_mse
    pipeline.rulekit_mse = rulekit_mse
    pipeline.xgb_mse = xgb_mse
    pipeline.cv_mse = cv_results.get("cv_mse")
    pipeline.cv_std = cv_results.get("cv_std")
    pipeline.cv_mae = cv_results.get("cv_mae")
    pipeline.train_accuracy = None
    pipeline.train_balanced_accuracy = None
    pipeline.train_f1_macro = None
    pipeline.train_mcc = None
    pipeline.cv_accuracy = None
    pipeline.cv_balanced_accuracy = None
    pipeline.cv_f1_macro = None
    pipeline.cv_precision_macro = None
    pipeline.cv_recall_macro = None
    pipeline.cv_mcc = None
    pipeline.cv_folds = cv_results.get("n_folds")
    pipeline.feature_importance = {"xgboost": xgb_importance, "rulekit": rulekit_importance}
    pipeline.is_trained = True
    pipeline.target_variable = target_column
    pipeline.predictions = None
    pipeline.prediction_metrics = None
    pipeline._training_columns = list(X.columns)
    pipeline._scaler_mean = [reg_medians.get(col, 0.0) for col in pipeline._training_columns]
    pipeline._scaler_scale = []
    pipeline._label_classes = []

    if cv_results.get("cv_mse") is not None and cv_results["cv_mse"] > 2 * ensemble_mse:
        warnings.append(
            f"Possible overfitting: CV MSE ({cv_results['cv_mse']:.6f}) "
            f"is much higher than training MSE ({ensemble_mse:.6f}). "
            f"Consider adding more training data."
        )
    pipeline.warnings = warnings

    pipeline.save_state()
    _cb(98, "Ukládám model...")

    return {
        "status": "success",
        "target_mode": "regression",
        "mse": ensemble_mse,
        "rulekit_mse": rulekit_mse,
        "xgb_mse": xgb_mse,
        "cv_mse": cv_results.get("cv_mse"),
        "cv_std": cv_results.get("cv_std"),
        "cv_mae": cv_results.get("cv_mae"),
        "cv_folds": cv_results.get("n_folds"),
        "rules_count": len(rules),
        "rules": rules,
        "feature_spec": pipeline.feature_spec,
        "feature_importance": {
            "xgboost": xgb_importance,
            "rulekit": rulekit_importance,
        },
        "warnings": warnings,
        "training_data_X": json.loads(pipeline.training_X.to_json(orient="records")),
    }


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

    _cb(10, "Předzpracovávám testovací příznaky...")
    feature_cols = [c for c in pipeline.feature_spec if c in pipeline.testing_X.columns]
    X_test = _preprocess_features(
        pipeline.testing_X[feature_cols],
        training_columns=pipeline._training_columns,
    )

    target_mode = getattr(pipeline, "target_mode", "regression")

    if target_mode == "classification":
        _cb(25, "RuleKit klasifikační predikce...")
        rulekit_classifier = getattr(pipeline, "model", None)
        xgb_classifier = getattr(pipeline, "xgb_model", None)
        legacy_xgb = rulekit_classifier is None and xgb_classifier is not None
        if rulekit_classifier is None and xgb_classifier is None:
            raise Exception("Classification model is missing. Train Phase 3 again.")

        if legacy_xgb:
            legacy_pred = xgb_classifier.predict(X_test)
            y_pred = np.asarray([str(label) for label in legacy_pred], dtype=object)
            y_pred_proba = None
        else:
            medians = dict(zip(pipeline._training_columns, pipeline._scaler_mean))
            X_test = _apply_median_imputer(X_test, medians)
            positive_label = _resolve_positive_label(getattr(pipeline, "_label_classes", []) or [])
            y_pred, y_pred_proba = _rulekit_classification_predict(
                rulekit_classifier,
                X_test,
                positive_label=positive_label,
                positive_threshold=CLASSIFICATION_POSITIVE_THRESHOLD,
            )

        coverage_matrix = None
        if rulekit_classifier is not None and hasattr(rulekit_classifier, "get_coverage_matrix") and pipeline.rules:
            try:
                coverage_matrix = rulekit_classifier.get_coverage_matrix(X_test)
            except Exception as e:
                logger.warning("get_coverage_matrix failed: %s", e)
                coverage_matrix = None

        actual_values: dict[str, str] = {}
        if testing_Y_df is not None:
            join_col = testing_Y_df.columns[0]
            testing_Y_df = testing_Y_df.rename(columns={join_col: "media_name"})
            testing_Y_df["media_name"] = testing_Y_df["media_name"].astype(str).str.strip()
            target_col = _resolve_eval_target_column(testing_Y_df, pipeline.target_variable)
            if target_col:
                for _, row in testing_Y_df.iterrows():
                    actual_values[normalize_media_name(row["media_name"])] = str(row[target_col])

        results = []
        pred_list: list[str] = []
        actual_list: list[str] = []
        total_items = len(pipeline.testing_X)
        _cb(60, f"Sestavuji výsledky (0/{total_items})...")
        report_every = max(1, total_items // 10)

        for row_num, (i, row) in enumerate(pipeline.testing_X.iterrows()):
            if row_num % report_every == 0:
                pct = 60 + int(((row_num + 1) / total_items) * 28)
                _cb(pct, f"Sestavuji výsledky ({row_num + 1}/{total_items})...")

            media_name = str(row.get("media_name", f"object_{i}"))
            media_key = normalize_media_name(media_name)
            raw_pred = y_pred[row_num]
            pred_label = str(raw_pred)
            confidence = None
            if y_pred_proba is not None:
                confidence = float(np.max(y_pred_proba[row_num]))
            rule_applied = "RuleKit (no single rule match)"
            if coverage_matrix is not None and row_num < len(coverage_matrix):
                covered = np.where(np.asarray(coverage_matrix[row_num]).astype(int) > 0)[0]
                if len(covered) > 0 and covered[0] < len(pipeline.rules):
                    rule_applied = pipeline.rules[int(covered[0])]
            elif legacy_xgb:
                rule_applied = "Legacy XGBoost classification model"

            item = {
                "media_name": media_name,
                "predicted_label": pred_label,
                "confidence": round(confidence, 4) if confidence is not None else None,
                "rule_applied": rule_applied,
                "extracted_features": {
                    k: row[k] for k in pipeline.feature_spec if k in row
                },
            }

            if media_key in actual_values:
                item["actual_label"] = actual_values[media_key]
                pred_list.append(pred_label)
                actual_list.append(actual_values[media_key])

            results.append(item)

        _cb(92, "Výpočet evaluačních metrik...")
        metrics = None
        if pred_list and actual_list:
            labels = sorted(set(actual_list) | set(pred_list))
            acc = float(accuracy_score(actual_list, pred_list))
            bal_acc = float(balanced_accuracy_score(actual_list, pred_list))
            f1m = float(f1_score(actual_list, pred_list, average="macro", zero_division=0))
            prec = float(precision_score(actual_list, pred_list, average="macro", zero_division=0))
            rec = float(recall_score(actual_list, pred_list, average="macro", zero_division=0))
            mcc = float(matthews_corrcoef(actual_list, pred_list))
            cm = confusion_matrix(actual_list, pred_list, labels=labels)
            class_prec, class_rec, class_f1, class_support = precision_recall_fscore_support(
                actual_list,
                pred_list,
                labels=labels,
                zero_division=0,
            )
            class_metrics = [
                {
                    "label": label,
                    "precision": round(float(class_prec[idx]), 6),
                    "recall": round(float(class_rec[idx]), 6),
                    "f1": round(float(class_f1[idx]), 6),
                    "support": int(class_support[idx]),
                }
                for idx, label in enumerate(labels)
            ]
            avg_confidence = None
            correct_confidence_avg = None
            incorrect_confidence_avg = None
            confidence_values = [
                float(item["confidence"]) for item in results if item.get("confidence") is not None
            ]
            if confidence_values:
                avg_confidence = round(float(np.mean(confidence_values)), 6)
            matched_items = [
                item for item in results
                if item.get("actual_label") is not None and item.get("confidence") is not None
            ]
            correct_conf = [
                float(item["confidence"]) for item in matched_items
                if item.get("predicted_label") == item.get("actual_label")
            ]
            incorrect_conf = [
                float(item["confidence"]) for item in matched_items
                if item.get("predicted_label") != item.get("actual_label")
            ]
            if correct_conf:
                correct_confidence_avg = round(float(np.mean(correct_conf)), 6)
            if incorrect_conf:
                incorrect_confidence_avg = round(float(np.mean(incorrect_conf)), 6)
            metrics = {
                "mode": "classification",
                "accuracy": round(acc, 6),
                "balanced_accuracy": round(bal_acc, 6),
                "f1_macro": round(f1m, 6),
                "precision_macro": round(prec, 6),
                "recall_macro": round(rec, 6),
                "mcc": round(mcc, 6),
                "labels": labels,
                "confusion_matrix": cm.tolist(),
                "class_metrics": class_metrics,
                "avg_confidence": avg_confidence,
                "correct_confidence_avg": correct_confidence_avg,
                "incorrect_confidence_avg": incorrect_confidence_avg,
                "matched_count": len(pred_list),
                "total_count": len(results),
            }

        pipeline.predictions = results
        pipeline.prediction_metrics = metrics
        pipeline.save_state()
        return {"predictions": results, "metrics": metrics}

    # ---------------- Regression branch ----------------
    _cb(25, "RuleKit predikce...")
    reg_medians = dict(zip(pipeline._training_columns, pipeline._scaler_mean))
    X_test = _apply_median_imputer(X_test, reg_medians)
    rulekit_pred = pipeline.model.predict(X_test)
    X_test_scaled = X_test

    if hasattr(pipeline, 'xgb_model') and pipeline.xgb_model is not None:
        _cb(40, "XGBoost predikce...")
        xgb_pred = pipeline.xgb_model.predict(X_test_scaled)
        _cb(55, "Ensemble kombinace (RuleKit + XGBoost)...")
        predictions = RULEKIT_WEIGHT * rulekit_pred + XGB_WEIGHT * xgb_pred
    else:
        predictions = rulekit_pred

    actual_values: dict[str, float] = {}
    if testing_Y_df is not None:
        join_col = testing_Y_df.columns[0]
        testing_Y_df = testing_Y_df.rename(columns={join_col: "media_name"})
        testing_Y_df["media_name"] = testing_Y_df["media_name"].astype(str).str.strip()
        target_col = _resolve_eval_target_column(testing_Y_df, pipeline.target_variable)
        if target_col:
            for _, row in testing_Y_df.iterrows():
                actual_values[normalize_media_name(row["media_name"])] = float(row[target_col])

    results = []
    pred_list: list[float] = []
    actual_list: list[float] = []
    total_items = len(pipeline.testing_X)
    _cb(60, f"Sestavuji výsledky (0/{total_items})...")
    report_every = max(1, total_items // 10)

    for row_num, (i, row) in enumerate(pipeline.testing_X.iterrows()):
        if row_num % report_every == 0:
            pct = 60 + int(((row_num + 1) / total_items) * 28)
            _cb(pct, f"Sestavuji výsledky ({row_num + 1}/{total_items})...")
        pred_score = float(predictions[row_num])
        media_name = str(row.get("media_name", f"object_{i}"))
        media_key = normalize_media_name(media_name)
        rule = _find_covering_rule(row, pipeline.rules) if pipeline.rules else "Default rule"

        item = {
            "media_name": media_name,
            "predicted_score": round(pred_score, 4),
            "rule_applied": rule,
            "extracted_features": {
                k: row[k] for k in pipeline.feature_spec if k in row
            },
        }

        if media_key in actual_values:
            item["actual_score"] = round(actual_values[media_key], 4)
            pred_list.append(pred_score)
            actual_list.append(actual_values[media_key])

        results.append(item)

    _cb(92, "Výpočet evaluačních metrik...")
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
