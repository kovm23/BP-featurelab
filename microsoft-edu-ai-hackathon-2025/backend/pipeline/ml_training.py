"""Phase 3: Model training (RuleKit + XGBoost ensemble), and Phase 5: batch prediction.

Training pipeline:
  1. Preprocess features (one-hot encoding, null handling)
  2. Normalise with StandardScaler
  3. Train RuleKit RuleRegressor (interpretable rules)
  4. Train XGBoost Regressor (high accuracy)
  5. Ensemble: weighted average of both predictions
  6. K-fold cross-validation for realistic error estimate
  7. Feature importance from XGBoost + rule frequency from RuleKit
"""
import json
import logging
import re

import numpy as np
import pandas as pd
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
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

logger = logging.getLogger(__name__)

# Ensemble weights (RuleKit is interpretable, XGBoost is more accurate)
RULEKIT_WEIGHT = 0.4
XGB_WEIGHT = 0.6


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
    return "Ensemble (no single rule match)"


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


def _run_cross_validation(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
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

        # RuleKit
        rk = RuleRegressor()
        try:
            rk.fit(X_train, y_train)
            rk_pred = rk.predict(X_val)
        except Exception:
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
    X: pd.DataFrame, y_encoded: np.ndarray, n_splits: int = 5
) -> dict:
    """Run K-fold CV for classification with XGBoost classifier."""
    class_counts = np.bincount(y_encoded)
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

    for train_idx, val_idx in kf.split(X, y_encoded):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

        clf = _build_classification_model(len(np.unique(y_encoded)))
        clf.fit(X_train, y_train, sample_weight=_classification_sample_weights(y_train))
        y_pred = clf.predict(X_val)

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


def _classification_sample_weights(y_encoded: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights for imbalanced classification."""
    classes, counts = np.unique(y_encoded, return_counts=True)
    total = counts.sum()
    n_classes = len(classes)
    weight_map = {
        int(cls): float(total / (n_classes * count))
        for cls, count in zip(classes, counts)
        if count > 0
    }
    return np.array([weight_map[int(cls)] for cls in y_encoded], dtype=float)


def _build_classification_model(n_classes: int) -> XGBClassifier:
    params = {
        "n_estimators": 180,
        "max_depth": 5,
        "learning_rate": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
        "verbosity": 0,
        "eval_metric": "logloss" if n_classes <= 2 else "mlogloss",
        "objective": "binary:logistic" if n_classes <= 2 else "multi:softprob",
    }
    if n_classes > 2:
        params["num_class"] = n_classes
    return XGBClassifier(**params)


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
    """Train RuleKit + XGBoost ensemble from stored training_X + training_Y_df.

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
            except Exception:
                pass

    target_mode = getattr(pipeline, "target_mode", "regression")

    # --- Feature matrix ready (no scaling by request) ---
    _cb(15, "Příznaky připraveny (bez škálování)...")

    warnings = []

    if target_mode == "classification":
        y = _validate_classification_target(y_raw)
        X = X.loc[y.index]
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        _cb(25, "Trénuji XGBoost klasifikátor...")
        xgb_model = _build_classification_model(len(le.classes_))
        xgb_model.fit(X, y_encoded, sample_weight=_classification_sample_weights(y_encoded))
        xgb_pred = xgb_model.predict(X)

        train_accuracy = round(float(accuracy_score(y_encoded, xgb_pred)), 6)
        train_balanced_accuracy = round(float(balanced_accuracy_score(y_encoded, xgb_pred)), 6)
        train_f1_macro = round(float(f1_score(y_encoded, xgb_pred, average="macro", zero_division=0)), 6)
        train_mcc = round(float(matthews_corrcoef(y_encoded, xgb_pred)), 6)

        _cb(80, "K-fold validace klasifikace...")
        cv_results = _run_cross_validation_classification(X, y_encoded)
        if cv_results.get("note"):
            warnings.append(cv_results["note"])

        xgb_importance = {}
        if hasattr(xgb_model, "feature_importances_"):
            for fname, imp in zip(X.columns, xgb_model.feature_importances_):
                if imp > 0.001:
                    xgb_importance[fname] = round(float(imp), 4)
            xgb_importance = dict(sorted(xgb_importance.items(), key=lambda x: -x[1]))

        pipeline.model = None
        pipeline.xgb_model = xgb_model
        pipeline.scaler = None
        pipeline.rules = []
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
        pipeline.feature_importance = {"xgboost": xgb_importance, "rulekit": {}}
        pipeline.is_trained = True
        pipeline.target_variable = target_column
        pipeline._training_columns = list(X.columns)
        pipeline._scaler_mean = []
        pipeline._scaler_scale = []
        pipeline._label_classes = [str(c) for c in le.classes_]

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
            "rules_count": 0,
            "rules": [],
            "feature_spec": pipeline.feature_spec,
            "feature_importance": {
                "xgboost": xgb_importance,
                "rulekit": {},
            },
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
    pipeline.scaler = None
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
    pipeline._training_columns = list(X.columns)
    pipeline._scaler_mean = []
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
    """Predict for all objects in testing_X using the ensemble model.

    Optionally compares with testing_Y_df for evaluation metrics.
    """
    def _cb(pct: int, msg: str) -> None:
        if progress_cb:
            try:
                progress_cb(pct, msg)
            except Exception:
                pass

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
        _cb(25, "XGBoost klasifikační predikce...")
        if not hasattr(pipeline, "xgb_model") or pipeline.xgb_model is None:
            raise Exception("Classification model is missing. Train Phase 3 again.")

        y_pred_encoded = pipeline.xgb_model.predict(X_test)
        y_pred_proba = None
        if hasattr(pipeline.xgb_model, "predict_proba"):
            y_pred_proba = pipeline.xgb_model.predict_proba(X_test)

        label_classes = getattr(pipeline, "_label_classes", []) or []

        actual_values: dict[str, str] = {}
        if testing_Y_df is not None:
            join_col = testing_Y_df.columns[0]
            testing_Y_df = testing_Y_df.rename(columns={join_col: "media_name"})
            testing_Y_df["media_name"] = testing_Y_df["media_name"].astype(str).str.strip()
            target_col = _resolve_eval_target_column(testing_Y_df, pipeline.target_variable)
            if target_col:
                for _, row in testing_Y_df.iterrows():
                    actual_values[str(row["media_name"]).strip()] = str(row[target_col])

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
            cls_idx = int(y_pred_encoded[row_num])
            pred_label = label_classes[cls_idx] if 0 <= cls_idx < len(label_classes) else str(cls_idx)
            confidence = None
            if y_pred_proba is not None:
                confidence = float(np.max(y_pred_proba[row_num]))

            item = {
                "media_name": media_name,
                "predicted_label": pred_label,
                "confidence": round(confidence, 4) if confidence is not None else None,
                "rule_applied": "XGBoost classification (no single rule match)",
                "extracted_features": {
                    k: row[k] for k in pipeline.feature_spec if k in row
                },
            }

            if media_name in actual_values:
                item["actual_label"] = actual_values[media_name]
                pred_list.append(pred_label)
                actual_list.append(actual_values[media_name])

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

        return {"predictions": results, "metrics": metrics}

    # ---------------- Regression branch ----------------
    _cb(25, "RuleKit predikce...")
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
                actual_values[str(row["media_name"]).strip()] = float(row[target_col])

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
        rule = _find_covering_rule(row, pipeline.rules) if pipeline.rules else "Default rule"

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

    return {"predictions": results, "metrics": metrics}
