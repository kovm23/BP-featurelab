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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

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

        # Scale
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val), columns=X_val.columns, index=X_val.index
        )

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
        xgb.fit(X_train_scaled, y_train)
        xgb_pred = xgb.predict(X_val_scaled)

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

    def _cb(pct: int, msg: str) -> None:
        if progress_cb:
            try:
                progress_cb(pct, msg)
            except Exception:
                pass

    # --- StandardScaler ---
    _cb(15, "Normalizace příznaků (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # --- RuleKit ---
    _cb(25, "Trénuji RuleKit (indukce pravidel)...")
    logger.info("Training RuleKit model...")
    rulekit_model = RuleRegressor()
    rulekit_model.fit(X, y)  # RuleKit works on unscaled data

    rules = (
        [str(rule) for rule in rulekit_model.model.rules]
        if hasattr(rulekit_model, 'model')
        else []
    )
    rulekit_pred = rulekit_model.predict(X)
    rulekit_mse = round(float(mean_squared_error(y, rulekit_pred)), 6)

    # --- XGBoost ---
    _cb(60, "Trénuji XGBoost...")
    logger.info("Training XGBoost model...")
    xgb_model = XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=42, verbosity=0,
    )
    xgb_model.fit(X_scaled, y)
    xgb_pred = xgb_model.predict(X_scaled)
    xgb_mse = round(float(mean_squared_error(y, xgb_pred)), 6)

    # --- Ensemble ---
    _cb(75, "Ensemble predikce...")
    ensemble_pred = RULEKIT_WEIGHT * rulekit_pred + XGB_WEIGHT * xgb_pred
    ensemble_mse = round(float(mean_squared_error(y, ensemble_pred)), 6)

    logger.info(
        "Training MSE — RuleKit: %.6f, XGBoost: %.6f, Ensemble: %.6f",
        rulekit_mse, xgb_mse, ensemble_mse,
    )

    # --- Cross-Validation ---
    _cb(80, "K-fold křížová validace...")
    logger.info("Running %d-fold cross-validation...", min(5, len(X)))
    cv_results = _run_cross_validation(X, y)
    logger.info("CV results: %s", cv_results)

    # --- Feature Importance ---
    xgb_importance = {}
    if hasattr(xgb_model, 'feature_importances_'):
        for fname, imp in zip(X.columns, xgb_model.feature_importances_):
            if imp > 0.001:
                xgb_importance[fname] = round(float(imp), 4)
        xgb_importance = dict(sorted(xgb_importance.items(), key=lambda x: -x[1]))

    rulekit_importance = _count_rule_features(rules, list(X.columns))

    # --- Store in pipeline ---
    pipeline.model = rulekit_model
    pipeline.xgb_model = xgb_model
    pipeline.scaler = scaler
    pipeline.rules = rules
    pipeline.mse = ensemble_mse
    pipeline.rulekit_mse = rulekit_mse
    pipeline.xgb_mse = xgb_mse
    pipeline.cv_mse = cv_results.get("cv_mse")
    pipeline.cv_std = cv_results.get("cv_std")
    pipeline.cv_mae = cv_results.get("cv_mae")
    pipeline.feature_importance = {"xgboost": xgb_importance, "rulekit": rulekit_importance}
    pipeline.is_trained = True
    pipeline.target_variable = target_column
    pipeline._training_columns = list(X.columns)
    pipeline._scaler_mean = scaler.mean_.tolist()
    pipeline._scaler_scale = scaler.scale_.tolist()

    pipeline.save_state()
    _cb(98, "Ukládám model...")

    # --- Overfitting warning ---
    warnings = []
    if cv_results.get("cv_mse") is not None:
        if cv_results["cv_mse"] > 2 * ensemble_mse:
            warnings.append(
                f"Possible overfitting: CV MSE ({cv_results['cv_mse']:.6f}) "
                f"is much higher than training MSE ({ensemble_mse:.6f}). "
                f"Consider adding more training data."
            )

    return {
        "status": "success",
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

    # RuleKit prediction
    _cb(25, "RuleKit predikce...")
    rulekit_pred = pipeline.model.predict(X_test)

    # XGBoost prediction (with scaling)
    if hasattr(pipeline, 'scaler') and pipeline.scaler is not None:
        X_test_scaled = pd.DataFrame(
            pipeline.scaler.transform(X_test),
            columns=X_test.columns, index=X_test.index,
        )
    elif hasattr(pipeline, '_scaler_mean') and pipeline._scaler_mean:
        # Reconstruct scaler from saved params
        scaler = StandardScaler()
        scaler.mean_ = np.array(pipeline._scaler_mean)
        scaler.scale_ = np.array(pipeline._scaler_scale)
        scaler.n_features_in_ = len(pipeline._scaler_mean)
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns, index=X_test.index,
        )
    else:
        X_test_scaled = X_test

    if hasattr(pipeline, 'xgb_model') and pipeline.xgb_model is not None:
        _cb(40, "XGBoost predikce...")
        xgb_pred = pipeline.xgb_model.predict(X_test_scaled)
        _cb(55, "Ensemble kombinace (RuleKit + XGBoost)...")
        predictions = RULEKIT_WEIGHT * rulekit_pred + XGB_WEIGHT * xgb_pred
    else:
        # Fallback: RuleKit only (legacy models)
        predictions = rulekit_pred

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
    total_items = len(pipeline.testing_X)
    _cb(60, f"Sestavuji výsledky (0/{total_items})...")
    report_every = max(1, total_items // 10)

    for row_num, (i, row) in enumerate(pipeline.testing_X.iterrows()):
        if row_num % report_every == 0:
            pct = 60 + int(((row_num + 1) / total_items) * 28)
            _cb(pct, f"Sestavuji výsledky ({row_num + 1}/{total_items})...")
        pred_score = float(predictions[i])
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
            "mse": round(mse, 6),
            "mae": round(mae, 6),
            "correlation": round(correlation, 4) if correlation is not None else None,
            "matched_count": len(pred_list),
            "total_count": len(results),
        }

    return {"predictions": results, "metrics": metrics}
