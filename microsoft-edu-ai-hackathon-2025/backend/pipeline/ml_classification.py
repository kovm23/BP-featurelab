"""Classification helpers: label resolution, probability alignment, CV, metrics."""
import logging
import os
from collections import Counter

import numpy as np
import pandas as pd
from rulekit.classification import RuleClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from pipeline.ml_preprocessing import _apply_median_imputer, _fit_median_imputer, _oversample_minority
from utils.csv_utils import normalize_media_name

logger = logging.getLogger(__name__)

CLASSIFICATION_POSITIVE_LABEL = os.getenv("CLASSIFICATION_POSITIVE_LABEL", "").strip()
CV_MAX_FOLDS = max(2, int(os.getenv("CV_MAX_FOLDS", "3")))


# ---------------------------------------------------------------------------
# Label utilities
# ---------------------------------------------------------------------------

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
    if unique_count > 20 and unique_ratio > 0.3 and (singleton_ratio > 0.5 or tiny_class_ratio > 0.8):
        raise Exception(
            "Classification mode expects a small, repeated set of class labels, but the selected "
            f"target column looks identifier-like or too sparse ({unique_count} unique values across "
            f"{len(y)} rows, singleton class ratio {singleton_ratio:.2f}). "
            "Choose a categorical label column with repeated classes."
        )
    return y


def _resolve_eval_target_column(testing_Y_df: pd.DataFrame, target_variable: str) -> str | None:
    """Find the evaluation target column or raise if missing."""
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


# ---------------------------------------------------------------------------
# Probability matrix helpers
# ---------------------------------------------------------------------------

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
    label_classes: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if rulekit_model is None:
        raise Exception("Classification RuleKit model is missing.")

    pred_labels = np.asarray(rulekit_model.predict(X), dtype=object)
    pred_proba = None
    if hasattr(rulekit_model, "predict_proba"):
        raw_proba = rulekit_model.predict_proba(X)
        source_labels = [
            str(label)
            for label in getattr(
                rulekit_model, "label_unique_values", pd.Index(pd.unique(pred_labels)).tolist()
            )
        ]
        target_labels = (
            label_classes
            or source_labels
            or [str(label) for label in pd.Index(pd.unique(pred_labels)).tolist()]
        )
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


# ---------------------------------------------------------------------------
# Ensemble (RF + GBT) helpers
# ---------------------------------------------------------------------------

def _train_ensemble_classifier(X_fit: pd.DataFrame, y_fit: pd.Series) -> tuple:
    """Train RF + GBT soft-voting ensemble alongside RuleKit."""
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=1,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    gbt = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42,
    )
    rf.fit(X_fit, y_fit)
    gbt.fit(X_fit, y_fit)
    return rf, gbt


def _ensemble_predict_proba(
    rf, gbt, X: pd.DataFrame, label_classes: list[str]
) -> np.ndarray:
    """Soft-vote probabilities from RF+GBT, aligned to label_classes order."""
    rf_classes = [str(c) for c in rf.classes_]
    gbt_classes = [str(c) for c in gbt.classes_]
    rf_p = _align_probability_columns(rf.predict_proba(X), rf_classes, label_classes)
    gbt_p = _align_probability_columns(gbt.predict_proba(X), gbt_classes, label_classes)
    return (rf_p + gbt_p) / 2.0


def _combined_predict_proba(
    rulekit_model, rf, gbt, X: pd.DataFrame, label_classes: list[str],
    rulekit_weight: float = 1 / 3,
) -> np.ndarray:
    """3-way soft vote: RuleKit + RF + GBT.

    rulekit_weight controls RuleKit's share (default 1/3 = equal weight).
    The remaining weight is split equally between RF and GBT.
    """
    ens_weight = 1.0 - rulekit_weight
    ens_proba = _ensemble_predict_proba(rf, gbt, X, label_classes)
    rk_proba_raw = rulekit_model.predict_proba(X)
    rk_source_labels = [
        str(label)
        for label in getattr(
            rulekit_model, "label_unique_values", pd.Index(pd.unique(rulekit_model.predict(X))).tolist()
        )
    ]
    rk_proba = _align_probability_columns(
        _ensure_probability_matrix(rk_proba_raw, len(rk_source_labels)),
        rk_source_labels, label_classes,
    )
    return rulekit_weight * rk_proba + ens_weight * ens_proba


def _rf_feature_importance(rf, feature_columns: list[str], original_spec: dict) -> dict:
    """Aggregate RF Gini importances back to original (pre-OHE) feature names."""
    col_imp = dict(zip(feature_columns, rf.feature_importances_))
    aggregated: dict[str, float] = {}
    for feat_name in original_spec:
        related = [c for c in feature_columns if c == feat_name or c.startswith(feat_name + "_")]
        aggregated[feat_name] = sum(col_imp.get(c, 0.0) for c in related)
    # Computed video features are not in feature_spec — pass through directly
    for col in feature_columns:
        if col.startswith("vid_") and col not in aggregated:
            aggregated[col] = col_imp.get(col, 0.0)
    total = sum(aggregated.values())
    if total > 0:
        return {k: round(v / total, 6) for k, v in aggregated.items()}
    return aggregated


def _run_cross_validation_ensemble(
    X: pd.DataFrame, y: pd.Series, label_classes: list[str],
    n_splits: int = CV_MAX_FOLDS,
    rulekit_weight: float = 1 / 3,
) -> dict:
    """Stratified K-fold CV for the 3-way combined model (RuleKit + RF + GBT).

    Trains all three models per fold so that blended CV estimates are accurate.
    """
    class_counts = y.value_counts()
    positive_counts = class_counts[class_counts > 0]
    max_splits = min(n_splits, len(X), int(positive_counts.min())) if len(positive_counts) else 0
    if max_splits < 2:
        return {
            "cv_accuracy": None, "cv_balanced_accuracy": None,
            "cv_f1_macro": None, "cv_precision_macro": None,
            "cv_recall_macro": None, "cv_mcc": None,
            "n_folds": max_splits,
            "note": "Too few samples per class for stratified CV",
        }

    kf = StratifiedKFold(n_splits=max_splits, shuffle=True, random_state=42)
    fold_acc, fold_bal_acc, fold_f1, fold_prec, fold_rec, fold_mcc = [], [], [], [], [], []

    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        medians = _fit_median_imputer(X_train)
        X_train = _apply_median_imputer(X_train, medians)
        X_val = _apply_median_imputer(X_val, medians)
        X_train_s, y_train_s = _oversample_minority(X_train, y_train)

        rk_fold = RuleClassifier()
        rk_fold.fit(X_train_s, y_train_s)
        rf_fold, gbt_fold = _train_ensemble_classifier(X_train_s, y_train_s)

        try:
            proba = _combined_predict_proba(
                rk_fold, rf_fold, gbt_fold, X_val, label_classes, rulekit_weight=rulekit_weight
            )
        except Exception:
            proba = _ensemble_predict_proba(rf_fold, gbt_fold, X_val, label_classes)

        y_pred = _proba_to_labels(proba, label_classes)

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


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def _run_cross_validation_classification(
    X: pd.DataFrame, y_labels: pd.Series, n_splits: int = CV_MAX_FOLDS,
    positive_threshold: float = 0.45,
) -> dict:
    """Run stratified CV for classification with RuleKit only."""
    class_counts = y_labels.value_counts()
    positive_counts = class_counts[class_counts > 0]
    max_splits = min(n_splits, len(X), int(positive_counts.min())) if len(positive_counts) else 0
    if max_splits < 2:
        return {
            "cv_accuracy": None, "cv_balanced_accuracy": None,
            "cv_f1_macro": None, "cv_precision_macro": None,
            "cv_recall_macro": None, "cv_mcc": None,
            "n_folds": max_splits,
            "note": "Too few samples per class for stratified CV",
        }

    kf = StratifiedKFold(n_splits=max_splits, shuffle=True, random_state=42)
    fold_acc, fold_bal_acc, fold_f1, fold_prec, fold_rec, fold_mcc = [], [], [], [], [], []
    cv_label_classes = [str(c) for c in pd.Index(pd.unique(y_labels)).tolist()]
    positive_label = _resolve_positive_label(cv_label_classes, y_reference=y_labels)

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
            rk_clf, X_val,
            positive_label=positive_label,
            positive_threshold=positive_threshold,
            label_classes=cv_label_classes,
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


# ---------------------------------------------------------------------------
# Prediction result helpers
# ---------------------------------------------------------------------------

def _load_actual_values(
    testing_Y_df: pd.DataFrame | None,
    target_variable: str,
    *,
    numeric: bool,
) -> dict:
    """Build a {normalised_media_name → label} dict from the ground-truth CSV."""
    out: dict = {}
    if testing_Y_df is None:
        return out
    df = testing_Y_df.copy()
    df = df.rename(columns={df.columns[0]: "media_name"})
    df["media_name"] = df["media_name"].astype(str).str.strip()
    target_col = _resolve_eval_target_column(df, target_variable)
    if target_col:
        for _, row in df.iterrows():
            key = normalize_media_name(row["media_name"])
            out[key] = float(row[target_col]) if numeric else str(row[target_col])
    return out


def _compute_classification_metrics(
    actual_list: list[str],
    pred_list: list[str],
    results: list[dict],
) -> dict | None:
    """Compute classification evaluation metrics from matched prediction pairs."""
    if not (pred_list and actual_list):
        return None
    labels = sorted(set(actual_list) | set(pred_list))
    cm = confusion_matrix(actual_list, pred_list, labels=labels)
    class_prec, class_rec, class_f1, class_support = precision_recall_fscore_support(
        actual_list, pred_list, labels=labels, zero_division=0,
    )
    confidence_values = [
        float(i["confidence"]) for i in results if i.get("confidence") is not None
    ]
    matched = [
        i for i in results
        if i.get("actual_label") is not None and i.get("confidence") is not None
    ]
    correct_conf = [float(i["confidence"]) for i in matched if i.get("predicted_label") == i.get("actual_label")]
    incorrect_conf = [float(i["confidence"]) for i in matched if i.get("predicted_label") != i.get("actual_label")]
    label_counts = Counter(actual_list)
    majority_label = label_counts.most_common(1)[0][0]
    baseline_accuracy = round(label_counts[majority_label] / max(len(actual_list), 1), 6)
    return {
        "mode": "classification",
        "accuracy": round(float(accuracy_score(actual_list, pred_list)), 6),
        "baseline_accuracy": baseline_accuracy,
        "majority_class": majority_label,
        "balanced_accuracy": round(float(balanced_accuracy_score(actual_list, pred_list)), 6),
        "f1_macro": round(float(f1_score(actual_list, pred_list, average="macro", zero_division=0)), 6),
        "precision_macro": round(float(precision_score(actual_list, pred_list, average="macro", zero_division=0)), 6),
        "recall_macro": round(float(recall_score(actual_list, pred_list, average="macro", zero_division=0)), 6),
        "mcc": round(float(matthews_corrcoef(actual_list, pred_list)), 6),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "class_metrics": [
            {
                "label": label,
                "precision": round(float(class_prec[idx]), 6),
                "recall": round(float(class_rec[idx]), 6),
                "f1": round(float(class_f1[idx]), 6),
                "support": int(class_support[idx]),
            }
            for idx, label in enumerate(labels)
        ],
        "avg_confidence": round(float(np.mean(confidence_values)), 6) if confidence_values else None,
        "correct_confidence_avg": round(float(np.mean(correct_conf)), 6) if correct_conf else None,
        "incorrect_confidence_avg": round(float(np.mean(incorrect_conf)), 6) if incorrect_conf else None,
        "matched_count": len(pred_list),
        "total_count": len(results),
    }
