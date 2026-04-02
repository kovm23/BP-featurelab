"""Helpers for target-column selection and prompt context generation."""
import pandas as pd


def _normalize_name(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_")


def find_target_column(
    labels_df: pd.DataFrame,
    target_variable: str,
    target_mode: str,
) -> str | None:
    """Find the most likely target column in a labels DataFrame.

    The first column is usually the media identifier, so fallback selection
    prefers the remaining columns.
    """
    if labels_df is None or labels_df.empty:
        return None

    normalized_target = _normalize_name(target_variable)
    for col in labels_df.columns:
        if _normalize_name(col) == normalized_target:
            return col

    candidate_cols = list(labels_df.columns[1:]) or list(labels_df.columns)
    if not candidate_cols:
        return None

    if target_mode == "regression":
        numeric_cols = [
            col for col in candidate_cols
            if pd.api.types.is_numeric_dtype(labels_df[col])
        ]
        if numeric_cols:
            return numeric_cols[-1]
        return None

    non_join_cols = [col for col in candidate_cols if _normalize_name(col) != "media_name"]
    return (non_join_cols or candidate_cols)[-1]


def build_labels_context(
    labels_df: pd.DataFrame | None,
    target_variable: str,
    target_mode: str,
) -> str:
    """Build prompt context for the selected target variable."""
    if labels_df is None or labels_df.empty:
        return ""

    target_col = find_target_column(labels_df, target_variable, target_mode)
    if target_col is None:
        return ""

    col_data = labels_df[target_col]

    if target_mode == "classification":
        labels = (
            col_data.astype(str)
            .str.strip()
            .replace({"": None, "nan": None, "none": None, "null": None})
            .dropna()
        )
        if labels.empty:
            return ""

        counts = labels.value_counts()
        top_counts = counts.head(10)
        counts_text = ", ".join(f"{label}={count}" for label, count in top_counts.items())
        sample_values = list(labels.head(10).values)
        return (
            f"\n\nYou also have access to the categorical target variable '{target_col}' "
            f"from the training labels:\n"
            f"- Number of classes: {int(counts.size)}\n"
            f"- Class distribution: {counts_text}\n"
            f"- Example labels: {sample_values}\n"
            f"Use this information to suggest features that separate the classes well.\n"
        )

    numeric = pd.to_numeric(col_data, errors="coerce").dropna()
    if numeric.empty:
        return ""

    return (
        f"\n\nYou also have access to the target variable '{target_col}' "
        f"from the training labels:\n"
        f"- Min: {numeric.min()}, Max: {numeric.max()}, "
        f"Mean: {numeric.mean():.4f}, Std: {numeric.std():.4f}\n"
        f"- Sample values: {list(numeric.head(10).values)}\n"
        f"Use this information to suggest features that would correlate "
        f"well with these target values.\n"
    )
