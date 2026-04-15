"""Unit tests for utils.target_context."""
import pandas as pd

from utils.target_context import build_labels_context, find_target_column


def _df(cols: dict) -> pd.DataFrame:
    return pd.DataFrame(cols)


def test_find_target_exact_match():
    df = _df({"media": ["a", "b"], "score": [1.0, 2.0]})
    assert find_target_column(df, "score", "regression") == "score"


def test_find_target_case_insensitive():
    df = _df({"Media": ["a"], "Score": [1.0]})
    assert find_target_column(df, "score", "regression") == "Score"


def test_find_target_normalizes_spaces_to_underscores():
    df = _df({"media": ["a"], "customer satisfaction": [1.0]})
    assert find_target_column(df, "customer_satisfaction", "regression") == "customer satisfaction"


def test_find_target_falls_back_to_last_numeric_for_regression():
    df = _df({"media": ["a"], "note": ["x"], "value": [1.0]})
    assert find_target_column(df, "unknown", "regression") == "value"


def test_find_target_returns_none_for_empty_df():
    assert find_target_column(_df({}), "x", "regression") is None


def test_find_target_returns_none_when_no_numeric_for_regression():
    df = _df({"media": ["a"], "label": ["x"]})
    assert find_target_column(df, "unknown", "regression") is None


def test_find_target_classification_returns_last_non_media():
    df = _df({"media": ["a"], "label": ["cat"]})
    assert find_target_column(df, "unknown", "classification") == "label"


def test_build_labels_context_regression_contains_stats():
    df = _df({"media": ["a", "b", "c"], "score": [1.0, 2.0, 3.0]})
    ctx = build_labels_context(df, "score", "regression")
    assert "score" in ctx
    assert "Min:" in ctx and "Max:" in ctx


def test_build_labels_context_classification_contains_class_counts():
    df = _df({"media": ["a", "b", "c"], "label": ["cat", "dog", "cat"]})
    ctx = build_labels_context(df, "label", "classification")
    assert "cat" in ctx and "dog" in ctx
    assert "classes" in ctx


def test_build_labels_context_empty_df_returns_empty_string():
    assert build_labels_context(_df({}), "x", "regression") == ""


def test_build_labels_context_none_returns_empty_string():
    assert build_labels_context(None, "x", "regression") == ""
