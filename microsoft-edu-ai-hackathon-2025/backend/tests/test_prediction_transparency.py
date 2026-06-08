"""Tests for transparent classification prediction details."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pipeline import ml_training


class FakeRuleKitClassifier:
    def __init__(self, coverage_matrix=None):
        self._coverage_matrix = coverage_matrix

    def get_coverage_matrix(self, _X):
        return self._coverage_matrix


class FakeModel:
    def __init__(self, name):
        self.name = name


def _pipeline(*, ensemble=True, coverage_matrix=None):
    return SimpleNamespace(
        is_trained=True,
        target_mode="classification",
        testing_X=pd.DataFrame([
            {"media_name": "row_1", "f1": 0.1},
            {"media_name": "row_2", "f1": 0.9},
        ]),
        feature_spec={"f1": [0, 1]},
        _training_columns=["f1"],
        _scaler_mean=[0.0],
        _label_classes=["A", "B"],
        _positive_label=None,
        model=FakeRuleKitClassifier(coverage_matrix=coverage_matrix),
        xgb_model={"rf": FakeModel("rf"), "gbt": FakeModel("gbt")} if ensemble else None,
        rules=[
            "IF f1 >= 0 THEN A",
            "IF f1 >= 0 THEN B",
            "IF f1 >= 0 THEN label = {A}",
            "IF f1 >= 0 THEN label = {B}",
        ],
        target_variable="label",
        predictions=None,
        prediction_metrics=None,
        save_state=lambda: None,
    )


def test_prediction_transparency_marks_ensemble_override_and_top_rules(monkeypatch):
    def fake_rulekit_predict(*_args, **_kwargs):
        return np.array(["A", "A"], dtype=object), np.array([[0.9, 0.1], [0.9, 0.1]])

    def fake_model_proba(model, _X, _labels):
        if model.name == "rf":
            return np.array([[0.8, 0.2], [0.1, 0.9]])
        return np.array([[0.7, 0.3], [0.1, 0.9]])

    monkeypatch.setattr(ml_training, "_rulekit_classification_predict", fake_rulekit_predict)
    monkeypatch.setattr(ml_training, "_model_predict_proba", fake_model_proba)

    pipeline = _pipeline(
        coverage_matrix=np.array([
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ])
    )

    result = ml_training.predict_batch(pipeline)
    first, second = result["predictions"]

    assert first["predicted_label"] == "A"
    assert first["rulekit_prediction"] == "A"
    assert first["rf_prediction"] == "A"
    assert first["gbt_prediction"] == "A"
    assert first["ensemble_override"] is False

    assert second["predicted_label"] == "B"
    assert second["rulekit_prediction"] == "A"
    assert second["rf_prediction"] == "B"
    assert second["gbt_prediction"] == "B"
    assert second["ensemble_override"] is True
    assert second["confidence_breakdown"]["final"] == {"label": "B", "confidence": 0.6333}
    assert second["confidence_breakdown"]["rulekit"] == {"label": "A", "confidence": 0.9}
    assert second["confidence_breakdown"]["rf"] == {"label": "B", "confidence": 0.9}
    assert second["confidence_breakdown"]["gbt"] == {"label": "B", "confidence": 0.9}

    assert first["rule_applied"] == "IF f1 >= 0 THEN A"
    assert first["top_rules"] == [
        "IF f1 >= 0 THEN A",
        "IF f1 >= 0 THEN label = {A}",
    ]
    assert second["rule_applied"] == "IF f1 >= 0 THEN A"
    assert second["top_rules"] == [
        "IF f1 >= 0 THEN A",
        "IF f1 >= 0 THEN label = {A}",
    ]


def test_prediction_transparency_degrades_without_ensemble(monkeypatch):
    def fake_rulekit_predict(*_args, **_kwargs):
        return np.array(["A", "B"], dtype=object), np.array([[0.8, 0.2], [0.3, 0.7]])

    monkeypatch.setattr(ml_training, "_rulekit_classification_predict", fake_rulekit_predict)

    result = ml_training.predict_batch(_pipeline(ensemble=False))
    first, second = result["predictions"]

    assert first["ensemble_override"] is False
    assert second["ensemble_override"] is False
    assert first["rf_prediction"] is None
    assert first["gbt_prediction"] is None
    assert first["confidence_breakdown"]["rf"] == {"label": None, "confidence": None}
    assert first["confidence_breakdown"]["gbt"] == {"label": None, "confidence": None}
    assert first["top_rules"] == [
        "IF f1 >= 0 THEN A",
        "IF f1 >= 0 THEN label = {A}",
    ]
    assert second["top_rules"] == [
        "IF f1 >= 0 THEN B",
        "IF f1 >= 0 THEN label = {B}",
    ]


def test_prediction_transparency_does_not_show_wrong_label_rule(monkeypatch):
    def fake_rulekit_predict(*_args, **_kwargs):
        return np.array(["A", "A"], dtype=object), np.array([[0.8, 0.2], [0.8, 0.2]])

    monkeypatch.setattr(ml_training, "_rulekit_classification_predict", fake_rulekit_predict)

    pipeline = _pipeline(
        ensemble=False,
        coverage_matrix=np.array([
            [0, 1, 0, 1],
            [0, 1, 0, 1],
        ])
    )

    result = ml_training.predict_batch(pipeline)
    first = result["predictions"][0]

    assert first["rulekit_prediction"] == "A"
    assert first["top_rules"] == []
    assert first["rule_applied"] == "RuleKit predicted A (no matching rule for this label)"
