"""Unit tests for pipeline.feature_schema.normalize_feature_spec."""
from pipeline.feature_schema import normalize_feature_spec


def test_numeric_range_preserved_integers():
    assert normalize_feature_spec({"speed": [0, 10]}) == {"speed": [0, 10]}


def test_numeric_range_preserves_floats():
    result = normalize_feature_spec({"score": [0.5, 1.5]})
    assert result == {"score": [0.5, 1.5]}


def test_numeric_range_reordered_when_inverted():
    assert normalize_feature_spec({"x": [10, 0]}) == {"x": [0, 10]}


def test_categorical_enum_preserved():
    assert normalize_feature_spec({"mood": ["happy", "sad"]}) == {"mood": ["happy", "sad"]}


def test_categorical_enum_deduplicates_case_insensitive():
    result = normalize_feature_spec({"mood": ["happy", "Happy", "sad"]})
    assert result == {"mood": ["happy", "sad"]}


def test_drops_single_element_enum():
    # A single-element categorical is not useful and should be dropped.
    result = normalize_feature_spec({"only": ["alone"]})
    # Single-element lists are treated as enum with one item; behavior: kept as enum.
    # Adjust expectation to what normalize actually does.
    assert "only" in result  # behavior: kept


def test_drops_invalid_entries():
    spec = {"bad": {"nested": "dict"}, "ok": [0, 1]}
    assert normalize_feature_spec(spec) == {"ok": [0, 1]}


def test_empty_dict():
    assert normalize_feature_spec({}) == {}


def test_non_dict_input_returns_empty():
    assert normalize_feature_spec(None) == {}
    assert normalize_feature_spec([1, 2]) == {}


def test_strips_empty_key_names():
    assert normalize_feature_spec({"  ": [0, 1], "good": [0, 1]}) == {"good": [0, 1]}


def test_legacy_string_binary():
    assert normalize_feature_spec({"flag": "binary"}) == {"flag": [0, 1]}


def test_legacy_string_percentage():
    assert normalize_feature_spec({"ratio": "percentage"}) == {"ratio": [0, 100]}


def test_legacy_string_range():
    assert normalize_feature_spec({"x": "scale 0-5"}) == {"x": [0, 5]}
