"""Feature value validation and clamping based on feature_spec descriptions.

Parses expected ranges/types from feature descriptions and ensures
extracted values conform. Logs statistics about clamped values.
"""
import logging
import re

logger = logging.getLogger(__name__)

# Pre-compiled patterns for range extraction
_RANGE_PATTERN = re.compile(
    r"(?:score|range|scale|hodnota)?\s*(\d+(?:\.\d+)?)\s*[-–—to]+\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_BINARY_PATTERN = re.compile(
    r"\b(?:binary|bool|boolean)\b|(?:0\s+or\s+1)|(?:0/1)",
    re.IGNORECASE,
)
_PERCENTAGE_PATTERN = re.compile(
    r"\b(?:percent|percentage|%)\b",
    re.IGNORECASE,
)


def parse_feature_range(description: str) -> dict:
    """Extract expected type and range from a feature description string.

    Returns a dict with keys:
        type: "numeric_range" | "binary" | "percentage" | "unknown"
        min: float (if applicable)
        max: float (if applicable)
    """
    if _BINARY_PATTERN.search(description):
        return {"type": "binary", "min": 0, "max": 1}

    if _PERCENTAGE_PATTERN.search(description):
        return {"type": "percentage", "min": 0, "max": 100}

    m = _RANGE_PATTERN.search(description)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        return {"type": "numeric_range", "min": lo, "max": hi}

    return {"type": "unknown"}


def validate_and_clamp(value, feature_name: str, feature_description: str) -> tuple:
    """Validate and clamp a single feature value.

    Returns (clamped_value, was_clamped: bool).
    """
    if value is None:
        return None, False

    # New schema-first path: [min, max] numeric range
    if isinstance(feature_description, list):
        if (
            len(feature_description) == 2
            and all(isinstance(v, (int, float)) for v in feature_description)
        ):
            lo, hi = float(feature_description[0]), float(feature_description[1])
            if lo > hi:
                lo, hi = hi, lo
            try:
                numeric_val = float(value)
            except (ValueError, TypeError):
                return value, False

            clamped = max(lo, min(hi, numeric_val))
            # If schema bounds are integers, return an integer value.
            if float(feature_description[0]).is_integer() and float(feature_description[1]).is_integer():
                clamped = int(round(clamped))
                original = int(round(numeric_val)) if isinstance(value, (int, float)) else value
                return clamped, clamped != original

            return clamped, clamped != numeric_val

        # Categorical enum path: ["cat_a", "cat_b", ...]
        if feature_description and all(isinstance(v, str) for v in feature_description):
            allowed = {v.strip().lower(): v for v in feature_description}
            val_norm = str(value).strip().lower()
            if val_norm in allowed:
                canonical = allowed[val_norm]
                return canonical, canonical != value
            return value, False

    spec = parse_feature_range(feature_description)

    if spec["type"] == "unknown":
        return value, False

    # Try to convert to numeric
    try:
        numeric_val = float(value)
    except (ValueError, TypeError):
        # Non-numeric value for a numeric feature — can't clamp
        return value, False

    was_clamped = False

    if spec["type"] == "binary":
        clamped = round(max(0, min(1, numeric_val)))
        was_clamped = clamped != numeric_val
        return clamped, was_clamped

    lo, hi = spec["min"], spec["max"]
    clamped = max(lo, min(hi, numeric_val))
    was_clamped = clamped != numeric_val
    return clamped, was_clamped


def validate_row(attrs: dict, feature_spec: dict) -> tuple[dict, dict]:
    """Validate and clamp all features in a row.

    Args:
        attrs: extracted feature values {name: value}
        feature_spec: feature definitions {name: description}

    Returns:
        (validated_attrs, stats) where stats = {
            "clamped_count": int,
            "clamped_features": list[str],
        }
    """
    validated = {}
    clamped_features = []

    for feat_name, description in feature_spec.items():
        val = attrs.get(feat_name)
        clamped_val, was_clamped = validate_and_clamp(val, feat_name, description)
        validated[feat_name] = clamped_val

        if was_clamped:
            clamped_features.append(feat_name)
            logger.info(
                "Clamped %s: %s -> %s (spec: %s)",
                feat_name, val, clamped_val, description,
            )

    return validated, {
        "clamped_count": len(clamped_features),
        "clamped_features": clamped_features,
    }
