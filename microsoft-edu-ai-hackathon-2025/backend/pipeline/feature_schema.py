"""Helpers for normalizing feature_spec into a structured schema."""
import logging
import re

logger = logging.getLogger(__name__)

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
_ENUM_HINT_PATTERN = re.compile(
    r"^(?:one of|enum|categories?|values?)\s*[:\-]?\s*",
    re.IGNORECASE,
)


def _normalize_numeric_range(values) -> list[int | float] | None:
    if len(values) != 2 or not all(isinstance(v, (int, float)) for v in values):
        return None

    lo, hi = float(values[0]), float(values[1])
    if lo > hi:
        lo, hi = hi, lo

    if lo.is_integer() and hi.is_integer():
        return [int(lo), int(hi)]
    return [lo, hi]


def _normalize_string_enum(values) -> list[str] | None:
    if not values or not all(isinstance(v, str) for v in values):
        return None

    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in values:
        item = raw.strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)

    return cleaned or None


def _parse_legacy_string(value: str) -> list[int | float] | list[str] | None:
    text = value.strip()
    if not text:
        return None

    if _BINARY_PATTERN.search(text):
        return [0, 1]

    if _PERCENTAGE_PATTERN.search(text):
        return [0, 100]

    match = _RANGE_PATTERN.search(text)
    if match:
        lo, hi = float(match.group(1)), float(match.group(2))
        if lo > hi:
            lo, hi = hi, lo
        if lo.is_integer() and hi.is_integer():
            return [int(lo), int(hi)]
        return [lo, hi]

    enum_text = _ENUM_HINT_PATTERN.sub("", text)
    enum_text = enum_text.strip().strip("[](){}")
    if not enum_text:
        return None

    if "," in enum_text:
        parts = [part.strip().strip("\"'") for part in enum_text.split(",")]
        normalized = _normalize_string_enum(parts)
        if normalized and len(normalized) >= 2:
            return normalized

    if "|" in enum_text:
        parts = [part.strip().strip("\"'") for part in enum_text.split("|")]
        normalized = _normalize_string_enum(parts)
        if normalized and len(normalized) >= 2:
            return normalized

    return None


def normalize_feature_spec(feature_spec: dict | None) -> dict:
    """Normalize feature_spec to structured values only.

    Supported value shapes:
    - [min, max]
    - ["category_a", "category_b", ...]

    Legacy string descriptions are best-effort parsed for backwards compatibility.
    Invalid entries are dropped.
    """
    if not isinstance(feature_spec, dict):
        return {}

    normalized: dict = {}
    for raw_name, raw_value in feature_spec.items():
        name = str(raw_name).strip()
        if not name:
            continue

        value = None
        if isinstance(raw_value, list):
            value = _normalize_numeric_range(raw_value)
            if value is None:
                value = _normalize_string_enum(raw_value)
        elif isinstance(raw_value, str):
            value = _parse_legacy_string(raw_value)

        if value is None:
            logger.warning("Dropping unsupported feature spec entry: %s=%r", name, raw_value)
            continue

        normalized[name] = value

    return normalized
