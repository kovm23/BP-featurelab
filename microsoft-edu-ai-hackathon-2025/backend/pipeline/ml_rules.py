"""RuleKit rule parsing and matching utilities."""
import re

import pandas as pd

# Compiled once at import — Czech math convention: < / [ = closed, ( / ) = open
_INTERVAL_RE = re.compile(
    r"([^=]+?)\s*=\s*([<(\[])\s*(-inf|-?\d+(?:\.\d+)?)\s*,\s*(inf|-?\d+(?:\.\d+)?)\s*([>)\]])",
    re.IGNORECASE,
)
_SET_RE = re.compile(r"([^=]+?)\s*=\s*\{([^}]+)\}")      # feat = {val}
_SIMPLE_RE = re.compile(r"(.+?)\s*(>=|<=|>|<|=)\s*(.+)")  # feat >= val (last resort)
_THEN_RE = re.compile(r"\bTHEN\b\s+(.+)$", re.IGNORECASE)
_THEN_SET_RE = re.compile(r"=\s*\{([^}]+)\}")
_THEN_ASSIGNMENT_RE = re.compile(r"=\s*(.+)$")


def _normalize_rule_label(label) -> str | None:
    """Return a comparison key for a class label from RuleKit output."""
    if label is None:
        return None
    text = str(label).strip()
    if not text:
        return None
    text = re.sub(r"\s+\[[^\]]+\]\s*$", "", text)
    text = text.strip().strip("{}\"'")
    text = re.sub(r"\s+", " ", text)
    return text.casefold() if text else None


def _extract_rule_then_label(rule_str: str) -> str | None:
    """Extract the predicted class label from a RuleKit rule's THEN clause."""
    match = _THEN_RE.search(str(rule_str))
    if not match:
        return None
    then_part = match.group(1).strip()
    set_match = _THEN_SET_RE.search(then_part)
    if set_match:
        return _normalize_rule_label(set_match.group(1))
    assignment_match = _THEN_ASSIGNMENT_RE.search(then_part)
    if assignment_match:
        then_part = assignment_match.group(1).strip()
    return _normalize_rule_label(then_part)


def _filter_rules_by_label(rules: list[str], expected_label) -> list[str]:
    """Keep only rules whose THEN label matches the expected RuleKit label."""
    expected_key = _normalize_rule_label(expected_label)
    if expected_key is None:
        return list(rules)
    return [
        rule_str for rule_str in rules
        if _extract_rule_then_label(rule_str) == expected_key
    ]


def _no_single_rule_match_label(predicted_label=None) -> str:
    """Fallback text for rows without a label-consistent displayed rule."""
    label = str(predicted_label).strip() if predicted_label is not None else ""
    if label:
        return f"RuleKit predicted {label} (no matching rule for this label)"
    return "RuleKit (no single rule match)"


def _rule_matches(row: pd.Series, rule_str: str) -> bool:
    """Return True when a RuleKit rule's IF-conditions are satisfied by row."""
    m = re.match(r"IF\s+(.+?)\s+THEN", rule_str, re.IGNORECASE)
    if not m:
        return False
    conditions = re.split(r"\s+AND\s+", m.group(1), flags=re.IGNORECASE)
    for cond in conditions:
        cond = cond.strip()

        iv = _INTERVAL_RE.match(cond)
        if iv:
            feat = iv.group(1).strip()
            l_br, lo_s, hi_s, r_br = iv.group(2), iv.group(3), iv.group(4), iv.group(5)
            if feat not in row.index:
                return False
            try:
                v = float(row[feat])
                lo = float("-inf") if lo_s.lower() == "-inf" else float(lo_s)
                hi = float("inf") if hi_s.lower() == "inf" else float(hi_s)
            except (ValueError, TypeError):
                return False
            lo_ok = (v >= lo) if l_br in ("<", "[") else (v > lo)
            hi_ok = (v <= hi) if r_br in (">", "]") else (v < hi)
            if not (lo_ok and hi_ok):
                return False
            continue

        sv = _SET_RE.match(cond)
        if sv:
            feat, val_s = sv.group(1).strip(), sv.group(2).strip()
            if feat not in row.index:
                return False
            try:
                if abs(float(row[feat]) - float(val_s)) >= 1e-9:
                    return False
            except (ValueError, TypeError):
                if str(row[feat]).strip() != val_s:
                    return False
            continue

        sm = _SIMPLE_RE.match(cond)
        if sm:
            feat, op, val_s = sm.group(1).strip(), sm.group(2), sm.group(3).strip()
            if feat not in row.index:
                return False
            try:
                v, thr = float(row[feat]), float(val_s)
            except (ValueError, TypeError):
                return False
            checks = {
                ">=": v >= thr, "<=": v <= thr,
                ">": v > thr, "<": v < thr,
                "=": abs(v - thr) < 1e-9,
            }
            if not checks.get(op, False):
                return False
            continue

        return False  # unparseable condition

    return True


def _find_covering_rules(
    row: pd.Series,
    rules: list[str],
    max_rules: int = 3,
    expected_label=None,
) -> list[str]:
    """Return up to max_rules whose IF-conditions are satisfied by row.

    Supports RuleKit's interval notation (Czech math convention):
      feat = <lo, hi)   ->  lo <= feat < hi
      feat = (lo, hi>   ->  lo < feat <= hi
      feat = <lo, inf)  ->  feat >= lo
      feat = (-inf, hi) ->  feat < hi
      feat = {val}      ->  exact match
    Also handles simple operators: feat >= val, feat <= val, etc.
    """
    matches = []
    expected_key = _normalize_rule_label(expected_label)
    for rule_str in rules:
        if expected_key is not None and _extract_rule_then_label(rule_str) != expected_key:
            continue
        if _rule_matches(row, rule_str):
            matches.append(rule_str)
            if len(matches) >= max_rules:
                break
    return matches


def _find_covering_rule(row: pd.Series, rules: list[str], expected_label=None) -> str:
    """Return the first rule whose IF-conditions are satisfied by row, or a fallback label."""
    matches = _find_covering_rules(row, rules, max_rules=1, expected_label=expected_label)
    if matches:
        return matches[0]
    return _no_single_rule_match_label(expected_label)


def _count_rule_features(rules: list[str], feature_names: list[str]) -> dict:
    """Return a normalised frequency dict of features appearing in rules."""
    counts = {
        feat: sum(1 for rule in rules if (feat + " ") in rule)
        for feat in feature_names
    }
    counts = {k: v for k, v in counts.items() if v > 0}
    total = sum(counts.values()) or 1
    return {k: round(v / total, 4) for k, v in sorted(counts.items(), key=lambda x: -x[1])}


def _extract_rules(model) -> list[str]:
    """Return rule strings from a trained RuleKit model, or [] if unavailable."""
    if hasattr(model, "model") and hasattr(model.model, "rules"):
        return [str(r) for r in model.model.rules]
    return []
