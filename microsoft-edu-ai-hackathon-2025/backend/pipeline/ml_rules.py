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


def _find_covering_rule(row: pd.Series, rules: list[str]) -> str:
    """Return the first rule whose IF-conditions are satisfied by *row*, or a fallback label.

    Supports RuleKit's interval notation (Czech math convention):
      feat = <lo, hi)   →  lo ≤ feat < hi
      feat = (lo, hi>   →  lo < feat ≤ hi
      feat = <lo, inf)  →  feat ≥ lo
      feat = (-inf, hi) →  feat < hi
      feat = {val}      →  exact match
    Also handles simple operators: feat >= val, feat <= val, etc.
    """
    for rule_str in rules:
        m = re.match(r"IF\s+(.+?)\s+THEN", rule_str, re.IGNORECASE)
        if not m:
            continue
        conditions = re.split(r"\s+AND\s+", m.group(1), flags=re.IGNORECASE)
        all_met = True
        for cond in conditions:
            cond = cond.strip()

            iv = _INTERVAL_RE.match(cond)
            if iv:
                feat = iv.group(1).strip()
                l_br, lo_s, hi_s, r_br = iv.group(2), iv.group(3), iv.group(4), iv.group(5)
                if feat not in row.index:
                    all_met = False
                    break
                try:
                    v = float(row[feat])
                    lo = float("-inf") if lo_s.lower() == "-inf" else float(lo_s)
                    hi = float("inf") if hi_s.lower() == "inf" else float(hi_s)
                except (ValueError, TypeError):
                    all_met = False
                    break
                lo_ok = (v >= lo) if l_br in ("<", "[") else (v > lo)
                hi_ok = (v <= hi) if r_br in (">", "]") else (v < hi)
                if not (lo_ok and hi_ok):
                    all_met = False
                continue

            sv = _SET_RE.match(cond)
            if sv:
                feat, val_s = sv.group(1).strip(), sv.group(2).strip()
                if feat not in row.index:
                    all_met = False
                    break
                try:
                    if abs(float(row[feat]) - float(val_s)) >= 1e-9:
                        all_met = False
                except (ValueError, TypeError):
                    if str(row[feat]).strip() != val_s:
                        all_met = False
                continue

            sm = _SIMPLE_RE.match(cond)
            if sm:
                feat, op, val_s = sm.group(1).strip(), sm.group(2), sm.group(3).strip()
                if feat not in row.index:
                    all_met = False
                    break
                try:
                    v, thr = float(row[feat]), float(val_s)
                except (ValueError, TypeError):
                    all_met = False
                    break
                checks = {
                    ">=": v >= thr, "<=": v <= thr,
                    ">": v > thr, "<": v < thr,
                    "=": abs(v - thr) < 1e-9,
                }
                if not checks.get(op, False):
                    all_met = False
                continue

            all_met = False
            break  # unparseable condition

        if all_met:
            return rule_str
    return "RuleKit (no single rule match)"


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
