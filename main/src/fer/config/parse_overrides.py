from __future__ import annotations

from typing import Any, Dict, List
import ast


def _parse_value(v: str) -> Any:
    """
    More capable parser for CLI overrides.

    Supports:
      - bool: true/false
      - none/null
      - int / float (incl. scientific notation)
      - quoted strings: "adamw", 'coatnet'
      - lists/tuples/dicts via python-literal:
          class_names='["anger","disgust"]'
          betas="(0.9,0.999)"
          aug='{"mixup_alpha":0.2}'
      - fallback: raw string
    """
    s = v.strip()
    if s == "":
        return ""

    low = s.lower()

    # bool
    if low in {"true", "false"}:
        return low == "true"

    # none
    if low in {"none", "null"}:
        return None

    # If user passed a quoted value, keep it as string (without quotes)
    if (len(s) >= 2) and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1]

    # Try python literal eval for containers and numbers
    # Examples: [1,2], {"a":1}, (0.9,0.999), 3e-4
    try:
        val = ast.literal_eval(s)
        # ast.literal_eval can return strings too; that's fine.
        return val
    except Exception:
        pass

    # As a last resort, try numeric parsing (covers cases like 3e-4 without quotes)
    # (literal_eval already handles many, but keep this as extra robust)
    try:
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
    except Exception:
        pass

    try:
        return float(s)
    except Exception:
        pass

    # fallback: raw string
    return s


def parse_overrides(argv: List[str]) -> Dict[str, Any]:
    """
    Parses CLI overrides in key=value format.

    Example:
      ["epochs=30", "lr=3e-4", "class_weight=true", "class_names='[\"a\",\"b\"]'"]
    -> {"epochs": 30, "lr": 0.0003, "class_weight": True, "class_names": ["a","b"]}
    """
    out: Dict[str, Any] = {}

    for item in argv:
        item = item.strip()
        if not item:
            continue

        if "=" not in item:
            raise ValueError(
                f"Invalid arg '{item}'. Expected key=value.\n"
                f"Examples: epochs=30 lr=3e-4 class_weight=true"
            )

        k, v = item.split("=", 1)
        k = k.strip()
        if not k:
            raise ValueError(f"Invalid arg '{item}'. Empty key.")

        out[k] = _parse_value(v)

    return out
