from __future__ import annotations
from typing import Dict, Any, List

def _parse_value(v: str) -> Any:
    s = v.strip()

    if s.lower() in {"true", "false"}:
        return s.lower() == "true"

    if s.lower() in {"none", "null"}:
        return None

    try:
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
    except Exception:
        pass

    try:
        return float(s)
    except Exception:
        pass

    return s

def parse_overrides(argv: List[str]) -> Dict[str, Any]:
    """
    Parses key=value pairs into a dict.
    Example: ["epochs=30", "lr=3e-4", "class_weight=true"] -> {...}
    """
    out: Dict[str, Any] = {}
    for item in argv:
        if "=" not in item:
            raise ValueError(f"Invalid arg '{item}'. Expected key=value.")
        k, v = item.split("=", 1)
        k = k.strip()
        if not k:
            raise ValueError(f"Invalid arg '{item}'. Empty key.")
        out[k] = _parse_value(v)
    return out
