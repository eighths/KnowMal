# analyzer/analyzer.py
from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any, Dict

_dispatcher_func = None
try:
    from .dispatcher import dispatch as _dispatcher_func  # 선호: dispatch(path) -> features
except Exception:
    try:
        from .dispatcher import parse_with_dispatcher as _dispatcher_func  # 대안
    except Exception:
        try:
            from .dispatcher import run as _dispatcher_func  # 최후 대안
        except Exception:
            _dispatcher_func = None

from .utils.feature_utils import (
    dedup_sort_strings, clamp_entropy_fields, filter_nonstandard_domains,
)
from .utils.feature_utils import domain_from_url

def _postprocess(features: Dict[str, Any]) -> Dict[str, Any]:
    f = dict(features or {})

    if isinstance(f.get("strings"), dict):
        for key in ("urls", "ips", "filepaths", "registry_keys"):
            if isinstance(f["strings"].get(key), list):
                f["strings"][key] = dedup_sort_strings(f["strings"][key])

    if isinstance(f.get("network_indicators"), dict):
        for key in ("urls", "domains", "user_agents"):
            if isinstance(f["network_indicators"].get(key), list):
                f["network_indicators"][key] = dedup_sort_strings(
                    f["network_indicators"][key]
                )

        cur = f["network_indicators"].get("domains")
        if not cur:
            urls = f.get("strings", {}).get("urls", []) or []
            doms = sorted({d for d in (domain_from_url(u) for u in urls) if d})
            f["network_indicators"]["domains"] = doms

        f["network_indicators"]["domains"] = filter_nonstandard_domains(
            f["network_indicators"].get("domains", [])
        )

    if isinstance(f.get("obfuscation"), dict):
        ent = f["obfuscation"].get("entropy")
        if isinstance(ent, dict):
            f["obfuscation"]["entropy"] = clamp_entropy_fields(ent)

    try:
        macros = f.get("macros", {}) or {}
        apis = f.get("apis", {}) or {}
        win_sum = sum(int(x.get("count", 0)) for x in (apis.get("winapi_calls") or []) if isinstance(x, dict))
        com_cnt = len(apis.get("com_progids") or [])
        calc = int(win_sum + com_cnt)
        if "suspicious_api_calls_count" not in macros or int(macros.get("suspicious_api_calls_count") or 0) < calc:
            macros["suspicious_api_calls_count"] = calc
            f["macros"] = macros
    except Exception:
        pass

    return f
