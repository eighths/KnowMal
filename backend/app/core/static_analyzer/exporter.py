from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional
from jsonschema import Draft202012Validator as Validator


def _normalize_report(report: Dict[str, Any], file_path: str) -> Dict[str, Any]:
    import hashlib
    from app.core.static_analyzer.dispatcher import choose_parser
    is_features_only = (
        isinstance(report, dict)
        and "schema_version" not in report
        and all(k in report for k in ["structure","macros","strings","apis","obfuscation","network_indicators","security_indicators"])
    )
    if is_features_only:
        try:
            with open(file_path, "rb") as f:
                sha256 = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            sha256 = ""

        try:
            kind = choose_parser(file_path)
        except Exception:
            kind = "ooxml"
        if kind == "ole":
            analyzer_name = "OLEParser"
            mime = "application/vnd.ms-office"
        elif kind == "wordml":
            analyzer_name = "WordMLParser(min)"
            mime = "application/xml"
        else:
            analyzer_name = "OOXMLParser"
            mime = "application/vnd.openxmlformats-officedocument"

        macros_in = report.get("macros", {}) or {}
        macros_out = dict(macros_in)
        if "vba_present" not in macros_out:
            macros_out["vba_present"] = bool(macros_out.get("has_vba", False))

        return {
            "schema_version": "1.0",
            "report_id": os.path.basename(file_path),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "analyzer": {
                "name": analyzer_name
            },
            "file": {
                "filename": os.path.basename(file_path),
                "extension": os.path.splitext(file_path)[1].lower(),
                "size_bytes": os.path.getsize(file_path),
                "mime_type": mime,
                "hash": {"sha256": sha256},
            },
            "features": report,
            "risk_assessment": {},
            "x_extensions": {},
        }

    try:
        macros = (((report.get("ooxml") or {}).get("macros")) 
                  or ((report.get("features") or {}).get("macros"))
                  or {})
        if "vba_present" not in macros and "has_vba" in macros:
            macros["vba_present"] = bool(macros.get("has_vba"))
    except Exception:
        pass
    return report


def validate_report(report: Dict[str, Any], schema_path: str) -> None:
    """ooxml_schema.json으로 JSON Schema validate. 위반 시 예외 발생."""
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    validator = Validator(schema)
    errors = sorted(validator.iter_errors(report), key=lambda e: e.path)
    if errors:
        first = errors[0]
        loc = "$." + ".".join([str(x) for x in first.path])
        raise ValueError(f"json schema validation failed at {loc}: {first.message}")


def filter_json_by_schema(input_path: str, schema_path: str, output_path: str) -> dict:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    def _filter_dict_by_schema(d: dict, s: dict) -> dict:
        if not isinstance(d, dict):
            return d
        result = {}
        props = s.get("properties", {})
        for key, subschema in props.items():
            if key in d:
                result[key] = _filter_dict_by_schema(d[key], subschema)
        return result

    filtered = _filter_dict_by_schema(data, schema)

    if "analyzer" in data:
        filtered["analyzer"] = data["analyzer"]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    return filtered

def filter_report_dict_by_schema(report: dict, schema_path: str) -> dict:
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    def _filter(d: dict, s: dict) -> dict:
        if not isinstance(d, dict):
            return d
        result = {}
        props = s.get("properties", {})
        for k, subschema in props.items():
            if k in d:
                result[k] = _filter(d[k], subschema)
    
        if s.get("additionalProperties"):
            for k, v in d.items():
                if k not in props:  
                    result[k] = v   
        return result


    filtered = _filter(report, schema)
    if "analyzer" in report:
        filtered["analyzer"] = report["analyzer"]
    return filtered


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="path to OOXML file (docx/docm/xlsx/xlsm/pptx/pptm)")
    ap.add_argument("--schema", required=True, help="path to ooxml_schema.json")
    ap.add_argument("--out", help="output json path")
    args = ap.parse_args()

    print(f"exported & validated: {args.out or (os.path.splitext(args.file)[0]+'.json')}")
