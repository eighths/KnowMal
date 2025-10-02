from __future__ import annotations

import os
import re
import json
import zipfile
import traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple, Iterable, Optional

import magic 
import olefile

from app.core.static_analyzer.parsers.ole_parser import OLEParser      # OLE 파서
from app.core.static_analyzer.exporter import _normalize_report, validate_report  # 내부 사용 허용

OOXML_EXTS = {".docx", ".docm", ".xlsx", ".xlsm", ".pptx", ".pptm", ".xlam"}
OLE_EXTS = {".doc", ".xls", ".ppt", ".xlsb", ".xla"}
ZIP_EXTS   = {".zip"}
ALL_EXTS   = OOXML_EXTS | OLE_EXTS | ZIP_EXTS

OOXML_MIME_PREFIXES = {"application/vnd.openxmlformats-officedocument."}
ZIP_MIME = {"application/zip"}
OLE_MIME_CANDIDATES = {
    "application/x-ole-storage",
    "application/vnd.ms-office",
    "application/msword",
    "application/vnd.ms-excel",
    "application/vnd.ms-powerpoint",
    "application/CDFV2-corrupt",
}

WORDML_SIG_RE = re.compile(
    br"<\?mso-application\s+progid=\"Word\.Document\"|<w:wordDocument\b",
    re.I
)

def _magic_mime(path: str) -> str:
    if magic is None:
        return ""
    try:
        return magic.from_file(path, mime=True) or ""
    except Exception:
        return ""

def _looks_like_ooxml_by_zip(path: str) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            return "[Content_Types].xml" in zf.namelist()
    except Exception:
        return False

def _looks_like_ole(path: str) -> bool:
    if olefile is None:
        return False
    try:
        return olefile.isOleFile(path)
    except Exception:
        return False

def _looks_like_wordml2003(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(8192)
        return bool(WORDML_SIG_RE.search(head))
    except Exception:
        return False

def _read_magic(path: str, n: int = 8) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read(n)
    except Exception:
        return b""

def choose_parser(file_path: str) -> str:
    head = _read_magic(file_path, 8)
    if head.startswith(b"PK\x03\x04"):
        if _looks_like_ooxml_by_zip(file_path):
            return "ooxml"
    if head.startswith(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"):
        if _looks_like_ole(file_path):
            return "ole"

    if _looks_like_ooxml_by_zip(file_path):
        return "ooxml"
    if _looks_like_ole(file_path):
        return "ole"
    if _looks_like_wordml2003(file_path):
        return "wordml"

    ext = os.path.splitext(file_path)[1].lower()
    if ext in OOXML_EXTS:
        return "ooxml"
    if ext in OLE_EXTS:
        if _looks_like_ooxml_by_zip(file_path):
            return "ooxml"
        if _looks_like_ole(file_path):  
            return "ole"
        if _looks_like_wordml2003(file_path):
            return "wordml"

    mime = _magic_mime(file_path)
    if mime:
        if any(mime.startswith(pfx) for pfx in OOXML_MIME_PREFIXES):
            return "ooxml"
        if mime in ZIP_MIME and _looks_like_ooxml_by_zip(file_path):
            return "ooxml"
        if mime in OLE_MIME_CANDIDATES and _looks_like_ole(file_path):
            return "ole"

    if _looks_like_ooxml_by_zip(file_path):
        return "ooxml"
    if _looks_like_ole(file_path):
        return "ole"
    if _looks_like_wordml2003(file_path):
        return "wordml"

    return "unknown"

def _parse_wordml_min(file_path: str) -> Dict[str, Any]:
    from app.core.static_analyzer.utils.feature_utils import (
        extract_urls, extract_ips, extract_filepaths, extract_registry_keys,
        extract_obfuscated_strings, detect_obfuscation_ops, extract_user_agents,
        domain_from_url, filter_nonstandard_domains
    )
    feats = {
        "structure": {"format": "wordml_2003", "streams_count": 1, "document_xml_wellformed": True, "metadata_anomalies": []},
        "macros": {}, "strings": {}, "apis": {}, "obfuscation": {}, "network_indicators": {}, "security_indicators": {},
    }
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        text = data.decode("utf-8", "ignore")
    except Exception:
        text = ""

    feats["macros"]["has_vba"] = ("macrosPresent=\"yes\"" in text) or ("w:macrosPresent=\"yes\"" in text)

    urls = extract_urls(text)
    ips = extract_ips(text)
    feats["strings"]["urls"] = urls
    feats["strings"]["ips"] = ips
    feats["strings"]["filepaths"] = extract_filepaths(text)
    feats["strings"]["registry_keys"] = extract_registry_keys(text)

    doms = sorted({d for d in (domain_from_url(u) for u in urls) if d})
    feats["network_indicators"]["urls"] = urls
    feats["network_indicators"]["domains"] = filter_nonstandard_domains(doms)
    feats["network_indicators"]["user_agents"] = extract_user_agents(text)

    feats["obfuscation"]["suspicious_strings"] = extract_obfuscated_strings(text)
    feats["obfuscation"]["obfuscation_ops"] = detect_obfuscation_ops(text)

    feats["apis"] = {"winapi_calls": [], "com_progids": []}
    feats["security_indicators"] = {"amsi_bypass": [], "edr_evasion": []}
    return feats

def parse_with_dispatcher(file_path: str) -> Dict[str, Any]:
    kind = choose_parser(file_path)

    if kind == "ole":
        if OLEParser is None:
            raise NotImplementedError("ole_parser not available")
        try:
            return OLEParser(file_path).parse()
        except OSError:
            raise

    if kind == "wordml":
        return _parse_wordml_min(file_path)

    raise NotImplementedError("unsupported file format")

dispatch = parse_with_dispatcher
run = parse_with_dispatcher

OOXML_SCHEMA_NAME = "ooxml_schema.json"
OLE_SCHEMA_NAME   = "ole_schema.json"  

def _iter_files(target: Path, exts: Optional[set[str]] = None) -> Iterable[Path]:
    if target.is_file():
        if not exts or target.suffix.lower() in exts:
            yield target
        return
    for p in target.rglob("*"):
        if p.is_file() and (not exts or p.suffix.lower() in exts):
            yield p

def _schema_for(kind: str, schemas_dir: Path) -> Path:
    if kind == "ooxml" or kind == "wordml":
        return schemas_dir / OOXML_SCHEMA_NAME
    if kind == "ole":
        return schemas_dir / OLE_SCHEMA_NAME
    raise ValueError(f"unknown kind for schema: {kind}")

def _process_one(file_path: Path, schemas_dir: Path, out_dir: Path) -> Tuple[str, str]:
    try:
        kind = choose_parser(str(file_path))
        if kind == "unknown":
            return "skip", f"unknown format: {file_path.name}"

        features = parse_with_dispatcher(str(file_path))
        report = _normalize_report(features, str(file_path))   

        schema_path = _schema_for(kind, schemas_dir)
        validate_report(report, str(schema_path))               

        from app.core.static_analyzer.exporter import filter_report_dict_by_schema
        report = filter_report_dict_by_schema(report, str(schema_path))


        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{file_path.name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        return "ok", f"{file_path.name} → {kind} → {schema_path.name} validated → {out_path}"
    except Exception as e:
        return "error", f"{file_path.name}: {e}\n{traceback.format_exc(limit=2)}"

def batch_dispatch(
    input_path: str | Path,
    schemas_dir: str | Path = "analyzer/schemas",
    out_dir: str | Path = "tests/logs",
    exts_csv: str = "",
) -> Dict[str, Any]:
    schemas_dir = Path(schemas_dir)
    out_dir = Path(out_dir)
    target = Path(input_path)

    if exts_csv.strip():
        exts = {e.strip().lower() for e in exts_csv.split(",") if e.strip().startswith(".")}
    else:
        exts = ALL_EXTS

    files = list(_iter_files(target, exts))
    if not files:
        return {"ok": 0, "skip": 0, "error": 0, "total": 0, "details": [], "note": "no input files matched"}

    ok = err = skip = 0
    details: List[Tuple[str, str]] = []
    for f in files:
        status, msg = _process_one(f, schemas_dir, out_dir)
        details.append((status, msg))
        if status == "ok":
            ok += 1
        elif status == "skip":
            skip += 1
        else:
            err += 1

    return {"ok": ok, "skip": skip, "error": err, "total": len(files), "details": details}

if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser(description="Dispatcher: auto-route (OOXML/OLE/WordML), validate, and export")
    ap.add_argument("input", help="파일 또는 폴더 경로 (예: tests/test_samples)")
    ap.add_argument("--schemas", default="analyzer/schemas", help="스키마 폴더 (ooxml_schema.json, ole_schema.json 포함)")
    ap.add_argument("--out_dir", default="tests/logs", help="JSON 리포트 저장 폴더")
    ap.add_argument("--exts", default="", help="확장자 필터(예: .docx,.docm). 비우면 OOXML+OLE 기본 세트")
    args = ap.parse_args()

    res = batch_dispatch(args.input, args.schemas, args.out_dir, args.exts)

    for status, msg in res.get("details", []):
        print(f"[{status}] {msg}")
    print(f"\nsummary: ok={res['ok']}, skip={res['skip']}, error={res['error']}, total={res['total']}")
    if res.get("note"):
        print(res["note"])
    if res["error"] > 0:
        sys.exit(1)
