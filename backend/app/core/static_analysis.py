from __future__ import annotations
import hashlib, json, os, io, magic, tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.static_analyzer.parsers.ole_parser import OLEParser
from app.external.virustotal import get_virustotal_client

from app.cache.redis_client import get_redis
from app.cache.keys import file_data_key, file_metadata_key
from jsonschema import validate

BASE_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = (BASE_DIR / "static_analyzer" / "schemas" / "ole_schema.json").resolve()
if not SCHEMA_PATH.exists():
    SCHEMA_PATH = Path(__file__).resolve().parent.parent / "core" / "static_analyzer" / "schemas" / "ole_schema.json"

def _sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _sha256_of_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sniff_mime(path: str | None = None, file_bytes: bytes | None = None) -> str:
    m = magic.Magic(mime=True)
    if file_bytes is not None:
        return m.from_buffer(file_bytes)
    elif path:
        return m.from_file(path)
    raise ValueError("sniff_mime: path or file_bytes required")

def extract_excerpt(path: str = None, mime: str = None, max_len: int = 4000, file_bytes: bytes = None) -> str:
    excerpt = ""
    
    if file_bytes is not None:
        if mime and (mime.startswith("text/") or mime.endswith("xml")):
            try:
                excerpt = file_bytes.decode("utf-8", errors="ignore")[:max_len]
            except Exception:
                pass
        return excerpt
    
    if path and mime and (mime.startswith("text/") or mime.endswith("xml")):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                excerpt = f.read(max_len)
        except Exception:
            pass
    return excerpt


def _normalize_report(features: Dict[str, Any], file_path: str, file_bytes: bytes = None) -> Dict[str, Any]:
    p = Path(file_path)
    
    if file_bytes is not None:
        sha256 = _sha256_of_bytes(file_bytes)
        size_bytes = len(file_bytes)
    else:
        sha256 = _sha256_of_file(file_path)
        size_bytes = p.stat().st_size

    macros = dict(features.get("macros") or {})
    if "vba_present" not in macros and "has_vba" in macros:
        macros["vba_present"] = bool(macros["has_vba"])
    features = dict(features)
    features["macros"] = macros

    return {
        "schema_version": "1.0",
        "report_id": p.name,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "analyzer": {"name": "OLEParser"},
        "file": {
            "filename": p.name,
            "extension": p.suffix.lower(),
            "size_bytes": size_bytes,
            "mime_type": "application/vnd.ms-office",
            "hash": {"sha256": sha256},
        },
        "features": features,
        "risk_assessment": {},
        "x_extensions": {},
    }

def _validate_report(report: Dict[str, Any], schema_path: str) -> None:
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    validate(instance=report, schema=schema)

def _redis():
    if get_redis:
        return get_redis()
    import redis
    return redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))

def analyze_file(file_path: str, *, ttl_sec: int = 3600, use_cache: bool = True) -> Dict[str, Any]:
    sha256 = _sha256_of_file(file_path)
    r = _redis()
    cache_key = file_metadata_key(sha256)
    if use_cache:
        cached = r.get(cache_key)
        if cached:
            return json.loads(cached)

    features = OLEParser(file_path).parse()  

    report = _normalize_report(features, file_path)
    _validate_report(report, str(SCHEMA_PATH))

    r.setex(cache_key, ttl_sec, json.dumps(report))
    return report

def get_cached_report(sha256: str) -> Optional[Dict[str, Any]]:
    r = _redis()
    cached = r.get(file_metadata_key(sha256))
    return json.loads(cached) if cached else None

def analyze_bytes(file_bytes: bytes, filename: str = "upload.bin", *, ttl_sec: int = 3600, use_cache: bool = True, include_virustotal: bool = True) -> dict:
    sha256 = _sha256_of_bytes(file_bytes)
    r = get_redis()
    k = file_data_key(sha256)  

    if use_cache:
        cached = r.get(k)
        if cached:
            return json.loads(cached)

    mime = sniff_mime(file_bytes=file_bytes)
    
    if "msword" in mime or mime.endswith("ole") or "olecf" in mime:
        try:
            parser = OLEParser(io.BytesIO(file_bytes))
            features = parser.parse()
            
            report = _normalize_report(features, filename, file_bytes)
            
            _validate_report(report, str(SCHEMA_PATH))
                    
        except Exception as e:
            report = {
                "schema_version": "1.0",
                "report_id": filename,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "analyzer": {"name": "OLEParser"},
                "file": {
                    "filename": filename, 
                    "extension": Path(filename).suffix.lower(),
                    "size_bytes": len(file_bytes),
                    "mime_type": mime,
                    "hash": {"sha256": sha256}
                },
                "features": {
                    "structure": {"format": "ole", "ole_header_valid": False, "metadata_anomalies": [f"analysis_failed: {str(e)}"]},
                    "macros": {},
                    "strings": {},
                    "apis": {},
                    "obfuscation": {},
                    "network_indicators": {},
                    "security_indicators": {}
                },
                "risk_assessment": {},
                "x_extensions": {}
            }
    else:
        report = {
            "schema_version": "1.0",
            "report_id": filename,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "analyzer": {"name": "OLEParser"},
            "file": {
                "filename": filename, 
                "extension": Path(filename).suffix.lower(),
                "size_bytes": len(file_bytes),
                "mime_type": mime,
                "hash": {"sha256": sha256}
            },
            "features": {
                "structure": {"format": "unknown", "metadata_anomalies": ["unsupported_format"]},
                "macros": {},
                "strings": {},
                "apis": {},
                "obfuscation": {},
                "network_indicators": {},
                "security_indicators": {}
            },
            "risk_assessment": {},
            "x_extensions": {}
        }

    if include_virustotal:
        vt_result = _get_virustotal_analysis(sha256)
        if vt_result:
            report["virustotal"] = vt_result
    
    r.setex(k, ttl_sec, json.dumps(report, ensure_ascii=False))
    return report

def _get_virustotal_analysis(sha256: str) -> Optional[Dict[str, Any]]:
    from app.config import get_settings
    
    settings = get_settings()
    if not settings.VT_ENABLED or not settings.VT_API_KEY:
        return None
    
    try:
        r = get_redis()
        vt_cache_key = f"vt:{sha256}"
        cached_result = r.get(vt_cache_key)
        
        if cached_result:
            print(f"VT 캐시 사용: {sha256[:16]}...")
            return json.loads(cached_result)
        
        vt_client = get_virustotal_client()
        vt_result = vt_client.get_file_analysis(sha256)
        
        cache_ttl = settings.VT_CACHE_TTL if vt_result.get("available") else 3600  
        r.setex(vt_cache_key, cache_ttl, json.dumps(vt_result, ensure_ascii=False))
        
        return vt_result
        
    except Exception as e:
        print(f"VirusTotal 분석 실패: {e}")
        return None
