from __future__ import annotations
import hashlib, json, os, io, magic, tempfile, zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

try:
    import rarfile
    RARFILE_AVAILABLE = True
except ImportError:
    RARFILE_AVAILABLE = False

try:
    import py7zr
    PY7ZR_AVAILABLE = True
except ImportError:
    PY7ZR_AVAILABLE = False

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


def is_archive_file(file_bytes: bytes, filename: str = "") -> bool:
    mime = sniff_mime(file_bytes=file_bytes)
    
    archive_mimes = [
        'application/zip', 'application/x-zip-compressed',
        'application/x-rar-compressed', 'application/vnd.rar',
        'application/x-7z-compressed', 'application/x-tar',
        'application/gzip', 'application/x-gzip'
    ]
    
    if any(mime.startswith(am) for am in archive_mimes):
        return True
    
    if filename:
        archive_extensions = ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2']
        if any(filename.lower().endswith(ext) for ext in archive_extensions):
            return True
    
    if file_bytes:
        # ZIP 시그니처
        if file_bytes.startswith(b'PK\x03\x04') or file_bytes.startswith(b'PK\x05\x06') or file_bytes.startswith(b'PK\x07\x08'):
            return True
        # RAR 시그니처
        if file_bytes.startswith(b'Rar!\x1a\x07\x00') or file_bytes.startswith(b'Rar!\x1a\x07\x01\x00'):
            return True
        # 7z 시그니처
        if file_bytes.startswith(b'7z\xbc\xaf\x27\x1c'):
            return True
    
    return False


def extract_archive_files(file_bytes: bytes, filename: str = "", password: str = "pass") -> List[Dict[str, Any]]:
    extracted_files = []
    
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file_bytes)
            temp_file.flush()
            temp_path = temp_file.name
        
        try:
            if zipfile.is_zipfile(temp_path):
                with zipfile.ZipFile(temp_path, 'r') as zf:
                    for file_info in zf.filelist:
                        if file_info.is_dir():
                            continue
                        
                        try:
                            try:
                                file_data = zf.read(file_info.filename, pwd=password.encode('utf-8'))
                            except (RuntimeError, zipfile.BadZipFile):
                                try:
                                    file_data = zf.read(file_info.filename)
                                except Exception:
                                    continue
                            
                            file_report = analyze_bytes(
                                file_bytes=file_data,
                                filename=file_info.filename,
                                use_cache=True,
                                include_virustotal=True
                            )
                            
                            extracted_files.append({
                                'filename': file_info.filename,
                                'size_bytes': len(file_data),
                                'compressed_size': file_info.compress_size,
                                'report': file_report
                            })
                            
                        except Exception as e:
                            continue
            
            elif RARFILE_AVAILABLE and rarfile.is_rarfile(temp_path):
                try:
                    with rarfile.RarFile(temp_path, 'r') as rf:
                        rf.setpassword(password)
                        for file_info in rf.infolist():
                            if file_info.is_dir():
                                continue
                            
                            try:
                                file_data = rf.read(file_info.filename)
                                
                                file_report = analyze_bytes(
                                    file_bytes=file_data,
                                    filename=file_info.filename,
                                    use_cache=True,
                                    include_virustotal=True
                                )
                                
                                extracted_files.append({
                                    'filename': file_info.filename,
                                    'size_bytes': len(file_data),
                                    'compressed_size': file_info.compress_size,
                                    'report': file_report
                                })
                                
                            except Exception as e:
                                continue
                except Exception as e:
                    print(f"RAR 파일 처리 오류: {e}")
            
            elif PY7ZR_AVAILABLE and py7zr.is_7zfile(temp_path):
                try:
                    with py7zr.SevenZipFile(temp_path, mode="r", password=password) as szf:
                        for file_info in szf.list():
                            if file_info.is_directory:
                                continue
                            
                            try:
                                extracted_data = szf.read([file_info.filename])
                                file_data = extracted_data[file_info.filename].read()
                                
                                file_report = analyze_bytes(
                                    file_bytes=file_data,
                                    filename=file_info.filename,
                                    use_cache=True,
                                    include_virustotal=True
                                )
                                
                                extracted_files.append({
                                    'filename': file_info.filename,
                                    'size_bytes': len(file_data),
                                    'compressed_size': file_info.compressed,
                                    'report': file_report
                                })
                                
                            except Exception as e:
                                continue
                except Exception as e:
                    print(f"7z 파일 처리 오류: {e}")
        
        finally:
            # 임시 파일 정리
            try:
                os.unlink(temp_path)
            except Exception:
                pass
    
    except Exception as e:
        print(f"압축 파일 처리 중 오류: {e}")
    
    return extracted_files


def _get_archive_type(file_bytes: bytes, filename: str = "") -> str:
    if zipfile.is_zipfile(io.BytesIO(file_bytes)):
        return "zip"
    elif file_bytes.startswith(b'Rar!'):
        return "rar"
    elif file_bytes.startswith(b'7z\xbc\xaf\x27\x1c'):
        return "7z"
    elif filename.lower().endswith('.tar'):
        return "tar"
    elif filename.lower().endswith('.gz'):
        return "gzip"
    else:
        return "unknown"


def _count_malicious_files(extracted_files: List[Dict[str, Any]]) -> int:
    count = 0
    for file_info in extracted_files:
        report = file_info.get('report', {})
        
        vt_result = report.get('virustotal', {})
        if vt_result.get('available') and vt_result.get('scan_summary', {}).get('malicious', 0) > 0:
            count += 1
            continue
        
        ai_prediction = report.get('ai_prediction', {})
        if ai_prediction and ai_prediction.get('ai_analysis', {}).get('predicted_types'):
            predicted_types = ai_prediction['ai_analysis']['predicted_types']
            if any(t != 'Normal' for t in predicted_types):
                count += 1
    
    return count


def _calculate_archive_risk_score(extracted_files: List[Dict[str, Any]]) -> float:
    if not extracted_files:
        return 0.0
    
    total_files = len(extracted_files)
    malicious_files = _count_malicious_files(extracted_files)
    
    base_risk = malicious_files / total_files if total_files > 0 else 0.0
    
    high_risk_extensions = ['.exe', '.scr', '.bat', '.cmd', '.com', '.pif', '.vbs', '.js']
    suspicious_files = 0
    
    for file_info in extracted_files:
        filename = file_info.get('filename', '').lower()
        if any(filename.endswith(ext) for ext in high_risk_extensions):
            suspicious_files += 1
    
    suspicious_ratio = suspicious_files / total_files if total_files > 0 else 0.0
    
    final_risk = min(1.0, base_risk + (suspicious_ratio * 0.3))
    
    return final_risk


def _get_most_dangerous_vt_result(extracted_files: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    max_detection_rate = 0.0
    most_dangerous_vt = None
    
    for file_info in extracted_files:
        report = file_info.get('report', {})
        vt_result = report.get('virustotal', {})
        
        if vt_result.get('available') and vt_result.get('scan_summary'):
            scan_summary = vt_result['scan_summary']
            malicious = scan_summary.get('malicious', 0)
            total = scan_summary.get('total', 0)
            
            if total > 0:
                detection_rate = malicious / total
                if detection_rate > max_detection_rate:
                    max_detection_rate = detection_rate
                    most_dangerous_vt = vt_result
    
    return most_dangerous_vt


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
    
    if is_archive_file(file_bytes, filename):
        extracted_files = extract_archive_files(file_bytes, filename)
        
        report = {
            "schema_version": "1.0",
            "report_id": filename,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "analyzer": {"name": "ArchiveAnalyzer"},
            "file": {
                "filename": filename,
                "extension": Path(filename).suffix.lower(),
                "size_bytes": len(file_bytes),
                "mime_type": mime,
                "hash": {"sha256": sha256},
                "is_archive": True
            },
            "features": {
                "structure": {
                    "format": "archive",
                    "container_type": _get_archive_type(file_bytes, filename),
                    "files_count": len(extracted_files),
                    "malicious_files_count": _count_malicious_files(extracted_files)
                },
                "macros": {"has_vba": False, "vba_present": False},
                "strings": {},
                "apis": {},
                "obfuscation": {},
                "network_indicators": {},
                "security_indicators": {}
            },
            "risk_assessment": {
                "overall_risk_score": _calculate_archive_risk_score(extracted_files)
            },
            "x_extensions": {},
            "embedded_files": extracted_files
        }
        
        if include_virustotal and extracted_files:
            most_dangerous_vt = _get_most_dangerous_vt_result(extracted_files)
            if most_dangerous_vt:
                report["virustotal"] = most_dangerous_vt
        
        r.setex(k, ttl_sec, json.dumps(report, ensure_ascii=False))
        return report
    
    is_office_file = ("msword" in mime or mime.endswith("ole") or "olecf" in mime or 
                      filename.lower().endswith(('.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx')))
    
    if is_office_file:
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
        try:
            text_content = ""
            try:
                text_content = file_bytes.decode("utf-8", "ignore")
            except:
                try:
                    text_content = file_bytes.decode("latin-1", "ignore")
                except:
                    text_content = ""
            
            from app.core.static_analyzer.utils.feature_utils import (
                extract_urls, extract_ips, extract_filepaths, extract_registry_keys,
                extract_obfuscated_strings, detect_obfuscation_ops, extract_user_agents,
                domain_from_url, filter_nonstandard_domains
            )
            
            urls = extract_urls(text_content)
            ips = extract_ips(text_content)
            domains = sorted({d for d in (domain_from_url(u) for u in urls) if d})
            
            features = {
                "structure": {
                    "format": "binary", 
                    "metadata_anomalies": [],
                    "streams_count": 1,
                    "storages_count": 0
                },
                "macros": {
                    "has_vba": False,
                    "vba_present": False,
                    "modules": [],
                    "autoexec_triggers": [],
                    "suspicious_api_calls_count": 0
                },
                "strings": {
                    "urls": urls,
                    "ips": ips,
                    "filepaths": extract_filepaths(text_content),
                    "registry_keys": extract_registry_keys(text_content)
                },
                "apis": {
                    "winapi_calls": [],
                    "com_progids": []
                },
                "obfuscation": {
                    "suspicious_strings": extract_obfuscated_strings(text_content),
                    "obfuscation_ops": detect_obfuscation_ops(text_content)
                },
                "network_indicators": {
                    "urls": urls,
                    "domains": filter_nonstandard_domains(domains),
                    "user_agents": extract_user_agents(text_content)
                },
                "security_indicators": {
                    "motw_present": False,
                    "macro_security_hint": "unknown",
                    "digital_signature": {"signed": False},
                    "amsi_bypass": [],
                    "edr_evasion": []
                }
            }
            
        except Exception as e:
            features = {
                "structure": {"format": "unknown", "metadata_anomalies": [f"analysis_failed: {str(e)}"]},
                "macros": {},
                "strings": {},
                "apis": {},
                "obfuscation": {},
                "network_indicators": {},
                "security_indicators": {}
            }
        
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
            "features": features,
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
            return json.loads(cached_result)
        
        vt_client = get_virustotal_client()
        vt_result = vt_client.get_file_analysis(sha256)
        
        cache_ttl = settings.VT_CACHE_TTL if vt_result.get("available") else 3600  
        r.setex(vt_cache_key, cache_ttl, json.dumps(vt_result, ensure_ascii=False))
        
        return vt_result
        
    except Exception as e:
        return None
