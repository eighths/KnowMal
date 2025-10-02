from __future__ import annotations
import math, re, zipfile, io
from collections import defaultdict, Counter
from typing import Iterable, List, Dict, Any, Tuple, Optional
from urllib.parse import urlsplit
from datetime import datetime

def calculate_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    freq = defaultdict(int)
    for b in data:
        freq[b] += 1
    total = len(data)
    return -sum((c/total) * math.log2(c/total) for c in freq.values())

def calculate_entropy_from_zip_bytes(file_bytes: bytes) -> float:
    return calculate_entropy(file_bytes or b"")

_URL_RE = re.compile(r"(?i)\b(?:https?|hxxps?)://[^\s\"'<>)\]]+")
_IP_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b")
_FILEPATH_RE = re.compile(r"(?:[A-Za-z]:\\|\\\\)[^\s\"']+")
_REGKEY_RE = re.compile(r"\bHKEY_[A-Z_]+\\[^\s\"']+\b")

_OBF_SNIPPET_RE = re.compile(
    r"(?i)\b(?:chrw?|strreverse|mid\$?|left\$?|right\$?|split|replace|join|array|val|ascw?|environ|frombase64string|base64|execute\w*|eval)\s*\([^)]*\)"
)

_USER_AGENT_LINE_RE = re.compile(r"(?im)^\s*User-Agent\s*:\s*(.+?)\s*$")
_USER_AGENT_TOKEN_RE = re.compile(r"\b(Mozilla/\d+\.\d+|Chrome/\d+\.\d+|Firefox/\d+\.\d+|Safari/\d+\.\d+|Edge/\d+\.\d+)\b")

def extract_urls(text: str) -> List[str]:
    return sorted(list(set(_URL_RE.findall(text or ""))))

def extract_ips(text: str) -> List[str]:
    return sorted(list(set(_IP_RE.findall(text or ""))))

def extract_filepaths(text: str) -> List[str]:
    return sorted(list(set(_FILEPATH_RE.findall(text or ""))))

def extract_registry_keys(text: str) -> List[str]:
    return sorted(list(set(_REGKEY_RE.findall(text or ""))))

def extract_obfuscated_strings(text: str) -> List[str]:
    return sorted(list(set(_OBF_SNIPPET_RE.findall(text or ""))))

def extract_user_agents(text: str) -> List[str]:
    lines = [m.strip() for m in _USER_AGENT_LINE_RE.findall(text or "")]
    tokens = _USER_AGENT_TOKEN_RE.findall(text or "")
    return sorted(list(set(lines + tokens)))

def domain_from_url(url: str) -> Optional[str]:
    try:
        host = urlsplit(url).hostname
        return host.lower() if host else None
    except Exception:
        return None

def derive_domains(urls: Iterable[str]) -> List[str]:
    doms = sorted({d for d in (domain_from_url(u) for u in (urls or [])) if d})
    return filter_nonstandard_domains(doms)

def detect_obfuscation_ops(blobs: Iterable[str]) -> List[str]:
    suspicious = set()
    keywords = ['chr', 'reverse', 'mid', 'left', 'right', 'xor', 'eval', 'execute', 'frombase64string', 'split', 'replace']
    for s in blobs or []:
        low = (s or "").lower()
        for kw in keywords:
            if kw in low:
                suspicious.add(kw)
    return sorted(list(suspicious))

def detect_amsi_bypass(blobs: Iterable[str]) -> bool:
    patterns = [
        r"System\.Management\.Automation\.AmsiUtils",
        r"AmsiScanBuffer",
        r"amsiInitFailed\s*=\s*1",
        r"(?i)AMSI(?=.*?(Utils|Scan|Context))",
    ]
    combined = "\n".join(list(blobs or []))
    for pat in patterns:
        if re.search(pat, combined):
            return True
    return False

WINAPI_RISK_MAP: Dict[str, str] = {
    # high
    "ShellExecute": "high", "ShellExecuteA": "high", "ShellExecuteW": "high",
    "CreateProcess": "high", "CreateProcessA": "high", "CreateProcessW": "high",
    "WinExec": "high",
    "URLDownloadToFile": "high", "URLDownloadToFileA": "high", "URLDownloadToFileW": "high",
    "WriteProcessMemory": "high",
    # medium
    "VirtualAlloc": "medium", "VirtualAllocEx": "medium",
    "GetProcAddress": "medium", "LoadLibrary": "medium", "LoadLibraryA": "medium", "LoadLibraryW": "medium",
    "CreateRemoteThread": "medium",
    "InternetOpen": "medium", "InternetOpenA": "medium", "InternetOpenW": "medium",
    "InternetOpenUrl": "medium", "InternetOpenUrlA": "medium", "InternetOpenUrlW": "medium",
    "HttpOpenRequest": "medium", "HttpOpenRequestA": "medium", "HttpOpenRequestW": "medium",
    "HttpSendRequest": "medium", "HttpSendRequestA": "medium", "HttpSendRequestW": "medium",
    "WinHttpOpen": "medium", "WinHttpConnect": "medium", "WinHttpOpenRequest": "medium", "WinHttpSendRequest": "medium",
    # low
    "GetSystemDirectory": "low", "GetTempPath": "low",
    "RegOpenKey": "low", "CreateFile": "low",
}

SUSPICIOUS_COM_PROGIDS = {
    "WScript.Shell", "WScript.Network", "Shell.Application", "Scripting.FileSystemObject",
    "Scripting.Dictionary", "ADODB.Stream",
    "MSXML2.XMLHTTP", "MSXML2.XMLHTTP.6.0", "MSXML2.ServerXMLHTTP", "MSXML2.ServerXMLHTTP.6.0",
    "WinHttp.WinHttpRequest.5.1",
    "WbemScripting.SWbemLocator",
    "Excel.Application", "Outlook.Application"
}

_API_TOKEN_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\s*\(", re.M)
_API_DECLARE_RE = re.compile(r"(?i)\bDeclare\s+(?:PtrSafe\s+)?(?:Function|Sub)\s+([A-Za-z_][A-Za-z0-9_]{2,})\b")

def find_winapi_calls(text: str) -> List[Dict[str, Any]]:
    counter: Counter[str] = Counter()
    t = text or ""
    for m in _API_TOKEN_RE.finditer(t):
        name = m.group(1)
        if name in WINAPI_RISK_MAP:
            counter[name] += 1
    for m in _API_DECLARE_RE.finditer(t):
        name = m.group(1)
        if name in WINAPI_RISK_MAP:
            counter[name] += 1
    hits: List[Dict[str, Any]] = []
    for name, cnt in counter.items():
        hits.append({"name": name, "count": int(cnt), "risk": WINAPI_RISK_MAP.get(name, "low")})
    risk_order = {"high": 0, "medium": 1, "low": 2}
    hits.sort(key=lambda x: (risk_order.get(x["risk"], 3), -x["count"], x["name"].lower()))
    return hits

def find_com_progids(text: str) -> List[str]:
    found = []
    low = (text or "").lower()
    for progid in SUSPICIOUS_COM_PROGIDS:
        if progid.lower() in low:
            found.append(progid)
    return sorted(list(set(found)))


ALLOWLIST_STD_DOMAINS = {
    "www.w3.org", "schemas.openxmlformats.org", "purl.org", "schemas.microsoft.com",
    "openxmlformats.org", "ns.adobe.com", "schemas.xmlsoap.org", "www.w3schools.com",
    "xml.apache.org", "www.ecma-international.org",
    "schema.org", "aka.ms", "go.microsoft.com", "learn.microsoft.com",
    "support.microsoft.com", "officeapps.live.com"
}

def filter_nonstandard_domains(domains):
    out = []
    for d in domains or []:
        host = (d or "").lower()
        if host in ALLOWLIST_STD_DOMAINS:
            continue
        for allow in ALLOWLIST_STD_DOMAINS:
            if host.endswith("." + allow):
                break
        else:
            out.append(d)
    return sorted(list(dict.fromkeys(out)))

def dedup_sort_strings(items: Iterable[str] | None) -> List[str]:
    if not items:
        return []
    try:
        return sorted({s for s in items if isinstance(s, str) and s.strip()})
    except Exception:
        return [s for s in items if isinstance(s, str)]

def _clamp(v: float, lo: float = 0.0, hi: float = 8.0) -> float:
    return max(lo, min(hi, float(v)))

def clamp_entropy_fields(entropy: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(entropy or {})
    for k in ("overall", "max", "min", "avg"):
        v = out.get(k)
        if isinstance(v, (int, float)):
            out[k] = _clamp(v)
    return out

def ensure_classification_booleans(cf: Dict[str, Any] | None) -> Dict[str, Any]:
    return dict(cf or {})

_CN_RE = re.compile(r"CN\s*=\s*([^,;\r\n]+)")
_DT_RE = re.compile(
    r"(\d{4})[-/\.](\d{2})[-/\.](\d{2})[ T](\d{2}):(\d{2})(?::(\d{2}))?Z?", re.I
)

def _ascii_safe(b: bytes) -> str:
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        try:
            return b.decode("latin-1", errors="ignore")
        except Exception:
            return ""

def guess_pkcs7_sign_info(pkcs7_bytes: bytes) -> Tuple[Optional[str], Optional[str]]:
    if not pkcs7_bytes:
        return (None, None)
    txt = _ascii_safe(pkcs7_bytes)

    publisher = None
    m = _CN_RE.search(txt)
    if m:
        publisher = m.group(1).strip()

    ts = None
    dm = _DT_RE.search(txt)
    if dm:
        y, mo, d, H, M, S = dm.groups()
        if S is None:
            S = "00"
        try:
            ts_iso = datetime(int(y), int(mo), int(d), int(H), int(M), int(S)).isoformat()
            ts = ts_iso
        except Exception:
            ts = None

    return (publisher, ts)

def count_vba_references(project_text: str) -> int:
    if not project_text:
        return 0
    cnt = 0
    for line in project_text.splitlines():
        if line.strip().lower().startswith("reference="):
            cnt += 1
    return cnt
