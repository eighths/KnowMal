from __future__ import annotations

import os
import io
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import re

import olefile
import pefile
from oletools.olevba import VBA_Parser, VBA_Scanner
from app.core.static_analyzer.utils.vba_deobfuscator import deobfuscate_vba_strings
from app.core.static_analyzer.utils.feature_utils import ALLOWLIST_STD_DOMAINS
from app.core.static_analyzer.utils.feature_utils import extract_urls, derive_domains, filter_nonstandard_domains


from app.core.static_analyzer.utils.feature_utils import (
    calculate_entropy,
    dedup_sort_strings,
    extract_urls,
    extract_ips,
    extract_filepaths,
    extract_registry_keys,
    extract_obfuscated_strings,
    detect_obfuscation_ops,
    detect_amsi_bypass,
    find_winapi_calls,
    find_com_progids,
    filter_nonstandard_domains,
    domain_from_url,
    extract_user_agents,
    guess_pkcs7_sign_info,
    count_vba_references,
)

MAX_STREAM_SIZE = 100 * 1024 * 1024  
MAX_TEXT_BLOB_SIZE = 10 * 1024 * 1024  
MAX_PE_SIZE = 50 * 1024 * 1024  
MAX_PROCESSING_TIME = 300  

_DDE_TOKEN_RE = re.compile(r"(?i)\bDDE(?:AUTO|EXEC)?\b")
_DDE_FORMULA_RE = re.compile(r"(?i)(?:^|[=\s])DDE\s*\(", re.M)

def _safe_text(b: bytes) -> str:
    if not b or len(b) > MAX_TEXT_BLOB_SIZE:
        return ""
    b = b[:MAX_TEXT_BLOB_SIZE]
    
    for enc in ("utf-8", "utf-16-le", "utf-16-be", "cp1252", "latin-1"):
        try:
            return b.decode(enc, errors="ignore")
        except Exception:
            continue
    return ""

def _read_all_safe(fp, max_size: int = MAX_STREAM_SIZE) -> bytes:
    try:
        chunks = []
        total_size = 0
        while True:
            chunk = fp.read(8192)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > max_size:
                break
            chunks.append(chunk)
        return b''.join(chunks)
    except Exception:
        return b""

def _is_ole_available() -> bool:
    return olefile is not None

class OLEParser:
    def __init__(self, file_input: str | Path | io.BytesIO):
        if isinstance(file_input, (str, Path)):
            self.file_path = str(file_input)
            self.file_data = None
            try:
                file_size = os.path.getsize(self.file_path)
                if file_size > MAX_STREAM_SIZE * 2:  
                    print(f"Warning: Large file detected ({file_size} bytes). Processing may be limited.")
            except Exception:
                pass
        elif isinstance(file_input, io.BytesIO):
            self.file_path = None
            self.file_data = file_input
        else:
            raise ValueError("file_input must be a file path (str/Path) or BytesIO object")
            
        self.features: Dict[str, Any] = {
            "structure": {},
            "macros": {},
            "strings": {},
            "apis": {},
            "obfuscation": {},
            "network_indicators": {},
            "security_indicators": {},
        }
        self._text_blobs: List[str] = []
        self._streams_cache: List[Dict[str, Any]] = []

    def parse(self) -> Dict[str, Any]:
        if not _is_ole_available():
            self.features["structure"] = {
                "format": "ole",
                "ole_header_valid": False,
                "streams_count": 0,
                "storages_count": 0,
                "dir_entries_count": 0,
                "metadata_anomalies": ["olefile_not_available"],
            }
            self._finalize_blocks()
            return self.features

        try:
            file_input = self.file_data if self.file_data is not None else str(self.file_path)
            with olefile.OleFileIO(file_input) as ole:
                self._extract_structure(ole)
                self._extract_directory_and_text(ole)
                self._extract_link_objects(ole)
                self._extract_embedded_objects(ole)
                self._extract_macros_safe(ole)
                self._extract_strings_block()
                self._extract_apis_block()
                self._extract_obfuscation_block(ole)
                self._extract_network_block()
                self._extract_security_block(ole)
        except Exception as e:
            self.features["structure"]["parse_error"] = str(e)
            self.features["structure"]["format"] = "ole"

        self._finalize_blocks()
        return self.features

    def _extract_structure(self, ole: "olefile.OleFileIO") -> None:
        st = self.features["structure"] = self.features.get("structure", {})
        st["format"] = "ole"
        
        try:
            st["ole_header_valid"] = True
            hdr = getattr(ole, "header", None)
            if hdr:
                st["sector_size"] = getattr(hdr, "secsize", None) or 512
                st["mini_sector_size"] = getattr(hdr, "minisector_size", None) or 64
                st["fat_sectors"] = getattr(hdr, "num_fat_sectors", None)
                st["mini_fat_sectors"] = getattr(hdr, "num_minifat_sectors", None)
        except Exception:
            st["ole_header_valid"] = True

        try:
            dirents = ole.direntries or []
            st["dir_entries_count"] = len([d for d in dirents if d is not None])
        except Exception:
            st["dir_entries_count"] = 0

        try:
            items = ole.listdir(streams=True, storages=True)
            storages = streams = 0
            for path in items:
                try:
                    de = ole._find(path)
                    if de and hasattr(de, "entry_type"):
                        if de.entry_type == olefile.STGTY_STORAGE:
                            storages += 1
                        elif de.entry_type == olefile.STGTY_STREAM:
                            streams += 1
                except Exception:
                    continue
            st["storages_count"] = storages
            st["streams_count"] = streams
        except Exception:
            st["storages_count"] = st.get("storages_count", 0)
            st["streams_count"] = st.get("streams_count", 0)

        self._extract_metadata_safe(ole, st)

    def _extract_metadata_safe(self, ole: "olefile.OleFileIO", st: Dict[str, Any]) -> None:
        meta = {}
        try:
            if ole.exists("\x05SummaryInformation"):
                with ole.openstream("\x05SummaryInformation") as fp:
                    meta_data = _read_all_safe(fp, 64*1024)  
                    meta_txt = _safe_text(meta_data)
                    
                    for line in meta_txt.splitlines()[:50]:  
                        if "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip().lower()
                            v = v.strip()[:200]  
                            if k == "author": meta["author"] = v
                            elif k == "lastsavedby": meta["last_saved_by"] = v
                            elif k == "revisionnumber": meta["revision_number"] = v
                            elif k == "createtime": meta["creation_time"] = v
                            elif k == "modifytime": meta["modification_time"] = v
                            elif k == "applicationname": meta["application"] = v
                            elif k == "template": meta["template"] = v
        except Exception:
            pass
        
        st["metadata_summary"] = meta
        
        anomalies = []
        if not meta.get("author"):
            anomalies.append("missing_author")
        if not meta.get("application"):
            anomalies.append("missing_application")
        st["metadata_anomalies"] = anomalies

    def _extract_directory_and_text(self, ole: "olefile.OleFileIO") -> None:
        directory: List[Dict[str, Any]] = []
        text_blobs: List[str] = []

        try:
            items = ole.listdir(streams=True, storages=True)
        except Exception:
            items = []

        processed_streams = 0
        for path in items:
            if processed_streams > 100:
                break

            apath = "/" + "/".join(path)
            node: Dict[str, Any] = {
                "path": apath,
                "name": path[-1] if path else "",
            }

            try:
                de = ole._find(path)
            except Exception:
                de = None

            entry_type = None
            if de is not None:
                try:
                    entry_type = de.entry_type
                except Exception:
                    entry_type = None

            if entry_type == olefile.STGTY_STORAGE:
                node["obj_type"] = "storage"
                directory.append(node)
                continue

            if entry_type == olefile.STGTY_STREAM:
                node["obj_type"] = "stream"
                processed_streams += 1
                try:
                    with ole.openstream(path) as fp:
                        data = _read_all_safe(fp)
                    node["size_bytes"] = len(data)
                    node["entropy"] = float(calculate_entropy(data))

                    sample = data[:256*1024]
                    text_content = _safe_text(sample)

                    if not text_content or len(text_content) < 10:
                        cleaned = re.sub(b'[\x00-\x08\x0b\x0c\x0e-\x1f]+', b' ', sample)
                        try:
                            text_content = cleaned.decode("latin-1", errors="ignore")
                        except Exception:
                            text_content = ""

                    if text_content and len(text_content) > 10:
                        text_blobs.append(text_content[:MAX_TEXT_BLOB_SIZE])
                except Exception:
                    pass

                directory.append(node)
                continue

            continue

        self.features["structure"]["directory"] = directory
        self._streams_cache = directory
        self._text_blobs = text_blobs

    def _extract_macros_safe(self, ole: "olefile.OleFileIO") -> None:
        macros = self.features.get("macros", {}) or {}
        vba_texts: List[str] = []
        recovered_urls: List[str] = []
    
        suspicious_calls = 0
        autoexec_triggers: set[str] = set()
        dde_found = False
        modules: List[Dict[str, Any]] = []
    
        vba = None
        vba_file_path = self.file_path
        temp_path = None
        
        try:
            if self.file_data is not None:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".doc") as tmp:
                    self.file_data.seek(0)
                    tmp.write(self.file_data.read())
                    temp_path = tmp.name
                    vba_file_path = temp_path
            
            try:
                vba = VBA_Parser(str(vba_file_path))
                if vba.detect_vba_macros():
                    macros["has_vba"] = True
                    for (filename, stream_path, vba_filename, vba_code) in vba.extract_macros():
                        name = (vba_filename or filename or "Module").strip()
                        line_count = vba_code.count("\n") + 1
                        obf = any(k in vba_code.lower() for k in ("chr(", "chrw(", "xor", "base64", "split(", "replace(", "mid("))
                        modules.append({"name": name, "line_count": int(line_count), "obfuscated": bool(obf)})
                        vba_texts.append(vba_code)
        
                        scanner = VBA_Scanner(vba_code)
                        for kw_type, keyword, desc in scanner.scan(include_decoded_strings=True):
                            k = (keyword or "").lower()
                            if kw_type == "AutoExec":
                                autoexec_triggers.add(k)
                            elif kw_type == "Suspicious":
                                suspicious_calls += 1
                            elif kw_type == "IOC" and "dde" in k:
                                dde_found = True
            except Exception as inner_e:
                macros.setdefault("errors", []).append(f"vba_parser_failed: {inner_e}")
        except Exception as e:
            macros.setdefault("errors", []).append(f"olevba_failed: {e}")
        finally:
            if vba:
                try:
                    vba.close()
                except Exception:
                    pass
            if temp_path:
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
                
        try:
            if vba_texts:
                recovered_strings = deobfuscate_vba_strings(vba_texts)
                if recovered_strings:
                    joined = "\n".join(recovered_strings)
                    recovered_urls = extract_urls(joined)
                    if not recovered_urls:
                        macros.setdefault("errors", []).append(
                            f"deobf_no_urls: recovered_strings={len(recovered_strings)}"
                        )
                else:
                    macros.setdefault("errors", []).append("deobf_empty: no strings recovered")
        except Exception as e:
            macros.setdefault("errors", []).append(f"deobfuscate_failed: {e}")
    
        macros["modules"] = modules
        macros["autoexec_triggers"] = sorted(list(autoexec_triggers))[:20]
        macros["suspicious_api_calls_count"] = int(min(suspicious_calls, 10000))
        macros["dde"] = {"has_dde": bool(dde_found)}
        macros["vba_project_properties"] = macros.get("vba_project_properties") or {"signed": False, "protected": False, "references_count": 0}
        self.features["macros"] = macros
    
        strings = self.features.get("strings", {}) or {}
        base_urls = strings.get("urls", []) or []

        strings_from_vba = []
        try:
            if vba_texts and 'recovered_strings' in locals() and recovered_strings:
                strings_from_vba = recovered_strings[:500]
        except Exception:
            pass
        
        joined_vba = "\n".join(strings_from_vba)
        recovered_urls = sorted(list({*recovered_urls, *extract_urls(joined_vba)}))
        strings["urls"] = sorted(list({*base_urls, *recovered_urls}))

        obs_base = strings.get("obfuscated_strings", []) or []
        strings["obfuscated_strings"] = dedup_sort_strings(list(obs_base) + strings_from_vba)[:200]

        self.features["strings"] = strings

        net = self.features.get("network_indicators", {}) or {}
        doms = derive_domains(strings.get("urls", []))
        net["urls"] = strings.get("urls", [])
        net["domains"] = filter_nonstandard_domains(doms)
        net["user_agents"] = net.get("user_agents", []) or []
        self.features["network_indicators"] = net
        
        try:
            if vba_texts:
                self._text_blobs.append("\n".join(vba_texts))
        except Exception:
            pass

    def _extract_embedded_objects(self, ole: "olefile.OleFileIO") -> None:
        embedded: List[Dict[str, Any]] = []

        object_count = 0
        for node in self._streams_cache:
            if object_count >= 20:  
                break
                
            if node.get("obj_type") != "stream":
                continue
                
            path = node.get("path") or ""
            if not path.lower().startswith("/objectpool/") or not path.lower().endswith("/contents"):
                continue
                
            parts = [p for p in path.strip("/").split("/") if p]
            try:
                with ole.openstream(parts) as fp:
                    data = _read_all_safe(fp, MAX_PE_SIZE)  
            except Exception:
                continue

            if not data:
                continue
                
            object_count += 1
            entry: Dict[str, Any] = {
                "type": "bin",
                "name": (parts[-2] if len(parts) >= 2 else "Object")[:50],
                "size_bytes": len(data),
                "hash": {"sha256": hashlib.sha256(data).hexdigest()},
                "pe_summary": {},
            }

            pe_summary = self._analyze_pe_safe(data)
            entry["pe_summary"] = pe_summary
            embedded.append(entry)

        self.features["structure"]["embedded_objects"] = embedded

    def _analyze_pe_safe(self, data: bytes) -> Dict[str, Any]:
        pe_summary: Dict[str, Any] = {
            "section_count": 0,
            "import_count": 0,
            "section_entropy_mean": float(calculate_entropy(data))
        }
        
        if pefile is not None and len(data) <= MAX_PE_SIZE:
            try:
                pe = pefile.PE(data=data, fast_load=True)
                pe.parse_data_directories(directories=[])
                
                sec_cnt = len(getattr(pe, "sections", []) or [])
                pe_summary["section_count"] = min(int(sec_cnt), 100)  
                
                try:
                    if hasattr(pe, "DIRECTORY_ENTRY_IMPORT") and pe.DIRECTORY_ENTRY_IMPORT:
                        pe_summary["import_count"] = min(len(pe.DIRECTORY_ENTRY_IMPORT), 1000)
                except Exception:
                    pass
                
                ent_vals: List[float] = []
                try:
                    sections = (pe.sections or [])[:20]  
                    for s in sections:
                        section_data = bytes(s.get_data() or b"")
                        if len(section_data) <= 10*1024*1024:  
                            ent_vals.append(calculate_entropy(section_data))
                except Exception:
                    pass
                    
                if ent_vals:
                    pe_summary["section_entropy_mean"] = float(sum(ent_vals) / len(ent_vals))
                    
            except Exception:
                pass

        return pe_summary

    def _extract_link_objects(self, ole: "olefile.OleFileIO") -> None:
        joined = "\n".join(self._text_blobs)[:MAX_TEXT_BLOB_SIZE]  
        links: List[Dict[str, Any]] = []

        if _DDE_TOKEN_RE.search(joined) or _DDE_FORMULA_RE.search(joined):
            links.append({
                "link_type": "DDE_LINK",
                "target": "heuristic_detected",
                "source_stream": "WordDocument"
            })

        for node in self._streams_cache:
            p = (node.get("path") or "").lower()
            if "/ole" in p and "link" in p:
                links.append({
                    "link_type": "OLE_LINK",
                    "target": "heuristic_detected",
                    "source_stream": (node.get("name") or "unknown")[:50]
                })
                break

        self.features["structure"]["link_objects"] = links

    def _extract_strings_block(self) -> None:
        joined = "\n".join(self._text_blobs)[:MAX_TEXT_BLOB_SIZE]

        backup_txt = ""
        try:
            if self.file_data is not None:
                self.file_data.seek(0)
                whole = self.file_data.read(5 * 1024 * 1024)
            else:
                with open(self.file_path, "rb") as f:
                    whole = f.read(5 * 1024 * 1024)
            cleaned = re.sub(b'[\x00-\x08\x0b\x0c\x0e-\x1f]+', b' ', whole)
            backup_txt = cleaned.decode("latin-1", errors="ignore")
        except Exception:
            pass
        
        mix = (joined + "\n" + backup_txt)[:MAX_TEXT_BLOB_SIZE]
    
        self.features["strings"] = {
            "urls": dedup_sort_strings(extract_urls(mix))[:200],
            "ips": dedup_sort_strings(extract_ips(mix))[:100],
            "filepaths": dedup_sort_strings(extract_filepaths(mix))[:200],
            "registry_keys": dedup_sort_strings(extract_registry_keys(mix))[:200],
            "obfuscated_strings": dedup_sort_strings(extract_obfuscated_strings(mix))[:100],
        }

    def _extract_apis_block(self) -> None:
        joined = "\n".join(self._text_blobs)[:MAX_TEXT_BLOB_SIZE]
        vba_and_deobf = (self.features.get("strings", {}).get("obfuscated_strings", []) or [])[:500]
        mix = (joined + "\n" + "\n".join(vba_and_deobf))[:MAX_TEXT_BLOB_SIZE]

        winapi_calls = find_winapi_calls(mix)
        com_progids = find_com_progids(mix)

        self.features["apis"] = {
            "winapi_calls": winapi_calls[:200],
            "com_progids": com_progids[:100],
        }

    def _extract_obfuscation_block(self, ole: "olefile.OleFileIO") -> None:
        try:
            if self.file_data is not None:
                self.file_data.seek(0)
                whole = self.file_data.read(10*1024*1024) 
            else:
                with open(self.file_path, "rb") as f:
                    whole = f.read(10*1024*1024)  
            overall = float(calculate_entropy(whole))
        except Exception:
            overall = 0.0

        entropy_by_part = {}
        count = 0
        for path in ole.listdir(streams=True, storages=False):
            if count >= 200:
                break
            try:
                with ole.openstream(path) as fp:
                    data = _read_all_safe(fp, 2*1024*1024)  
                part_name = "/" + "/".join(path)
                entropy_by_part[part_name[:200]] = float(calculate_entropy(data))
                count += 1
            except Exception:
                continue

        text_sample = "\n".join(self._text_blobs)[:MAX_TEXT_BLOB_SIZE]
        suspicious_ops = detect_obfuscation_ops([text_sample])
        amsi_bypass = detect_amsi_bypass([text_sample])

        self.features["obfuscation"] = {
            "entropy": {
                "overall": overall, 
                "max": max(entropy_by_part.values()) if entropy_by_part else overall
            },
            "entropy_by_part": entropy_by_part,
            "suspicious_ops": suspicious_ops[:20],  
            "amsi_bypass": bool(amsi_bypass),
        }

    def _extract_network_block(self) -> None:
        urls = self.features.get("strings", {}).get("urls", []) or []
    
        filtered_urls = []
        for u in urls:
            d = domain_from_url(u)
            if not d:
                continue
            skip = False
            for allow in ALLOWLIST_STD_DOMAINS:
                if d == allow or d.endswith("." + allow):
                    skip = True
                    break
            if not skip:
                filtered_urls.append(u)

        domains = sorted({d for d in (domain_from_url(u) for u in filtered_urls) if d})
        domains = filter_nonstandard_domains(domains)

        joined = "\n".join(self._text_blobs)[:MAX_TEXT_BLOB_SIZE]

        self.features["network_indicators"] = {
            "urls": filtered_urls[:100],
            "domains": domains[:100],
            "user_agents": dedup_sort_strings(extract_user_agents(joined))[:20],
        }

    def _extract_security_block(self, ole: "olefile.OleFileIO") -> None:
        motw = False
        if self.file_path:
            try:
                with open(self.file_path + ":Zone.Identifier", "r", encoding="utf-8", errors="ignore") as f:
                    if "ZoneId=" in f.read(1024):  
                        motw = True
            except Exception:
                pass

        signed = False
        publisher: Optional[str] = None
        timestamp: Optional[str] = None

        signature_streams = [
            node for node in self._streams_cache[:20]  
            if node.get("obj_type") == "stream" and 
               any(tok in (node.get("path") or "").lower() 
                   for tok in ("signature", "mssignature", "digitalsignature"))
        ]

        for node in signature_streams:
            try:
                parts = [s for s in (node.get("path") or "").strip("/").split("/") if s]
                with ole.openstream(parts) as fp:
                    sig_bytes = _read_all_safe(fp, 1024*1024) 
                pub, ts = guess_pkcs7_sign_info(sig_bytes)
                if pub or ts:
                    signed = True
                    publisher = (publisher or pub or "")[:100]  
                    timestamp = timestamp or ts
                    break
            except Exception:
                continue

        self.features["security_indicators"] = {
            "motw_present": bool(motw),
            "macro_security_hint": "present" if self.features.get("macros", {}).get("has_vba") else "absent",
            "trusted_location_hint": False,
            "digital_signature": {
                "signed": bool(signed),
                "publisher": publisher or "",
                "timestamp": timestamp or "",
            },
            "is_encrypted_container": False,
        }

    def _finalize_blocks(self) -> None:
        for sec in ("strings", "network_indicators"):
            v = self.features.get(sec)
            if isinstance(v, dict):
                for k in v:
                    if isinstance(v[k], list):
                        v[k] = dedup_sort_strings(v[k])