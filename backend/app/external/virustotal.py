import json
import requests
import os
from datetime import datetime, UTC
from typing import Dict, Any, Optional
from app.config import get_settings

class VirusTotalAPI:
    def __init__(self, api_key: str = None):
        settings = get_settings()
        self.api_key = api_key or getattr(settings, 'VT_API_KEY', None)
        self.base_url = "https://www.virustotal.com/api/v3"
        
        if not self.api_key:
            raise ValueError("VirusTotal API key is required")
    
    def query_file_report(self, sha256: str) -> Dict[str, Any]:
        url = f"{self.base_url}/files/{sha256}"
        headers = {"x-apikey": self.api_key}
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return {"error": "file_not_found", "message": "File not found in VirusTotal database"}
            elif response.status_code == 429:
                return {"error": "rate_limit", "message": "API rate limit exceeded"}
            else:
                return {"error": "api_error", "message": f"API request failed: {response.status_code}"}
                
        except requests.RequestException as e:
            return {"error": "network_error", "message": f"Network error: {str(e)}"}
    
    def parse_vt_response(self, vt_response: Dict[str, Any]) -> Dict[str, Any]:
        if "error" in vt_response:
            return {
                "available": False,
                "error": vt_response["error"],
                "message": vt_response["message"]
            }
        
        try:
            attrs = vt_response["data"]["attributes"]
            stats = attrs.get("last_analysis_stats", {})
            results = attrs.get("last_analysis_results", {})
            
            total_engines = len(results)  
            malicious = stats.get("malicious", 0)
            suspicious = stats.get("suspicious", 0)
            
            print(f"🔍 VT 통계: malicious={malicious}, suspicious={suspicious}, total_engines={total_engines}")
            print(f"🔍 VT 실제 엔진 수: {len(results)}")
            print(f"🔍 VT stats 합계: {sum(stats.values())}")
            
            threat_classification = attrs.get("popular_threat_classification", {})
            suggested_label = threat_classification.get("suggested_threat_label", "")
            
            major_vendors = ["Microsoft", "Symantec", "McAfee", "Kaspersky", "BitDefender",
                           "TrendMicro", "FireEye", "Avast", "AVG", "Sophos", "ESET-NOD32", "AhnLab-V3"]
            
            vendor_results = {}
            for vendor in major_vendors:
                if vendor in results:
                    result = results[vendor]
                    vendor_results[vendor] = {
                        "category": result.get("category"),
                        "result": result.get("result"),
                        "detected": result.get("category") in ["malicious", "suspicious"]
                    }
            
            return {
                "available": True,
                "scan_summary": {
                    "malicious": malicious,
                    "suspicious": suspicious,
                    "total": total_engines,
                    "detection_rate": round((malicious + suspicious) / total_engines * 100, 2) if total_engines > 0 else 0
                },
                "threat_info": {
                    "suggested_label": suggested_label,
                    "tags": attrs.get("tags", []),
                    "names": attrs.get("names", [])[:5]  
                },
                "timestamps": {
                    "first_submission": self._parse_timestamp(attrs.get("first_submission_date")),
                    "last_analysis": self._parse_timestamp(attrs.get("last_analysis_date")),
                    "first_seen_itw": self._parse_timestamp(attrs.get("first_seen_itw_date"))
                },
                "vendor_results": vendor_results,
                "reputation": {
                    "score": attrs.get("reputation", 0),
                    "votes": attrs.get("total_votes", {})
                },
                "file_info": {
                    "size": attrs.get("size"),
                    "type_tags": attrs.get("type_tags", []),
                    "times_submitted": attrs.get("times_submitted", 0)
                },
                "permalink": f"https://www.virustotal.com/gui/file/{attrs['sha256']}"
            }
            
        except (KeyError, TypeError) as e:
            return {
                "available": False,
                "error": "parse_error",
                "message": f"Failed to parse VirusTotal response: {str(e)}"
            }
    
    def _parse_timestamp(self, timestamp: Optional[int]) -> Optional[str]:
        if timestamp is None:
            return None
        try:
            return datetime.fromtimestamp(timestamp, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        except (ValueError, OSError):
            return str(timestamp)
    
    def get_file_analysis(self, sha256: str) -> Dict[str, Any]:
        print(f"🔍 VirusTotal 조회 중: {sha256[:16]}...")
        
        raw_response = self.query_file_report(sha256)
        parsed_result = self.parse_vt_response(raw_response)
        
        if parsed_result.get("available"):
            detection_rate = parsed_result["scan_summary"]["detection_rate"]
            malicious_count = parsed_result["scan_summary"]["malicious"]
            print(f"✅ VT 결과: {malicious_count}개 엔진 탐지 ({detection_rate}%)")
        else:
            error_type = parsed_result.get("error", "unknown")
            print(f"❌ VT 조회 실패: {error_type}")
        
        return parsed_result

_vt_client = None

def get_virustotal_client() -> VirusTotalAPI:
    """VirusTotal 클라이언트 인스턴스 반환"""
    global _vt_client
    if _vt_client is None:
        _vt_client = VirusTotalAPI()
    return _vt_client
