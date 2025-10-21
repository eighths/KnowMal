import json
import requests
import os
import logging
from datetime import datetime, UTC
from typing import Dict, Any, Optional
from app.config import get_settings

logger = logging.getLogger(__name__)

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
        
        logger.info(f"[VT_API] VirusTotal API 호출 시작 - SHA256: {sha256[:16]}...")
        logger.info(f"[VT_API] 요청 URL: {url}")
        logger.info(f"[VT_API] API 키: {self.api_key[:8]}..." if self.api_key else "[VT_API] API 키: None")
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            
            logger.info(f"[VT_API] 응답 상태 코드: {response.status_code}")
            logger.info(f"[VT_API] 응답 헤더: {dict(response.headers)}")
            
            if response.status_code == 200:
                logger.info(f"[VT_API] API 호출 성공 - SHA256: {sha256[:16]}...")
                return response.json()
            elif response.status_code == 404:
                logger.warning(f"[VT_API] 파일을 찾을 수 없음 - SHA256: {sha256[:16]}...")
                return {"error": "file_not_found", "message": "File not found in VirusTotal database"}
            elif response.status_code == 429:
                logger.error(f"[VT_API] API 호출 제한 초과 - SHA256: {sha256[:16]}...")
                return {"error": "rate_limit", "message": "API rate limit exceeded"}
            else:
                logger.error(f"[VT_API] API 요청 실패 - 상태 코드: {response.status_code}, SHA256: {sha256[:16]}...")
                logger.error(f"[VT_API] 응답 내용: {response.text[:500]}...")
                return {"error": "api_error", "message": f"API request failed: {response.status_code}"}
                
        except requests.RequestException as e:
            logger.error(f"[VT_API] 네트워크 오류 - SHA256: {sha256[:16]}..., 오류: {str(e)}")
            return {"error": "network_error", "message": f"Network error: {str(e)}"}
    
    def parse_vt_response(self, vt_response: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"[VT_API] 응답 파싱 시작")
        
        if "error" in vt_response:
            logger.warning(f"[VT_API] 응답에 오류 포함: {vt_response['error']} - {vt_response['message']}")
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
            
            result = {
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
            
            logger.info(f"[VT_API] 파싱 완료 - 악성: {malicious}, 의심: {suspicious}, 전체: {total_engines}")
            logger.info(f"[VT_API] 탐지율: {result['scan_summary']['detection_rate']}%")
            logger.info(f"[VT_API] 제안 라벨: {suggested_label}")
            
            return result
            
        except (KeyError, TypeError) as e:
            logger.error(f"[VT_API] 응답 파싱 실패: {str(e)}")
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
        logger.info(f"[VT_API] get_file_analysis 호출됨 - SHA256: {sha256[:16]}...")
        
        raw_response = self.query_file_report(sha256)
        parsed_result = self.parse_vt_response(raw_response)
        
        if parsed_result.get("available"):
            detection_rate = parsed_result["scan_summary"]["detection_rate"]
            malicious_count = parsed_result["scan_summary"]["malicious"]
            logger.info(f"[VT_API] 최종 분석 결과 - 탐지율: {detection_rate}%, 악성: {malicious_count}개")
        else:
            logger.warning(f"[VT_API] 분석 결과 사용 불가 - 오류: {parsed_result.get('error', 'unknown')}")
        
        return parsed_result

_vt_client = None

def get_virustotal_client() -> VirusTotalAPI:
    global _vt_client
    if _vt_client is None:
        logger.info("[VT_API] VirusTotal 클라이언트 초기화 중...")
        try:
            _vt_client = VirusTotalAPI()
            logger.info("[VT_API] VirusTotal 클라이언트 초기화 완료")
        except Exception as e:
            logger.error(f"[VT_API] VirusTotal 클라이언트 초기화 실패: {e}")
            raise
    return _vt_client
