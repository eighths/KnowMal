from __future__ import annotations

import base64
import os
import time
from datetime import datetime, timedelta
from typing import Optional

import requests
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.db.models import OAuthAccount
from app.core.tokens import try_verify, TokenError

router = APIRouter(prefix="/gmail", tags=["gmail"])

GMAIL_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"
HTTP_TIMEOUT = 15


class ScanRequest(BaseModel):
    message_id: str
    filename: Optional[str] = None


def get_google_sub_from_auth(authorization: Optional[str], ext_id: Optional[str] = None) -> str:

    if ext_id:
        return ext_id

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    try:
        return try_verify(token, max_age=3600)
    except TokenError:
        raise HTTPException(401, "invalid/expired token")

def ensure_fresh_access_token(db: Session, rec: OAuthAccount) -> str:
    now = datetime.utcnow()
    if rec.access_token and rec.expires_at and rec.expires_at > now + timedelta(seconds=120):
        return rec.access_token

    if not rec.refresh_token:
        raise HTTPException(401, "no refresh_token stored")

    data = {
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
        "refresh_token": rec.refresh_token,
        "grant_type": "refresh_token",
    }
    if not data["client_id"] or not data["client_secret"]:
        raise HTTPException(500, "server misconfigured: missing google client env")

    r = requests.post("https://oauth2.googleapis.com/token", data=data, timeout=HTTP_TIMEOUT)
    if r.status_code != 200:
        raise HTTPException(401, f"refresh failed: {r.text}")
    js = r.json()
    access_token = js.get("access_token")
    expires_in = js.get("expires_in", 3600)
    token_type = js.get("token_type")

    if not access_token:
        raise HTTPException(401, "no access_token from refresh")

    rec.access_token = access_token
    rec.token_type = token_type or rec.token_type
    rec.expires_at = datetime.utcnow() + timedelta(seconds=int(expires_in))
    db.add(rec)
    db.commit()
    return access_token

def _find_attachment_part(payload: dict, filename: Optional[str]) -> Optional[dict]:
    stack = [payload]
    while stack:
        node = stack.pop()
        if "parts" in node and isinstance(node["parts"], list):
            stack.extend(node["parts"])
        fname = node.get("filename") or ""
        body = node.get("body") or {}
        att_id = body.get("attachmentId")
        if att_id and (not filename or filename == fname):
            return node
    return None

@router.post("/scan")
def scan(
    req: ScanRequest,
    authorization: Optional[str] = Header(None),
    ext_id: Optional[str] = Header(None, alias="X-KM-Ext-Id"),
    account_email: Optional[str] = Header(None, alias="X-KM-Account-Email"),
    db: Session = Depends(get_db),
):
    google_sub = get_google_sub_from_auth(authorization, ext_id)
    user_id_key = f"{google_sub}:{account_email}" if account_email else google_sub
    print(f"[GMAIL][scan] ext_id={ext_id} account_email={account_email} -> user_id_key={user_id_key}")
    print(f"[GMAIL][scan] msg_id={req.message_id} filename={req.filename}")

    rec = None
    if account_email:
        rec = (
            db.query(OAuthAccount)
            .filter(OAuthAccount.provider == "gmail", OAuthAccount.email == account_email)
            .order_by(OAuthAccount.updated_at.desc())
            .first()
        )
    if rec is None:
        rec = (
            db.query(OAuthAccount)
            .filter(OAuthAccount.provider == "gmail", OAuthAccount.user_id == user_id_key)
            .first()
        )
    if rec is None and account_email:
        rec = (
            db.query(OAuthAccount)
            .filter(OAuthAccount.provider == "gmail", OAuthAccount.user_id == google_sub)
            .order_by(OAuthAccount.updated_at.desc())
            .first()
        )
    if not rec:
        print(f"[GMAIL][scan] OAuthAccount not found for user_id={user_id_key}")
        raise HTTPException(401, "not linked")

    try:
        if rec.scope:
            granted_scopes = set((rec.scope or "").split())
            if "https://www.googleapis.com/auth/gmail.readonly" not in granted_scopes:
                print("[GMAIL][scan] insufficient scopes on record; forcing reauth")
                raise HTTPException(401, "insufficient_scopes")
    except HTTPException:
        raise
    except Exception:
        pass

    access_token = ensure_fresh_access_token(db, rec)

    try:
        if account_email:
            ui = requests.get(
                "https://openidconnect.googleapis.com/v1/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=HTTP_TIMEOUT,
            )
            if ui.status_code == 200:
                token_email = (ui.json() or {}).get("email")
                if token_email and token_email.lower() != account_email.lower():
                    print(
                        f"[GMAIL][scan] access_token email mismatch: token_email={token_email} account_email={account_email}"
                    )
                    raise HTTPException(401, "not linked")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[GMAIL][scan] userinfo check failed: {e}")

    headers = {"Authorization": f"Bearer {access_token}"}
    r = requests.get(f"{GMAIL_BASE}/messages/{req.message_id}?format=full", headers=headers, timeout=HTTP_TIMEOUT)
    if r.status_code == 403:
        raise HTTPException(401, "insufficient_scopes")
    if r.status_code != 200:
        raise HTTPException(400, f"gmail message get failed: {r.text}")
    msg = r.json()
    payload = msg.get("payload") or {}

    part = _find_attachment_part(payload, req.filename)
    if not part:
        part = _find_attachment_part(payload, None)
        if not part:
            raise HTTPException(404, "attachment not found")

    att_id = part["body"]["attachmentId"]
    fname = part.get("filename") or (req.filename or "attachment.bin")

    r2 = requests.get(f"{GMAIL_BASE}/messages/{req.message_id}/attachments/{att_id}", headers=headers, timeout=HTTP_TIMEOUT)
    if r2.status_code == 403:
        raise HTTPException(401, "insufficient_scopes")
    if r2.status_code != 200:
        raise HTTPException(400, f"gmail attachment get failed: {r2.text}")
    data64 = r2.json().get("data")
    if not data64:
        raise HTTPException(400, "attachment has no data")
    file_bytes = base64.urlsafe_b64decode(data64 + "===")

    report_url = None
    try:
        from app.db.models import FileRecord
        from app.cache.redis_client import get_redis
        from app.cache.keys import file_data_key
        from app.api.share import create_share
        from app.services.ai_model_service import get_ai_model_service
        from app.external.virustotal import get_virustotal_client
        import hashlib
        import json
        import time

        sha256 = hashlib.sha256(file_bytes).hexdigest()
        mime_type = "application/octet-stream" 

        try:
            from app.core.static_analysis import sniff_mime
            mime_type = sniff_mime(file_bytes=file_bytes)
        except:
            pass

        file_record = FileRecord(
            filename=fname,
            mime_type=mime_type,
            size_bytes=len(file_bytes),
            content_excerpt="", 
            source="gmail",
            source_url=f"gmail://{req.message_id}",
            page_url="",
            sha256=sha256,
        )
        db.add(file_record)
        db.commit()
        db.refresh(file_record)

        from app.core.static_analysis import analyze_bytes
        analysis_result = analyze_bytes(
            file_bytes, 
            filename=fname,
            ttl_sec=86400, 
            use_cache=False,
            include_virustotal=True
        )

        ai_prediction = None
        try:
            ai_model_service = get_ai_model_service()
            if ai_model_service.model_loaded:
                ai_prediction = ai_model_service.predict_malware_type(analysis_result)
                print(f"Gmail AI 예측 완료: {fname}")
        except Exception as e:
            print(f"Gmail AI 예측 실패: {e}")

        virustotal_result = None
        try:
            print(f"[GMAIL] VirusTotal 조회 시작: {fname} (SHA256: {sha256[:16]}...)")
            vt_client = get_virustotal_client()
            virustotal_result = vt_client.get_file_analysis(sha256)
            if virustotal_result and virustotal_result.get("available"):
                print(f"[GMAIL] VirusTotal 조회 완료: {fname} - 탐지율: {virustotal_result.get('scan_summary', {}).get('detection_rate', 0)}%")
            else:
                print(f"[GMAIL] VirusTotal 조회 결과 없음: {fname} - {virustotal_result.get('error', 'unknown')}")
        except Exception as e:
            print(f"[GMAIL] VirusTotal 조회 실패: {fname} - {e}")

        redis_client = get_redis()
        file_cache_key = file_data_key(sha256)
        cache_data = {
            "filename": fname,
            "mime_type": mime_type,
            "size_bytes": len(file_bytes),
            "sha256": sha256,
            "file_id": file_record.id,
            "analysis_report": analysis_result,
            "ai_prediction": ai_prediction,
            "virustotal": virustotal_result
        }
        redis_client.setex(
            file_cache_key, 
            86400, 
            json.dumps(cache_data, ensure_ascii=False)
        )

        from app.api.share import create_share
        from fastapi import Request

        class MockRequest:
            def __init__(self):
                self.base_url = "http://localhost:8000"
        
        share_result = create_share(
            request=MockRequest(),
            file_id=file_record.id,
            db=db
        )
        report_url = share_result["report_url"]
        
    except Exception as e:
        print(f"Gmail 상세 분석 실패: {e}")
        try:
            from app.core.static_analysis import analyze_bytes
            result = analyze_bytes(file_bytes, filename=fname)
            if isinstance(result, dict):
                report_url = result.get("report_url")
        except Exception:
            pass
        
        if not report_url:
            report_url = f"http://localhost:8000/r/gmail_{req.message_id}"

    status = "unknown"
    try:
        if analysis_result and analysis_result.get('embedded_files'):
            embedded_files = analysis_result['embedded_files']
            print(f"[GMAIL] 압축 파일 내부 파일 검사: {len(embedded_files)}개")
            
            for item in embedded_files:
                item_report = item.get('report', {})
                item_vt = item_report.get('virustotal', {})
                if item_vt.get('available') and item_vt.get('scan_summary', {}).get('malicious', 0) > 0:
                    status = "danger"
                    print(f"[GMAIL] 내부 파일 VT 탐지로 위험 판정")
                    break
                
                item_ai = item_report.get('ai_prediction', {})
                if item_ai and item_ai.get('ai_analysis', {}).get('predicted_types'):
                    item_types = item_ai['ai_analysis']['predicted_types']
                    dangerous_types = [t for t in item_types if t != 'Normal']
                    if dangerous_types:
                        status = "danger"
                        print(f"[GMAIL] 내부 파일 AI 예측으로 위험 판정")
                        break
            
            if status == "unknown":
                status = "safe"
                print(f"[GMAIL] 모든 내부 파일 안전 확인")
                
        elif virustotal_result and virustotal_result.get('available'):
            scan_summary = virustotal_result.get('scan_summary', {})
            malicious = scan_summary.get('malicious', 0)
            if malicious > 0:
                status = "danger"
            else:
                status = "safe"
            print(f"[GMAIL] VT 기반 상태 결정: {status}")
        
        elif ai_prediction and ai_prediction.get('ai_analysis'):
            predicted_types = ai_prediction['ai_analysis'].get('predicted_types', [])
            dangerous_types = [t for t in predicted_types if t != 'Normal']
            if dangerous_types:
                status = "danger"
            else:
                status = "safe"
            print(f"[GMAIL] AI 기반 상태 결정: {status}")
        
        else:
            status = "safe"
            print(f"[GMAIL] 기본 상태 결정: {status}")
            
    except Exception as e:
        print(f"[GMAIL] 상태 결정 중 오류: {e}")
        status = "unknown"

    return {
        "ok": True, 
        "report_url": report_url, 
        "filename": fname,
        "status": status,
        "analysis_result": analysis_result,
        "ai_prediction": ai_prediction,
        "virustotal": virustotal_result
    }
