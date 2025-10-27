import json
import secrets
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.db import SessionLocal
from app.db.models import FileRecord
from app.cache.redis_client import get_redis
from app.cache.keys import share_key, file_data_key
from app.config import get_settings
from app.core.static_analysis import get_cached_report
from app.services.ai_model_service import get_ai_model_service

router = APIRouter(prefix="/share", tags=["share"])
templates = Jinja2Templates(directory="app/templates")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/create")
def create_share(
    request: Request,
    file_id: int = Query(...),
    db: Session = Depends(get_db),
    settings=Depends(get_settings),
):
    row = db.execute(
        select(FileRecord).where(FileRecord.id == file_id)
    ).scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="file not found")

    share_id = secrets.token_urlsafe(10)

    analysis_report = None
    ai_prediction = None
    virustotal_result = None
    gemini_explanation = None
    
    if row.sha256:
        analysis_report = get_cached_report(row.sha256)
        if not analysis_report:
            r = get_redis()
            cached_data = r.get(file_data_key(row.sha256))
            if cached_data:
                try:
                    data = json.loads(cached_data)
                    report = data.get("analysis_report") or data.get("report")
                    if isinstance(report, dict):
                        while isinstance(report, dict) and report.get("report") and isinstance(report.get("report"), dict):
                            report = report.get("report")
                        analysis_report = report
                        
                        # report 안에서 ai_prediction과 gemini_explanation 추출
                        ai_prediction = report.get("ai_prediction")
                        gemini_explanation = report.get("gemini_explanation")
                        virustotal_result = report.get("virustotal")
                    
                    # fallback: data 레벨에서도 확인
                    if not ai_prediction:
                        ai_prediction = data.get("ai_prediction")
                    if not gemini_explanation:
                        gemini_explanation = data.get("gemini_explanation")
                    if not virustotal_result:
                        virustotal_result = data.get("virustotal")
                except Exception as e:
                    print(f"[ERROR] create_share - Redis 데이터 파싱 실패: {e}")
                    pass
        else:
            # get_cached_report로 가져온 경우에도 내부 데이터 추출
            if isinstance(analysis_report, dict):
                ai_prediction = analysis_report.get("ai_prediction")
                gemini_explanation = analysis_report.get("gemini_explanation")
                virustotal_result = analysis_report.get("virustotal")
        
        if not ai_prediction and analysis_report:
            ai_model_service = get_ai_model_service()
            if ai_model_service.model_loaded:
                ai_prediction = ai_model_service.predict_malware_type(analysis_report)

    # virustotal fallback
    if not virustotal_result and isinstance(analysis_report, dict):
        embedded_vt = analysis_report.get("virustotal")
        if embedded_vt:
            virustotal_result = embedded_vt
    
    payload = {
        "file_id": row.id,
        "filename": row.filename,
        "mime_type": row.mime_type,
        "size_bytes": row.size_bytes,
        "content_excerpt": row.content_excerpt or "",
        "sha256": row.sha256,
        "analysis_report": analysis_report,
        "ai_prediction": ai_prediction,
        "virustotal": virustotal_result,
        "gemini_explanation": gemini_explanation,
        "created_at": int(time.time()),
    }
    
    print(f"[DEBUG] create_share - analysis_report 타입: {type(analysis_report)}")
    if isinstance(analysis_report, dict):
        print(f"[DEBUG] create_share - analysis_report 키들: {list(analysis_report.keys())}")
        print(f"[DEBUG] create_share - is_archive: {analysis_report.get('file', {}).get('is_archive', False)}")
        print(f"[DEBUG] create_share - embedded_files 수: {len(analysis_report.get('embedded_files', []))}")
    else:
        print(f"[DEBUG] create_share - analysis_report가 dict가 아님: {analysis_report}")

    try:
        if isinstance(virustotal_result, dict):
            ss = virustotal_result.get('scan_summary') or {}
        if isinstance(analysis_report, dict) and analysis_report.get('virustotal'):
            ss2 = analysis_report['virustotal'].get('scan_summary') or {}
    except Exception:
        pass

    try:
        if isinstance(analysis_report, dict) and analysis_report.get("embedded_files"):
            first_item = (analysis_report.get("embedded_files") or [])[:1]
            if first_item:
                first = first_item[0]
                if not analysis_report.get("file", {}).get("is_archive"):
                    payload["filename"] = first.get("filename") or payload.get("filename")
                payload["size_bytes"] = first.get("size_bytes") or payload.get("size_bytes")
                payload["sha256"] = first.get("sha256") or payload.get("sha256")
    except Exception:
        pass

    r = get_redis()
    key = share_key(share_id)
    ttl = int(getattr(settings, "SHARE_TTL_SECONDS", 86400) or 86400)
    r.set(key, json.dumps(payload, ensure_ascii=False), ex=ttl)

    base_url = (getattr(settings, "BASE_URL", "") or "").strip().rstrip("/")
    if not base_url or "localhost" in base_url:
        inferred = str(request.base_url).rstrip("/") if request else ""
        if inferred:
            base_url = inferred
    if not base_url:

        raise HTTPException(status_code=500, detail="BASE_URL not configured")

    status = "unknown"
    try:
        if isinstance(analysis_report, dict) and analysis_report.get("file", {}).get("is_archive") and analysis_report.get("embedded_files"):
            for i, item in enumerate(analysis_report.get("embedded_files") or []):
                rep = (item or {}).get("report") or {}
                vt = (rep or {}).get("virustotal") or {}
                if (vt.get("available") and (vt.get("scan_summary") or {}).get("malicious", 0) > 0):
                    status = "malicious"
                    break
                ai = (rep or {}).get("ai_prediction") or {}
                if ai and (ai.get("ai_analysis") or {}).get("predicted_types"):
                    types = ai["ai_analysis"]["predicted_types"]
                    if any(t != "Normal" for t in types):
                        status = "malicious"
                        break
            if status == "unknown":
                status = "safe"
        
        if status == "unknown" and isinstance(ai_prediction, dict):
            prediction_label = ai_prediction.get("prediction", "").lower()
            if "malicious" in prediction_label or "malware" in prediction_label:
                status = "malicious"
            elif "benign" in prediction_label or "safe" in prediction_label:
                status = "safe"
        
        if status == "unknown" and isinstance(virustotal_result, dict):
            scan_summary = virustotal_result.get("scan_summary", {})
            malicious_count = scan_summary.get("malicious", 0)
            total_count = scan_summary.get("total", 0)
            if malicious_count > 0:
                status = "malicious"
            elif total_count > 0:
                status = "safe"
    except Exception as e:
        status = "unknown"

    report_url = f"{base_url}/r/{share_id}?status={status}"
    return {
        "ok": True,
        "share_id": share_id,
        "report_url": report_url,
        "status": status,
        "ttl_seconds": ttl,
    }


def load_report_data(share_id: str):
    r = get_redis()
    key = share_key(share_id)
    raw = r.get(key)
    if not raw:
        raise HTTPException(status_code=404, detail="share not found or expired")

    data = json.loads(raw)
    ttl = r.ttl(key)
    created_at = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(data.get("created_at", int(time.time())))
    )
    return data, ttl, created_at


@router.get("/view/{share_id}")
def view_share(request: Request, share_id: str):
    data, ttl, created_at = load_report_data(share_id)
    vt_context = None
    try:
        if isinstance(data, dict):
            vt_context = data.get("virustotal") or (
                (data.get("analysis_report") or {}).get("virustotal")
                if isinstance(data.get("analysis_report"), dict)
                else None
            )
    except Exception:
        vt_context = None

    is_archive = False
    try:
        if isinstance(data, dict):
            analysis_report = data.get("analysis_report")
            if isinstance(analysis_report, dict):
                file_info = analysis_report.get("file")
                if isinstance(file_info, dict):
                    is_archive = file_info.get("is_archive", False)
    except Exception:
        is_archive = False

    if isinstance(data, dict):
        report = data.get("analysis_report") or data.get("report")
        if report:
            embedded_files = report.get("embedded_files", [])
            print(f"[DEBUG] share.py - is_archive: {is_archive}, embedded_files 수: {len(embedded_files)}")
            print(f"[DEBUG] share.py data 키들: {list(data.keys())}")
            print(f"[DEBUG] share.py analysis_report 키들: {list(report.keys()) if isinstance(report, dict) else 'Not dict'}")
            if isinstance(report, dict) and report.get('file'):
                file_info = report.get('file')
                print(f"[DEBUG] share.py file 정보: {file_info}")
            if embedded_files:
                print(f"[DEBUG] share.py 첫 번째 embedded file: {embedded_files[0].get('filename')}")
        else:
            print(f"[DEBUG] share.py - analysis_report/report가 없음")

    template_name = "report.html" if is_archive else "report_unzip.html"
    print(f"[DEBUG] share.py - 선택된 템플릿: {template_name} (is_archive={is_archive})")
    print(f"[DEBUG] share.py - data.gemini_explanation 존재: {data.get('gemini_explanation') is not None}")
    if data.get('gemini_explanation'):
        gx = data.get('gemini_explanation')
        print(f"[DEBUG] share.py - gemini_explanation 타입: {type(gx)}")
        if isinstance(gx, dict):
            print(f"[DEBUG] share.py - gemini_explanation 키들: {list(gx.keys())}")
    
    print(f"[DEBUG] share.py - data.ai_prediction 존재: {data.get('ai_prediction') is not None}")
    if data.get('ai_prediction'):
        ai = data.get('ai_prediction')
        if isinstance(ai, dict) and ai.get('ai_analysis'):
            ai_analysis = ai.get('ai_analysis')
            if isinstance(ai_analysis, dict) and ai_analysis.get('model_info'):
                model_info = ai_analysis.get('model_info')
                if isinstance(model_info, dict) and model_info.get('enhanced_features'):
                    enhanced = model_info.get('enhanced_features')
                    if isinstance(enhanced, dict):
                        fi = enhanced.get('feature_importance')
                        print(f"[DEBUG] share.py - feature_importance 존재: {fi is not None}, 개수: {len(fi) if fi else 0}")
    
    return templates.TemplateResponse(
        template_name,
        {
            "request": request,
            "share_id": share_id,
            "data": data,
            "virustotal": vt_context, 
            "ttl": ttl,
            "created_at": created_at,
        },
    )