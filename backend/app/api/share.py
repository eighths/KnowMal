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
from app.cache.keys import share_key
from app.config import get_settings

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
    """
    1) DB에서 파일 레코드 조회
    2) Redis에 공유 페이로드 저장 (TTL 적용)
    3) BASE_URL(.env) 우선 사용하되, 비어있거나 localhost면 request.base_url로 보정
    4) report_url 반환
    """
    row = db.execute(
        select(FileRecord).where(FileRecord.id == file_id)
    ).scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="file not found")

    share_id = secrets.token_urlsafe(10)
    payload = {
        "file_id": row.id,
        "filename": row.filename,
        "mime_type": row.mime_type,
        "size_bytes": row.size_bytes,
        "content_excerpt": row.content_excerpt or "",
        "created_at": int(time.time()),
    }

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

    report_url = f"{base_url}/r/{share_id}"
    return {
        "ok": True,
        "share_id": share_id,
        "report_url": report_url,
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
    return templates.TemplateResponse(
        "report.html",
        {
            "request": request,
            "share_id": share_id,
            "data": data,
            "ttl": ttl,
            "created_at": created_at,
        },
    )