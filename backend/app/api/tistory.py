from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, HttpUrl
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime

import requests, hashlib, html, re, io, zipfile, json

from app.db import get_db
from app.config import get_settings
from app.core.static_analysis import analyze_bytes
from app.cache.redis_client import get_redis
from app.cache.keys import file_data_key

router = APIRouter(prefix="/tistory", tags=["tistory"])


class FetchReq(BaseModel):
    url: HttpUrl
    filename: str | None = None
    page_url: HttpUrl | None = None
    cookies: str | None = None
    user_agent: str | None = None


def _safe_excerpt_bytes(b: bytes, limit_chars: int = 4000) -> str:
    # 바이너리 제어문자 제거 후 안전 디코딩
    s = b.replace(b"\x00", b"")
    try:
        txt = s.decode("utf-8", "ignore")
    except Exception:
        txt = s.decode("latin-1", "ignore")
    txt = re.sub(r"\s+\n", "\n", txt)
    return txt[:limit_chars]


def _guess_mime(resp: requests.Response, fallback: str = "application/octet-stream") -> str:
    ct = resp.headers.get("Content-Type", "").split(";")[0].strip()
    return ct or fallback


def _looks_like_docx(filename: str, raw: bytes) -> bool:
    name_hint = filename.lower().endswith(".docx")
    magic = raw[:2] == b"PK"  # zip 시그니처
    return name_hint or magic


def _extract_docx_text(raw: bytes, limit_chars: int = 4000) -> str:
    """
    DOCX(zip)에서 word/document.xml을 읽고 태그 제거 후 텍스트 추출.
    실패 시 빈 문자열 반환.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            with zf.open("word/document.xml") as f:
                xml_bytes = f.read()
        xml = xml_bytes.decode("utf-8", "ignore")
        xml = xml.replace("</w:p>", "\n")
        text_only = re.sub(r"<[^>]+>", "", xml)
        text_only = re.sub(r"[ \t\r\f\v]+", " ", text_only)
        text_only = re.sub(r"\n{3,}", "\n\n", text_only)
        return text_only.strip()[:limit_chars]
    except Exception:
        return ""


@router.post("/fetch_url")
def fetch_url(req: FetchReq, db: Session = Depends(get_db), settings=Depends(get_settings)):
    headers = {
        "User-Agent": (req.user_agent or getattr(settings, "REMOTE_USER_AGENT", "Mozilla/5.0 MalOffice/1.0")),
        "Referer": str(req.page_url or req.url),
    }
    if req.cookies:
        headers["Cookie"] = req.cookies
    # 일반 브라우저와 유사한 기본 헤더 보강 (일부 CDN의 보수적 설정 대응)
    headers.setdefault("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8")
    headers.setdefault("Accept-Language", "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7")
    headers.setdefault("Connection", "keep-alive")

    sess = requests.Session()
    sess.headers.update(headers)
    timeout = int(getattr(settings, "REMOTE_TIMEOUT", 12))
    try:
        sess.head(
            req.url,
            timeout=timeout,
            allow_redirects=True,
        )
    except requests.RequestException:
        pass

    max_bytes = int(getattr(settings, "REMOTE_MAX_BYTES", 20 * 1024 * 1024))
    try:
        resp = sess.get(
            req.url,
            timeout=timeout,
            allow_redirects=True,
            stream=True,
        )
    except requests.RequestException as e:
        raise HTTPException(502, f"get error: {e}")

    if resp.status_code >= 400:
        raise HTTPException(502, f"get failed {resp.status_code} {resp.url}")

    sha = hashlib.sha256()
    size = 0
    buf = io.BytesIO()
    for chunk in resp.iter_content(chunk_size=128 * 1024):
        if not chunk:
            continue
        size += len(chunk)
        sha.update(chunk)

        if buf.tell() < max_bytes:
            need = max_bytes - buf.tell()
            if need > 0:
                buf.write(chunk[:need])

    raw_for_excerpt = buf.getvalue()
    mime = _guess_mime(resp)
    filename = html.unescape(req.filename or resp.headers.get("Content-Disposition", "")
                             .split("filename=")[-1].strip('"')
                             or req.url.split("/")[-1] or "document.bin")

    excerpt_limit_chars = int(getattr(settings, "EXCERPT_LIMIT", 4000))
    excerpt = ""
    if _looks_like_docx(filename, raw_for_excerpt):
        excerpt = _extract_docx_text(raw_for_excerpt, excerpt_limit_chars)
    if not excerpt:
        excerpt = _safe_excerpt_bytes(raw_for_excerpt[:4 * 1024 * 1024], excerpt_limit_chars)

    sha_hex = sha.hexdigest()

    insert_sql = text(
        """
        INSERT INTO files (
            filename, mime_type, size_bytes, content_excerpt,
            source, source_url, page_url, sha256, created_at
        )
        VALUES (
            :filename, :mime, :size_bytes, :excerpt,
            'tistory', :source_url, :page_url, :sha256, :created_at
        )
        RETURNING id
        """
    )
    params = dict(
        filename=filename,
        mime=mime,
        size_bytes=size,
        excerpt=excerpt,
        source_url=str(req.url),
        page_url=str(req.page_url or ""),
        sha256=sha_hex,
        created_at=datetime.utcnow(),
    )

    try:
        new_id = db.execute(insert_sql, params).scalar_one()
        db.commit()
    except Exception:
        db.rollback()
        params2 = dict(params, excerpt="")
        new_id = db.execute(insert_sql, params2).scalar_one()
        db.commit()

    # 정적 분석 수행 (파일이 다운로드된 경우)
    analysis_result = None
    if raw_for_excerpt and len(raw_for_excerpt) > 0:
        try:
            # 모든 파일에 대해 analyze_bytes() 사용 - ZIP 파일은 내부적으로 처리됨
            analysis_result = analyze_bytes(
                file_bytes=raw_for_excerpt,
                filename=filename,
                ttl_sec=3600,  # 1시간 캐시
                use_cache=True,
                include_virustotal=True
            )

            # AI 예측 추가 (내부 파일들에 대해서도 처리됨)
            try:
                from app.services.ensemble_model_service import get_ensemble_model_service
                ens = get_ensemble_model_service()
                if ens.model_loaded:
                    # 일반 파일의 경우
                    if 'embedded_files' not in analysis_result:
                        try:
                            analysis_result['ai_prediction'] = ens.predict_malware_type(analysis_result)
                        except Exception:
                            pass
                    else:
                        # 아카이브 파일의 경우 - 내부 파일들에 AI 예측 추가
                        for item in analysis_result.get('embedded_files', []) or []:
                            rep = item.get('report')
                            if rep:
                                try:
                                    ai_prediction = ens.predict_malware_type(rep)
                                    if ai_prediction:
                                        rep['ai_prediction'] = ai_prediction
                                except Exception:
                                    rep['ai_prediction'] = None
            except Exception:
                pass

            # 분석 결과를 Redis에 저장 (ZIP인 경우에도 동일 키에 저장)
            r = get_redis()
            r.set(
                file_data_key(sha_hex), 
                json.dumps({
                    "filename": filename,
                    "mime_type": mime,
                    "size_bytes": size,
                    "sha256": sha_hex,
                    "report": analysis_result,
                }, ensure_ascii=False), 
                ex=3600
            )

        except Exception as e:
            print(f"정적 분석 실패 (파일: {filename}, SHA256: {sha_hex}): {e}")
            pass

    return {
        "ok": True,
        "id": new_id,
        "filename": filename,
        "mime_type": mime,
        "size_bytes": size,
        "sha256": sha_hex,
        "excerpt_preview": excerpt[:1000],
        "analysis_performed": analysis_result is not None,
    }