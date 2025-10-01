from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime

import hashlib, html, re, io, zipfile

from app.db import get_db
from app.config import get_settings

router = APIRouter(prefix="/scan", tags=["scan"])


def _safe_excerpt_bytes(b: bytes, limit_chars: int = 4000) -> str:
    s = b.replace(b"\x00", b"")
    try:
        txt = s.decode("utf-8", "ignore")
    except Exception:
        txt = s.decode("latin-1", "ignore")
    txt = re.sub(r"\s+\n", "\n", txt)
    return txt[:limit_chars]


def _looks_like_docx(filename: str, raw: bytes) -> bool:
    name_hint = filename.lower().endswith(".docx")
    magic = raw[:2] == b"PK"
    return name_hint or magic


def _extract_docx_text(raw: bytes, limit_chars: int = 4000) -> str:
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


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    settings=Depends(get_settings),
):
    try:
        raw = await file.read()
    except Exception as e:
        raise HTTPException(400, f"파일 읽기 실패: {e}")

    size = len(raw)
    sha_hex = hashlib.sha256(raw).hexdigest()
    mime = file.content_type or "application/octet-stream"
    filename = html.unescape(file.filename or "document.bin")

    excerpt_limit_chars = int(getattr(settings, "EXCERPT_LIMIT", 4000))
    excerpt = ""
    if _looks_like_docx(filename, raw):
        excerpt = _extract_docx_text(raw, excerpt_limit_chars)
    if not excerpt:
        excerpt = _safe_excerpt_bytes(raw[:4 * 1024 * 1024], excerpt_limit_chars)

    insert_sql = text(
        """
        INSERT INTO files (
            filename, mime_type, size_bytes, content_excerpt,
            source, source_url, page_url, sha256, created_at
        )
        VALUES (
            :filename, :mime, :size_bytes, :excerpt,
            'upload', '', '', :sha256, :created_at
        )
        RETURNING id
        """
    )
    params = dict(
        filename=filename,
        mime=mime,
        size_bytes=size,
        excerpt=excerpt,
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

    return {
        "ok": True,
        "id": new_id,
        "filename": filename,
        "mime_type": mime,
        "size_bytes": size,
        "sha256": sha_hex,
        "excerpt_preview": excerpt[:1000],
    }