import os, hashlib, json
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import insert

from app.db import SessionLocal
from app.db import schema
from app.db.models import FileRecord
from app.db.schema import FileOut
from app.core.static_analysis import sniff_mime, extract_excerpt, analyze_bytes, analyze_file, get_cached_report, analyze_zip_bytes
from app.cache.redis_client import get_redis
from app.cache.keys import file_data_key
from app.config import get_settings
from app.services.ai_model_service import get_ai_model_service
from app.services.ensemble_model_service import get_ensemble_model_service


router = APIRouter(prefix="/scan", tags=["scan"])
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/upload", response_model=schema.FileOut)
async def upload(file: UploadFile, background_tasks: BackgroundTasks, db: Session = Depends(get_db), settings=Depends(get_settings)):
    file_bytes = await file.read()
    filename = file.filename or "upload.bin"

    mime = sniff_mime(file_bytes=file_bytes)
    size = len(file_bytes)
    excerpt = extract_excerpt(mime=mime, max_len=settings.EXCERPT_LIMIT, file_bytes=file_bytes)

    sha256 = hashlib.sha256(file_bytes).hexdigest()

    result = db.execute(
        insert(FileRecord).values(
            filename=filename,
            mime_type=mime,
            size_bytes=size,
            content_excerpt=excerpt,
            source="upload",
            source_url=None,
            page_url=None,
            sha256=sha256,
        ).returning(
            FileRecord.id, FileRecord.filename, FileRecord.mime_type, FileRecord.size_bytes, FileRecord.content_excerpt
        )
    )
    rec = result.first()
    db.commit()

    try:
        def _looks_like_zip_bytes(b: bytes) -> bool:
            sig = b[:4]
            return sig.startswith(b"PK\x03\x04") or sig.startswith(b"PK\x05\x06") or sig.startswith(b"PK\x07\x08") or b[:2] == b"PK"

        is_zip = (
            filename.lower().endswith('.zip') or
            (mime or '').lower().startswith('application/zip') or
            (mime or '').lower().endswith('zip') or
            _looks_like_zip_bytes(file_bytes)
        )

        if is_zip:
            report = analyze_zip_bytes(
                zip_bytes=file_bytes,
                filename=filename,
                ttl_sec=settings.SHARE_TTL_SECONDS,
                use_cache=True,
                include_virustotal=True,
                passwords=[None, 'pass']
            )

            ensemble_model_service = get_ensemble_model_service()
            if ensemble_model_service.model_loaded:
                for item in report.get('embedded_files', []) or []:
                    rep = item.get('report')
                    if rep:
                        try:
                            item['ai_prediction'] = ensemble_model_service.predict_malware_type(rep)
                        except Exception:
                            item['ai_prediction'] = None

            cache_data = {
                "filename": filename,
                "mime_type": mime,
                "size_bytes": size,
                "sha256": sha256,
                "file_id": rec.id,
                "report": report
            }
        else:
            report = analyze_bytes(file_bytes, filename, ttl_sec=settings.SHARE_TTL_SECONDS, use_cache=True)

            ensemble_model_service = get_ensemble_model_service()
            ai_prediction = None
            if ensemble_model_service.model_loaded and report:
                ai_prediction = ensemble_model_service.predict_malware_type(report)

            cache_data = {
                "filename": filename,
                "mime_type": mime,
                "size_bytes": size,
                "sha256": sha256,
                "file_id": rec.id,
                "report": report
            }
            if ai_prediction:
                cache_data["ai_prediction"] = ai_prediction

        redis_client = get_redis()
        file_cache_key = file_data_key(sha256)
        redis_client.setex(file_cache_key, settings.SHARE_TTL_SECONDS, json.dumps(cache_data, ensure_ascii=False))

    except Exception as e:
        background_tasks.add_task(analyze_bytes, file_bytes, filename)

    return JSONResponse({
        "ok": True,
        "id": rec.id,
        "filename": rec.filename,
        "mime_type": rec.mime_type,
        "size_bytes": rec.size_bytes,
        "excerpt_preview": (rec.content_excerpt or "")[:300],
        "sha256": sha256
    })

@router.get("/recent", response_model=list[FileOut])
def recent(db: Session = Depends(get_db)):
    rows = db.query(FileRecord)\
             .order_by(FileRecord.id.desc())\
             .limit(50).all()
    return [
        {
            "id": r.id, "filename": r.filename, "mime_type": r.mime_type,
            "size_bytes": r.size_bytes,
            "excerpt_preview": (r.content_excerpt or "")[:300]
        } for r in rows
    ]

@router.post("/analyze/{file_id}")
def analyze_by_id(file_id: int, db: Session = Depends(get_db)):
    rec = db.query(FileRecord).filter(FileRecord.id == file_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="file not found")

    report = analyze_file(rec.path, ttl_sec=3600, use_cache=True)
    return {"file_id": rec.id, "sha256": report["file"]["hash"]["sha256"], "report": report}


@router.get("/report/{sha256}")
def get_report(sha256: str):
    report = get_cached_report(sha256)
    if report:
        return report
    
    r = get_redis()
    cached_data = r.get(file_data_key(sha256))
    if cached_data:
        try:
            data = json.loads(cached_data)
            report = data.get("report")
            
            while isinstance(report, dict) and "report" in report and isinstance(report["report"], dict):
                report = report["report"]
            
            return report
        except Exception:
            pass
    
    raise HTTPException(status_code=404, detail="report not found")

@router.get("/model/status")
def get_model_status():
    ensemble_service = get_ensemble_model_service()
    ai_service = get_ai_model_service()
    
    return {
        "ensemble_model": ensemble_service.get_model_status(),
        "legacy_ai_model": {
            "model_loaded": ai_service.model_loaded,
            "models_dir_exists": ensemble_service.models_dir.exists()
        }
    }

@router.post("/model/reload")
def reload_models():
    ensemble_service = get_ensemble_model_service()
    ensemble_success = ensemble_service.reload_model()
    
    ai_service = get_ai_model_service()
    ai_success = ai_service.load_models()
    
    return {
        "ensemble_model": {
            "success": ensemble_success,
            "message": "Ensemble model reloaded successfully" if ensemble_success else "Failed to reload ensemble model"
        },
        "legacy_ai_model": {
            "success": ai_success,
            "message": "Legacy AI model reloaded successfully" if ai_success else "Failed to reload legacy AI model"
        }
    }