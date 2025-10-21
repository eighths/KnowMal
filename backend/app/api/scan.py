import os, hashlib, json
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import insert

from app.db import SessionLocal
from app.db import schema
from app.db.models import FileRecord
from app.db.schema import FileOut
from app.core.static_analysis import sniff_mime, extract_excerpt, analyze_bytes, analyze_file, get_cached_report, test_zip_detection
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
        print(f"[DEBUG] analyze_bytes 호출 시작 - 파일: {filename}")
        report = analyze_bytes(
            file_bytes=file_bytes, 
            filename=filename, 
            ttl_sec=settings.SHARE_TTL_SECONDS, 
            use_cache=True,
            include_virustotal=True
        )
        print(f"[DEBUG] analyze_bytes 호출 완료 - report 타입: {type(report)}")
        
        if isinstance(report, dict):
            print(f"[DEBUG] analyze_bytes 결과 - is_archive: {report.get('file', {}).get('is_archive')}")
            print(f"[DEBUG] analyze_bytes 결과 - embedded_files 수: {len(report.get('embedded_files', []))}")
        else:
            print(f"[DEBUG] analyze_bytes 결과가 dict가 아님: {report}")

        print(f"[DEBUG] AI 모델 서비스 시작")
        ensemble_model_service = get_ensemble_model_service()
        print(f"[DEBUG] ensemble_model_service.model_loaded: {ensemble_model_service.model_loaded}")
        
        if ensemble_model_service.model_loaded and report:
            if 'embedded_files' not in report:
                print(f"[DEBUG] 일반 파일 AI 예측 시작")
                try:
                    ai_prediction = ensemble_model_service.predict_malware_type(report)
                    if ai_prediction:
                        report['ai_prediction'] = ai_prediction
                    print(f"[DEBUG] 일반 파일 AI 예측 완료")
                except Exception as e:
                    print(f"[ERROR] 일반 파일 AI 예측 실패: {e}")
                    pass
            else:
                print(f"[DEBUG] 압축 파일 내부 파일들 AI 예측 시작")
                for i, item in enumerate(report.get('embedded_files', []) or []):
                    rep = item.get('report')
                    if rep:
                        try:
                            print(f"[DEBUG] 내부 파일 {i+1} AI 예측 중: {item.get('filename')}")
                            ai_prediction = ensemble_model_service.predict_malware_type(rep)
                            if ai_prediction:
                                rep['ai_prediction'] = ai_prediction
                            print(f"[DEBUG] 내부 파일 {i+1} AI 예측 완료")
                        except Exception as e:
                            print(f"[ERROR] 내부 파일 {i+1} AI 예측 실패: {e}")
                            rep['ai_prediction'] = None
                print(f"[DEBUG] 압축 파일 내부 파일들 AI 예측 완료")

        cache_data = {
            "filename": filename,
            "mime_type": mime,
            "size_bytes": size,
            "sha256": sha256,
            "file_id": rec.id,
            "report": report
        }
        
        if report and isinstance(report, dict):
            is_archive = report.get('file', {}).get('is_archive', False)
            embedded_files = report.get('embedded_files', [])
            print(f"[DEBUG] scan.py - is_archive: {is_archive}, embedded_files 수: {len(embedded_files)}")
            if embedded_files:
                print(f"[DEBUG] 첫 번째 embedded file: {embedded_files[0].get('filename')}")

        redis_client = get_redis()
        file_cache_key = file_data_key(sha256)
        print(f"[DEBUG] Redis 캐시 저장 - 키: {file_cache_key}")
        print(f"[DEBUG] 저장될 데이터: filename={cache_data.get('filename')}, report_type={type(cache_data.get('report'))}")
        if isinstance(cache_data.get('report'), dict):
            report_data = cache_data.get('report')
            print(f"[DEBUG] report 내용: is_archive={report_data.get('file', {}).get('is_archive')}, embedded_files수={len(report_data.get('embedded_files', []))}")
        redis_client.setex(file_cache_key, settings.SHARE_TTL_SECONDS, json.dumps(cache_data, ensure_ascii=False))

    except Exception as e:
        print(f"[ERROR] analyze_bytes 실패: {e}")
        import traceback
        print(f"[ERROR] 상세 에러 정보: {traceback.format_exc()}")
        background_tasks.add_task(analyze_bytes, file_bytes, filename)

    resp = {
        "ok": True,
        "id": rec.id,
        "filename": rec.filename,
        "mime_type": rec.mime_type,
        "size_bytes": rec.size_bytes,
        "excerpt_preview": (rec.content_excerpt or "")[:300],
        "sha256": sha256
    }
    
    print(f"[DEBUG] scan.py 업로드 응답: {resp}")
    try:
        vt = (report or {}).get('virustotal') if isinstance(report, dict) else None
        ss = (vt or {}).get('scan_summary') or {}
    except Exception:
        pass
    return JSONResponse(resp)

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

@router.post("/test-zip-detection")
async def test_zip_detection_endpoint(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        result = test_zip_detection(file_bytes, file.filename)
        return JSONResponse({
            "ok": True,
            "test_result": result
        })
    except Exception as e:
        return JSONResponse({
            "ok": False,
            "error": str(e)
        }, status_code=500)