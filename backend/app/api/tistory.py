from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, HttpUrl
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime
import logging
import json

import requests, hashlib, html, re, io, zipfile

from app.db import get_db
from app.config import get_settings
from app.core.static_analysis import analyze_bytes
from app.services.ai_model_service import get_ai_model_service
from app.external.virustotal import get_virustotal_client
from app.cache.redis_client import get_redis
from app.cache.keys import file_data_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tistory", tags=["tistory"])


class FetchReq(BaseModel):
    url: HttpUrl
    filename: str | None = None
    page_url: HttpUrl | None = None
    cookies: str | None = None


def _safe_excerpt_bytes(b: bytes, limit_chars: int = 4000) -> str:
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
        "User-Agent": getattr(settings, "REMOTE_USER_AGENT", "Mozilla/5.0 MalOffice/1.0"),
        "Referer": str(req.page_url or req.url),
    }
    if req.cookies:
        headers["Cookie"] = req.cookies

    try:
        head_resp = requests.head(
            req.url,
            headers=headers,
            timeout=int(getattr(settings, "REMOTE_TIMEOUT", 12)),
            allow_redirects=True,
        )
    except requests.RequestException as e:
        logger.error(f"헤드 요청 실패: {e}")

    max_bytes = int(getattr(settings, "REMOTE_MAX_BYTES", 20 * 1024 * 1024))
    timeout = int(getattr(settings, "REMOTE_TIMEOUT", 12))
    
    try:
        resp = requests.get(
            req.url,
            headers=headers,
            timeout=timeout,
            allow_redirects=True,
            stream=True,
        )
        logger.info(f"GET 응답 상태: {resp.status_code}, Content-Type: {resp.headers.get('Content-Type')}")
    except requests.exceptions.Timeout as e:
        logger.error(f"타임아웃 발생: {e}")
        raise HTTPException(504, f"파일 다운로드 타임아웃 ({timeout}초 초과)")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"연결 에러: {e}")
        raise HTTPException(502, f"파일 서버 연결 실패: {e}")
    except requests.exceptions.RequestException as e:
        logger.error(f"네트워크 요청 실패: {e}")
        raise HTTPException(502, f"파일 다운로드 실패: {e}")

    if resp.status_code >= 400:
        logger.error(f"HTTP 에러 응답: {resp.status_code} - {resp.text[:500]}")
        raise HTTPException(502, f"파일 서버 에러 {resp.status_code}: {resp.text[:100] if resp.text else 'No response body'}")

    logger.info("파일 다운로드 시작...")
    sha = hashlib.sha256()
    size = 0
    buf = io.BytesIO()
    chunk_count = 0
    
    try:
        for chunk in resp.iter_content(chunk_size=128 * 1024):
            if not chunk:
                continue
            size += len(chunk)
            sha.update(chunk)
            chunk_count += 1

            if buf.tell() < max_bytes:
                need = max_bytes - buf.tell()
                if need > 0:
                    buf.write(chunk[:need])
            
            if chunk_count % 80 == 0:  
                logger.info(f"다운로드 진행: {size / (1024*1024):.1f}MB")

        logger.info(f"파일 다운로드 완료: {size / (1024*1024):.1f}MB, {chunk_count}개 청크")
    except Exception as e:
        logger.error(f"파일 다운로드 중 에러: {e}")
        raise HTTPException(502, f"파일 다운로드 중 에러 발생: {e}")

    raw_for_excerpt = buf.getvalue()
    mime = _guess_mime(resp)
    
    filename = html.unescape(req.filename or resp.headers.get("Content-Disposition", "")
                             .split("filename=")[-1].strip('"')
                             or req.url.split("/")[-1] or "document.bin")
    
    if filename.lower().endswith('.doc') and mime == "application/octet-stream":
        mime = "application/msword"
        logger.info(f"MIME 타입을 application/msword로 수정: {filename}")

    logger.info(f"파일 정보: {filename}, MIME: {mime}, 크기: {size}바이트")

    excerpt_limit_chars = int(getattr(settings, "EXCERPT_LIMIT", 4000))
    excerpt = ""
    if _looks_like_docx(filename, raw_for_excerpt):
        logger.debug("DOCX 파일로 감지, 텍스트 추출 시도")
        excerpt = _extract_docx_text(raw_for_excerpt, excerpt_limit_chars)
    if not excerpt:
        logger.debug("일반 텍스트 추출 시도")
        excerpt = _safe_excerpt_bytes(raw_for_excerpt[:4 * 1024 * 1024], excerpt_limit_chars)

    sha_hex = sha.hexdigest()
    logger.info(f"SHA256: {sha_hex}")

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
        logger.info("데이터베이스에 파일 정보 저장 시도...")
        new_id = db.execute(insert_sql, params).scalar_one()
        db.commit()
        logger.info(f"파일 정보 저장 성공: ID {new_id}")
    except Exception as e:
        logger.warning(f"excerpt 포함 저장 실패, excerpt 없이 재시도: {e}")
        db.rollback()
        try:
            params2 = dict(params, excerpt="")
            new_id = db.execute(insert_sql, params2).scalar_one()
            db.commit()
            logger.info(f"excerpt 없이 파일 정보 저장 성공: ID {new_id}")
        except Exception as e2:
            logger.error(f"데이터베이스 저장 완전 실패: {e2}")
            raise HTTPException(500, f"파일 정보 저장 실패: {e2}")

    analysis_result = None
    ai_prediction = None
    virustotal_result = None
    
    try:
        logger.info(f"파일 분석 시작: {filename}")
        
        analysis_result = analyze_bytes(
            raw_for_excerpt, 
            filename, 
            ttl_sec=getattr(settings, "SHARE_TTL_SECONDS", 3600), 
            use_cache=False 
        )
        logger.info(f"정적 분석 완료: {filename}")
        
        # 분석 결과 확인
        if analysis_result:
            logger.info(f"분석 결과 타입: {type(analysis_result)}")
            if isinstance(analysis_result, dict):
                logger.info(f"분석 결과 키들: {list(analysis_result.keys())}")
                if analysis_result.get('embedded_files'):
                    logger.info(f"embedded_files 수: {len(analysis_result['embedded_files'])}")
                else:
                    logger.info("embedded_files 없음")
        else:
            logger.error(f"분석 결과가 None임!")
        
        if analysis_result:
            try:
                logger.info(f"분석 결과 확인: {filename}")
                logger.info(f"Features 키: {list(analysis_result.get('features', {}).keys())}")
                
                ai_model_service = get_ai_model_service()
                if ai_model_service.model_loaded:
                    # 1) 상위(컨테이너/단일 파일)에 대한 예측
                    ai_prediction = ai_model_service.predict_malware_type(analysis_result)
                    if ai_prediction:
                        logger.info(f"AI 예측 완료: {filename}")
                    else:
                        logger.warning(f"AI 예측 실패: {filename}")

                    # 2) 압축 파일 내부 파일들에 대한 예측 포함
                    try:
                        if analysis_result.get('embedded_files'):
                            for idx, item in enumerate(analysis_result.get('embedded_files') or []):
                                child_report = (item or {}).get('report') or {}
                                if isinstance(child_report, dict) and not child_report.get('ai_prediction'):
                                    try:
                                        child_pred = ai_model_service.predict_malware_type(child_report)
                                        if child_pred:
                                            child_report['ai_prediction'] = child_pred
                                            item['report'] = child_report
                                            logger.info(f"내부 파일 AI 예측 추가: index={idx}, name={item.get('filename')}")
                                    except Exception as ce:
                                        logger.warning(f"내부 파일 AI 예측 실패(index={idx}): {ce}")
                    except Exception as ie:
                        logger.warning(f"내부 파일 AI 예측 처리 중 오류: {ie}")
                else:
                    logger.warning("AI 모델이 로드되지 않음")
            except Exception as e:
                logger.error(f"AI 예측 중 오류: {e}")
        else:
            logger.warning(f"분석 결과가 없음: {filename}")
        
        try:
            vt_api_key = getattr(settings, 'VT_API_KEY', None)
            if not vt_api_key:
                logger.warning("VirusTotal API 키가 설정되지 않음")
                virustotal_result = None
            else:
                logger.info(f"VirusTotal API 키 확인됨: {vt_api_key[:8]}...")
                vt_client = get_virustotal_client()
                # 상위(컨테이너/단일 파일)에 대한 VT 조회
                virustotal_result = vt_client.get_file_analysis(sha_hex)
                if virustotal_result and virustotal_result.get("available"):
                    logger.info(f"VirusTotal 조회 완료: {filename}")
                # 압축 내부 파일들의 VT 결과 보강
                try:
                    if analysis_result and analysis_result.get('embedded_files'):
                        for idx, item in enumerate(analysis_result.get('embedded_files') or []):
                            child_report = (item or {}).get('report') or {}
                            if not isinstance(child_report, dict):
                                continue
                            if child_report.get('virustotal'):
                                continue
                            child_file = child_report.get('file') or {}
                            child_hash = child_file.get('hash') or {}
                            child_sha = child_hash.get('sha256')
                            if not child_sha:
                                continue
                            try:
                                child_vt = vt_client.get_file_analysis(child_sha)
                                if child_vt:
                                    child_report['virustotal'] = child_vt
                                    item['report'] = child_report
                                    logger.info(f"내부 파일 VT 결과 추가: {item.get('filename')} ({child_sha})")
                            except Exception as ve:
                                logger.warning(f"내부 파일 VT 조회 실패(index={idx}): {ve}")
                except Exception as inner_e:
                    logger.warning(f"내부 파일 VT 보강 처리 중 오류: {inner_e}")
        except Exception as e:
            logger.error(f"VirusTotal 조회 중 오류: {e}")
            virustotal_result = None
        
        cache_data = {
            "filename": filename,
            "mime_type": mime,
            "size_bytes": size,
            "sha256": sha_hex,
            "file_id": new_id,
            "analysis_report": analysis_result,
            "ai_prediction": ai_prediction,
            "virustotal": virustotal_result
        }
        
        redis_client = get_redis()
        file_cache_key = file_data_key(sha_hex)
        redis_client.setex(
            file_cache_key, 
            getattr(settings, "SHARE_TTL_SECONDS", 3600), 
            json.dumps(cache_data, ensure_ascii=False)
        )
        logger.info(f"분석 결과 캐시 저장 완료: {filename}")
        
    except Exception as e:
        logger.error(f"파일 분석 중 오류 발생: {e}")

    status = "unknown"
    
    try:
        if analysis_result and analysis_result.get('embedded_files'):
            embedded_files = analysis_result['embedded_files']
            logger.info(f"압축 파일 내부 파일 검사 시작: {len(embedded_files)}개 파일")
            
            for i, item in enumerate(embedded_files):
                item_report = item.get('report', {})
                item_filename = item.get('filename', f'파일{i+1}')
                
                item_vt = item_report.get('virustotal', {})
                if item_vt.get('available') and item_vt.get('scan_summary', {}).get('malicious', 0) > 0:
                    vt_malicious = item_vt['scan_summary']['malicious']
                    vt_total = item_vt['scan_summary'].get('total', 0)
                    status = "danger"
                    logger.info(f"내부 파일 VT 탐지로 위험 판정: {item_filename} ({vt_malicious}/{vt_total} 탐지)")
                    break
                
                item_ai = item_report.get('ai_prediction', {})
                if item_ai and item_ai.get('ai_analysis', {}).get('predicted_types'):
                    item_types = item_ai['ai_analysis']['predicted_types']
                    dangerous_types = [t for t in item_types if t != 'Normal']
                    if dangerous_types:
                        status = "danger"
                        logger.info(f"내부 파일 AI 예측으로 위험 판정: {item_filename} (위험 유형: {dangerous_types})")
                        break
            
            if status == "unknown":
                status = "safe"
                logger.info(f"모든 내부 파일 안전 확인 - 압축 파일 안전 판정")
                
        elif virustotal_result and virustotal_result.get('available'):
            scan_summary = virustotal_result.get('scan_summary', {})
            malicious = scan_summary.get('malicious', 0)
            total = scan_summary.get('total', 0)
            
            if total > 0:
                detection_rate = malicious / total
                if malicious > 0:  
                    status = "danger"
                else:
                    status = "safe"
                logger.info(f"VT 기반 상태 결정: {status} (탐지: {malicious}/{total})")
        
        elif ai_prediction and ai_prediction.get('ai_analysis'):
            predicted_types = ai_prediction['ai_analysis'].get('predicted_types', [])
            dangerous_types = [t for t in predicted_types if t != 'Normal']
            
            if dangerous_types:
                status = "danger"
                logger.info(f"AI 기반 상태 결정: {status} (위험 유형: {dangerous_types})")
            else:
                status = "safe"
                logger.info(f"AI 기반 상태 결정: {status}")
        
        elif analysis_result:
            status = "safe"
            logger.info(f"기본 상태 결정: {status}")
            
    except Exception as e:
        logger.error(f"상태 결정 중 오류: {e}")
        status = "unknown"

    result = {
        "ok": True,
        "id": new_id,
        "filename": filename,
        "mime_type": mime,
        "size_bytes": size,
        "sha256": sha_hex,
        "excerpt_preview": excerpt[:1000],
        "analysis_report": analysis_result,
        "ai_prediction": ai_prediction,
        "virustotal": virustotal_result,
        "status": status  
    }
    
    logger.info(f"공유 링크 생성을 위해 file_id {new_id} 반환")
    return result