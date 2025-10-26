from __future__ import annotations

import datetime as dt
import os
import secrets
from typing import Optional
from urllib.parse import urlencode

import requests
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.db import get_db
from app.db.models import OAuthAccount

router = APIRouter(prefix="/auth/google", tags=["auth-google"])

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
OAUTH_BASE = os.getenv("OAUTH_BASE", "https://knowmal.duckdns.org").rstrip("/")
CALLBACK_PATH = "/auth/google/callback"
REDIRECT_URI = f"{OAUTH_BASE}{CALLBACK_PATH}"

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"

SCOPES = [
    "openid",
    "email",
    "profile",
    "https://www.googleapis.com/auth/gmail.readonly",
]

_PENDING: dict[str, str] = {}


def _require_google_env():
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET 환경변수가 설정되지 않았습니다.",
        )


@router.get("/start")
def start(ext_id: str = Query(..., min_length=5)):
    """
    Google OAuth 동의 화면으로 리다이렉트.
    """
    _require_google_env()
    state = secrets.token_urlsafe(24)
    _PENDING[state] = ext_id

    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "include_granted_scopes": "true",
        "prompt": "select_account consent",
        "state": state,
    }
    url = f"{AUTH_URL}?{urlencode(params)}"
    return RedirectResponse(url, status_code=302)


@router.get("/callback")
def callback(code: Optional[str] = None, state: Optional[str] = None, db: Session = Depends(get_db)):
    _require_google_env()
    if not code or not state or state not in _PENDING:
        raise HTTPException(status_code=400, detail="잘못된 요청입니다(state/code).")
    ext_id = _PENDING.pop(state)

    data = {
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": REDIRECT_URI,
    }
    tok = requests.post(TOKEN_URL, data=data, timeout=15)
    if tok.status_code != 200:
        raise HTTPException(status_code=400, detail=f"토큰 교환 실패: {tok.text}")
    t = tok.json()
    access_token = t["access_token"]
    refresh_token = t.get("refresh_token")
    token_type = t.get("token_type")
    scope = t.get("scope")
    expires_in = t.get("expires_in", 3600)

    ui = requests.get(USERINFO_URL, headers={"Authorization": f"Bearer {access_token}"}, timeout=10)
    email = None
    if ui.status_code == 200:
        email = ui.json().get("email")

    expires_at = dt.datetime.utcnow() + dt.timedelta(seconds=expires_in)

    print(f"[OAuth] Received scopes: {scope}")
    print(f"[OAuth] Required scopes: {SCOPES}")
    if scope:
        granted_scopes = scope.split()
        required_scopes = SCOPES
        missing_scopes = [s for s in required_scopes if s not in granted_scopes]
        if missing_scopes:
            print(f"[OAuth] WARNING: Missing scopes: {missing_scopes}")
        else:
            print(f"[OAuth] All required scopes granted")
    
    user_id_key = f"{ext_id}:{email}" if email else ext_id
    row = (
        db.query(OAuthAccount)
        .filter(OAuthAccount.provider == "gmail", OAuthAccount.user_id == user_id_key)
        .one_or_none()
    )
    if row is None:
        row = OAuthAccount(
            provider="gmail",
            user_id=user_id_key,
            email=email,
            access_token=access_token,
            refresh_token=refresh_token,
            token_type=token_type,
            scope=scope,
            expires_at=expires_at,
        )
        db.add(row)
        print(f"[OAuth] Created new OAuth account for {ext_id}")
    else:
        row.email = email or row.email
        row.access_token = access_token
        if refresh_token:
            row.refresh_token = refresh_token
        row.token_type = token_type
        row.scope = scope
        row.expires_at = expires_at
        print(f"[OAuth] Updated existing OAuth account for {ext_id}")
    db.commit()

    html = """
<!doctype html>
<meta charset="utf-8">
<title>KnowMal · Google 연결 완료</title>
<style>
  body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;padding:24px;line-height:1.5}
  .card{max-width:520px;margin:40px auto;padding:24px;border:1px solid #e5e7eb;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,.06)}
  h1{font-size:20px;margin:0 0 8px}
  p{margin:8px 0}
  .ok{display:inline-block;margin-top:12px;padding:8px 12px;border-radius:8px;background:#16a34a;color:#fff;text-decoration:none}
</style>
<div class="card">
  <h1>Google 계정 연결이 완료되었습니다.</h1>
  <p>이 창은 자동으로 닫힙니다.</p>
  <a class="ok" href="#" onclick="window.close()">창 닫기</a>
</div>
<script>
  try { window.opener && window.opener.postMessage({source:'knowmal-oauth', ok:true}, '*'); } catch (e) {}
  setTimeout(()=>window.close(), 800);
</script>
"""
    return HTMLResponse(content=html)


@router.get("/status")
def status(
    ext_id: str = Query(...),
    email: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    row = None
    if email:
        row = (
            db.query(OAuthAccount)
            .filter(OAuthAccount.provider == "gmail", OAuthAccount.email == email)
            .order_by(OAuthAccount.updated_at.desc())
            .first()
        )
    if row is None and email:
        row = (
            db.query(OAuthAccount)
            .filter(OAuthAccount.provider == "gmail", OAuthAccount.user_id == f"{ext_id}:{email}")
            .first()
        )
    if row is None:
        row = (
            db.query(OAuthAccount)
            .filter(OAuthAccount.provider == "gmail", OAuthAccount.user_id == ext_id)
            .first()
        )
    # 진단 로그
    try:
        print(f"[OAuth][status] ext_id={ext_id} email={email} -> authorized={bool(row)} key={(row.user_id if row else None)}")
    except Exception:
        pass
    return {
        "ok": True,
        "authorized": row is not None,
        "connected": row is not None,
        "email": getattr(row, "email", None) if row else None,
    }
