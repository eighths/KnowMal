from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from app.api.scan import router as scan_router
from app.api.share import router as share_router, load_report_data, templates
from app.db import init_schema
from app.config import get_settings
from fastapi.staticfiles import StaticFiles
from app.api.tistory import router as tistory_router
from starlette.responses import RedirectResponse
from app.api.google_oauth import router as google_oauth_router
from app.api.gmail_scan import router as gmail_router

settings = get_settings()
app = FastAPI(title=settings.APP_NAME)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

allowed = [o.strip() for o in (getattr(settings, "ALLOWED_ORIGINS", "") or "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tistory_router)

@app.on_event("startup")
def startup():
    init_schema()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/r/{share_id}", include_in_schema=False)
def report_alias(share_id: str):
    return RedirectResponse(url=f"/share/view/{share_id}", status_code=307)

@app.get("/r/{share_id}", response_class=HTMLResponse, tags=["share"])
def render_share_report(request: Request, share_id: str):
    data, ttl, created_at = load_report_data(share_id)
    return templates.TemplateResponse(
        "report.html",
        {"request": request, "share_id": share_id, "data": data, "ttl": ttl, "created_at": created_at}
    )

app.include_router(scan_router)
app.include_router(share_router)

app.include_router(google_oauth_router)
app.include_router(gmail_router)

@app.get("/auth/success")
def auth_success():
    return {"ok": True, "msg": "Google 연결 완료. 창을 닫고 다시 시도하세요."}