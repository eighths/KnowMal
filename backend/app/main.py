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

settings = get_settings()
app = FastAPI(title=settings.APP_NAME)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

allowed = [o.strip() for o in (getattr(settings, "ALLOWED_ORIGINS", "") or "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    #allow_origins=settings.get_allowed_origins(),
    # allow_origins=[origin.strip() for origin in settings.ALLOWED_ORIGINS.split(",")],
    allow_origins=["*"],  # 개발용으로 모든 origin 허용
    allow_credentials=True,
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

    template_name = "report.html" if is_archive else "report_unzip.html"
    
    return templates.TemplateResponse(
        template_name,
        {"request": request, "share_id": share_id, "data": data, "ttl": ttl, "created_at": created_at}
    )

app.include_router(scan_router)
app.include_router(share_router)
