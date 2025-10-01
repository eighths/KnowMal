from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from app.api.scan import router as scan_router
from app.api.share import router as share_router, load_report_data, templates
from app.db import init_schema
from app.config import get_settings
from fastapi.staticfiles import StaticFiles
from app.api.tistory import router as tistory_router

settings = get_settings()
app = FastAPI(title=settings.APP_NAME)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins(),
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

@app.get("/r/{share_id}", response_class=HTMLResponse, tags=["share"])
def render_share_report(request: Request, share_id: str):
    data, ttl, created_at = load_report_data(share_id)
    return templates.TemplateResponse(
        "report.html",
        {"request": request, "share_id": share_id, "data": data, "ttl": ttl, "created_at": created_at}
    )

app.include_router(scan_router)
app.include_router(share_router)
