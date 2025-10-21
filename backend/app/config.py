from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List
import os
from pathlib import Path

def load_env_file():
    env_file = Path(".env.dev")
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env_file()

class Settings(BaseSettings):
    APP_NAME: str = "MalOffice API"
    API_PREFIX: str = "/scan"
    DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/maloffice"
    EXCERPT_LIMIT: int = 4000

    ALLOWED_ORIGINS: str = "http://localhost"

    # Redis/Share
    REDIS_URL: str = "redis://localhost:6379/0"
    SHARE_TTL_SECONDS: int = 86400
    BASE_URL: str = "http://localhost:8000"

    # VirusTotal API
    VT_API_KEY: str = ""
    VT_ENABLED: bool = False
    VT_CACHE_TTL: int = 604800  # 7일 (VT 결과 캐시 기간)

    REMOTE_TIMEOUT: int = 30  # 원격 파일 다운로드 타임아웃 (초)
    REMOTE_MAX_BYTES: int = 50 * 1024 * 1024  # 최대 다운로드 크기 (50MB)
    REMOTE_USER_AGENT: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    class Config:
        env_file = ".env.dev"
        extra = "ignore"

    def get_allowed_origins(self) -> List[str]:
        """
        ALLOWED_ORIGINS 값을 파싱해서 리스트로 반환.
        예: "http://localhost,chrome-extension://abcd" → ["http://localhost", "chrome-extension://abcd"]
        """
        v = (self.ALLOWED_ORIGINS or "").strip()
        if v == "*" or v == "":
            return ["*"]
        return [s.strip() for s in v.split(",")]

@lru_cache
def get_settings() -> Settings:
    return Settings()
