from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List

class Settings(BaseSettings):
    APP_NAME: str = "MalOffice API"
    API_PREFIX: str = "/scan"
    DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/maloffice"
    EXCERPT_LIMIT: int = 4000

    ALLOWED_ORIGINS: str = "http://localhost"

    REDIS_URL: str = "redis://localhost:6379/0"
    SHARE_TTL_SECONDS: int = 86400
    BASE_URL: str = "http://localhost:8000"

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
