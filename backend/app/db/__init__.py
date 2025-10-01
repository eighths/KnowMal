from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
from .models import Base
import os, time
from typing import Generator  

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/maloffice"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True) 
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False) 

def _wait_for_db(max_tries: int = 60, delay: float = 1.0):
    tries = 0
    while True:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return
        except OperationalError:
            tries += 1
            if tries >= max_tries:
                raise
            time.sleep(delay)

def init_schema():
    _wait_for_db()
    Base.metadata.create_all(bind=engine)

def get_db() -> Generator:
    """
    FastAPI Depends(get_db)에서 사용.
    yield 이후에 세션을 반드시 닫아줍니다.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
