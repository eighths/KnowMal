from sqlalchemy.orm import Session
from sqlalchemy import select
from .models import FileRecord

def insert_file(db: Session, filename: str, mime: str | None, size: int | None, excerpt: str | None) -> FileRecord:
    rec = FileRecord(filename=filename, mime_type=mime, size_bytes=size, content_excerpt=excerpt)
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return rec

def list_files(db: Session, limit: int = 20):
    return db.execute(select(FileRecord).order_by(FileRecord.id.desc()).limit(limit)).scalars().all()
