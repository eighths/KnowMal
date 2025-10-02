from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy import String, BigInteger, Text, TIMESTAMP, func, Integer

Base = declarative_base()

class FileRecord(Base):
    __tablename__ = "files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String, nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    content_excerpt: Mapped[str | None] = mapped_column(Text, nullable=True)

    source: Mapped[str | None] = mapped_column(String(32), nullable=True)  # 'upload'/'tistory'/'gmail'/...
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)    # 원본 다운로드 URL
    page_url: Mapped[str | None] = mapped_column(Text, nullable=True)      # 발견된 페이지 URL
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)  # 전체 다운로드 해시(64 hex)

    created_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=False), server_default=func.now())