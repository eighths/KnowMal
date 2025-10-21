from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy import String, BigInteger, Text, TIMESTAMP, func, Integer, UniqueConstraint

Base = declarative_base()

class FileRecord(Base):
    __tablename__ = "files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String, nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    content_excerpt: Mapped[str | None] = mapped_column(Text, nullable=True)

    source: Mapped[str | None] = mapped_column(String(32), nullable=True) 
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True) 
    page_url: Mapped[str | None] = mapped_column(Text, nullable=True) 
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True) 

    created_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=False), server_default=func.now())


class OAuthAccount(Base):
    """
    Gmail 등 외부 OAuth 자격 증명 저장용.
    provider + user_id(이메일/고유ID) 로 유니크.
    """
    __tablename__ = "oauth_accounts"
    __table_args__ = (UniqueConstraint("provider", "user_id", name="uq_oauth_provider_user"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    provider: Mapped[str] = mapped_column(String(32), nullable=False)   
    user_id: Mapped[str] = mapped_column(String(320), nullable=False)  
    email: Mapped[str | None] = mapped_column(String(320), nullable=True)

    access_token: Mapped[str] = mapped_column(Text, nullable=False)
    refresh_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    scope: Mapped[str | None] = mapped_column(Text, nullable=True)

    expires_at: Mapped[str | None] = mapped_column(TIMESTAMP(timezone=False), nullable=True)

    created_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=False), server_default=func.now())
    updated_at: Mapped[str] = mapped_column(
        TIMESTAMP(timezone=False), server_default=func.now(), onupdate=func.now()
    )
