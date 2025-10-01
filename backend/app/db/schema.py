from pydantic import BaseModel

class FileOut(BaseModel):
    id: int
    filename: str
    mime_type: str | None
    size_bytes: int | None
    excerpt_preview: str | None

    class Config:
        from_attributes = True
