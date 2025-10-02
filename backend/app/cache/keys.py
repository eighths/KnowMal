def share_key(share_id: str) -> str:
    return f"share:{share_id}"

def file_data_key(sha256: str) -> str:
    return f"file:data:{sha256}"

def file_metadata_key(sha256: str) -> str:
    return f"file:meta:{sha256}"