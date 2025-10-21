from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
import os

SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-dev")
SIGNER = URLSafeTimedSerializer(SECRET_KEY, salt="knowmal-api-token")

def issue_api_token(user_id: str) -> str:
    """확장→서버 호출용 단기 API 토큰 발급"""
    return SIGNER.dumps({"uid": user_id})

def verify_api_token(token: str, max_age: int = 3600) -> str:
    """토큰 검증 + 만료 확인(기본 1시간). 통과 시 user_id 반환"""
    data = SIGNER.loads(token, max_age=max_age)
    return data["uid"]

class TokenError(Exception): ...
def try_verify(token: str, max_age: int = 3600) -> str:
    try:
        return verify_api_token(token, max_age=max_age)
    except (BadSignature, SignatureExpired) as e:
        raise TokenError(str(e))
