from dataclasses import dataclass
from urllib.parse import unquote

from fastapi import Depends, Header, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import settings

PUBLIC_GROUP = "public"
bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Principal:
    user_id: str
    groups: frozenset[str]

    @property
    def allowed_groups(self) -> frozenset[str]:
        return self.groups | {PUBLIC_GROUP}


def _decode_jwt(token: str) -> Principal:
    try:
        import jwt

        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        user_id = str(payload["sub"])
        groups = payload.get("groups", [])
        if not isinstance(groups, list) or not all(isinstance(group, str) for group in groups):
            raise ValueError("groups must be a string list")
        return Principal(user_id=user_id, groups=frozenset(groups))
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid access token") from exc


def get_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    x_dev_user: str | None = Header(default=None),
    x_dev_groups: str | None = Header(default=None),
) -> Principal:
    if settings.auth_mode == "jwt":
        if credentials is None:
            raise HTTPException(status_code=401, detail="Bearer token required")
        return _decode_jwt(credentials.credentials)

    # HTTP 头只允许 ASCII：客户端对中文组名做 URL 百分号编码（如 %E8%B4%A2%E5%8A%A1），这里解码
    user_id = unquote(x_dev_user or "anonymous").strip()
    groups = frozenset(
        unquote(group).strip()
        for group in (x_dev_groups or "").split(",")
        if group.strip()
    )
    return Principal(user_id=user_id, groups=groups)

