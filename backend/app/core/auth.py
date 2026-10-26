"""后台账号与令牌。

口令用标准库 pbkdf2_hmac 存（加盐 + 高迭代次数，不额外引入 bcrypt/passlib）；
会话令牌是 HS256 签名的 JWT。角色写在令牌里只用于路由收权，每次请求仍回库
确认账号状态——否则禁用/删除一个坐席要等到令牌自然过期才生效。
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select

from app.config import settings
from app.core.log import get_logger
from app.models.database import AdminUser
from app.core.timeutil import utcnow_naive
from app.models.engine import async_session_factory

ROLE_ADMIN = "admin"
ROLE_SUPERVISOR = "supervisor"    # 可审批高危操作、处理工单
ROLE_OPERATOR = "operator"        # 只读运营视图

STAFF_ROLES = (ROLE_ADMIN, ROLE_SUPERVISOR, ROLE_OPERATOR)
DECIDER_ROLES = (ROLE_ADMIN, ROLE_SUPERVISOR)

_ALGO = "pbkdf2_sha256"
_ITERATIONS = 240_000
_DEV_SECRET_WARNED = False
_DEV_SECRET = "ma-cs-dev-insecure-secret-change-me"


class AuthError(Exception):
    pass


@dataclass(frozen=True)
class Staff:
    id: int
    username: str
    role: str


# ── 口令 ──────────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), _ITERATIONS)
    return f"{_ALGO}${_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iterations, salt, digest = stored.split("$")
    except ValueError:
        return False
    if algo != _ALGO:
        return False
    calc = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iterations))
    return hmac.compare_digest(calc.hex(), digest)


# ── 令牌 ──────────────────────────────────────────────────────────────────────

def _secret() -> str:
    global _DEV_SECRET_WARNED
    if settings.jwt_secret:
        return settings.jwt_secret
    if not _DEV_SECRET_WARNED:
        _DEV_SECRET_WARNED = True
        get_logger(__name__).warning("未配置 JWT_SECRET，使用开发默认密钥——上生产必须显式设置")
    return _DEV_SECRET


def create_token(user: AdminUser) -> tuple[str, int]:
    import jwt

    now = int(time.time())
    expires_at = now + settings.jwt_ttl_s
    payload = {
        "sub": user.username,
        "uid": user.id,
        "role": user.role,
        "iat": now,
        "exp": expires_at,
        "jti": secrets.token_hex(8),
    }
    token = jwt.encode(payload, _secret(), algorithm=settings.jwt_algorithm)
    return token, expires_at - now


def decode_token(token: str) -> dict:
    import jwt

    try:
        return jwt.decode(token, _secret(), algorithms=[settings.jwt_algorithm])
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("登录已过期，请重新登录") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError("登录凭证无效") from exc


# ── 登录与依赖 ────────────────────────────────────────────────────────────────

async def authenticate(username: str, password: str) -> AdminUser | None:
    async with async_session_factory() as session:
        user = (await session.execute(
            select(AdminUser).where(AdminUser.username == username)
        )).scalar_one_or_none()
        if user is None or user.deleted_at is not None or user.status != "active":
            # 账号不存在与密码错误返回同一结果，避免账号枚举
            return None
        if not verify_password(password, user.password_hash):
            return None
        user.last_login_at = utcnow_naive()
        await session.commit()
        return user


bearer = HTTPBearer(auto_error=False)


async def require_staff(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> Staff:
    if credentials is None or not credentials.credentials:
        raise HTTPException(401, "未登录", headers={"WWW-Authenticate": "Bearer"})
    try:
        claims = decode_token(credentials.credentials)
    except AuthError as exc:
        raise HTTPException(401, str(exc), headers={"WWW-Authenticate": "Bearer"}) from exc

    async with async_session_factory() as session:
        user = (await session.execute(
            select(AdminUser).where(AdminUser.id == claims.get("uid"))
        )).scalar_one_or_none()

    if user is None or user.status != "active" or user.deleted_at is not None:
        raise HTTPException(401, "账号已禁用或不存在", headers={"WWW-Authenticate": "Bearer"})
    if user.role not in STAFF_ROLES:
        raise HTTPException(403, "账号角色无后台权限")

    return Staff(id=user.id, username=user.username, role=user.role)


def require_roles(*roles: str):
    """接口级收权：Depends(require_roles(*DECIDER_ROLES))。"""

    async def dependency(staff: Staff = Depends(require_staff)) -> Staff:
        if staff.role not in roles:
            raise HTTPException(403, f"需要角色: {' 或 '.join(roles)}")
        return staff

    return dependency
