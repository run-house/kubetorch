"""
Authentication middleware for kubetorch_controller.

Delegates validation to mgmt_controller's auth service.
"""

import os
from typing import Optional

from auth.client import AuthClient

from auth.models import AuthenticatedUser

from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware


# Global auth client instance (set during setup)
_auth_client: Optional[AuthClient] = None
_auth_enabled: bool = False

# Security scheme for OpenAPI docs
security = HTTPBearer(auto_error=False)


def setup_auth(auth_endpoint: Optional[str] = None) -> Optional[AuthClient]:
    """
    Initialize authentication.

    Args:
        auth_endpoint: Full URL to mgmt_controller's /api/v1/auth/validate endpoint.
                       If None or empty, auth is disabled.

    Returns:
        AuthClient if enabled, None otherwise.
    """
    global _auth_client, _auth_enabled

    # Get from env if not provided
    if auth_endpoint is None:
        auth_endpoint = os.getenv("AUTH_ENDPOINT", "")

    if not auth_endpoint:
        print("[Auth] Authentication disabled (AUTH_ENDPOINT not set)")
        _auth_enabled = False
        _auth_client = None
        return None

    print(f"[Auth] Authentication enabled, endpoint: {auth_endpoint}")
    _auth_enabled = True
    _auth_client = AuthClient(auth_endpoint)
    return _auth_client


def is_auth_enabled() -> bool:
    """Check if authentication is enabled."""
    return _auth_enabled


def get_auth_client() -> Optional[AuthClient]:
    """Get the auth client instance."""
    return _auth_client


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware that authenticates requests using mgmt_controller.

    If auth is enabled and fails, returns 401.
    If auth is disabled, passes through without authentication.
    """

    # Paths that skip authentication
    SKIP_AUTH_PATHS = {
        "/health",
        "/healthz",
        "/ready",
        "/readyz",
        "/metrics",
        "/docs",
        "/openapi.json",
        "/redoc",
    }

    async def dispatch(self, request: Request, call_next):
        # Skip auth for certain paths
        if request.url.path in self.SKIP_AUTH_PATHS:
            return await call_next(request)

        # Skip if auth is disabled
        if not _auth_enabled or not _auth_client:
            return await call_next(request)

        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return JSONResponse(
                status_code=401,
                content={"detail": "Authentication required"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Parse Bearer token
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Invalid authorization header format. Use: Bearer <token>"
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = parts[1]

        # Validate with mgmt_controller
        valid, user, error = _auth_client.validate(token)

        if not valid or not user:
            # Check if it's a service availability issue
            if error and "unavailable" in error.lower():
                return JSONResponse(
                    status_code=503,
                    content={"detail": f"Auth service error: {error}"},
                )
            return JSONResponse(
                status_code=401,
                content={"detail": error or "Invalid token"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Store authenticated user in request state
        request.state.user = user
        return await call_next(request)


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[AuthenticatedUser]:
    """
    Get the current authenticated user (if any).

    This is a FastAPI dependency that returns the user from request state,
    or None if not authenticated. Use this for optional authentication.
    """
    return getattr(request.state, "user", None)


async def require_auth(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> AuthenticatedUser:
    """
    Require authentication.

    This is a FastAPI dependency that returns the authenticated user,
    or raises HTTPException if not authenticated.
    """
    # Check if auth is disabled
    if not _auth_enabled:
        raise HTTPException(
            status_code=503,
            detail="Authentication is not configured",
        )

    # Check if mgmt_controller is reachable
    if not _auth_client:
        raise HTTPException(
            status_code=503,
            detail="Auth service not available",
        )

    user = getattr(request.state, "user", None)
    if user:
        return user

    # Get error message if available
    error = getattr(request.state, "auth_error", None)

    if error and "unavailable" in error.lower():
        raise HTTPException(
            status_code=503,
            detail=f"Auth service error: {error}",
        )

    raise HTTPException(
        status_code=401,
        detail=error or "Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


def check_namespace_access(user: Optional[AuthenticatedUser], namespace: str) -> None:
    """
    Check if the user can access a namespace.

    Args:
        user: The authenticated user (or None if auth is disabled)
        namespace: The namespace to check

    Raises:
        HTTPException: 403 if namespace access is denied
    """
    # If auth is disabled, allow all
    if not _auth_enabled or user is None:
        return

    if not user.can_access_namespace(namespace):
        raise HTTPException(
            status_code=403,
            detail=f"Access denied: namespace '{namespace}' is not in your allowed namespaces. "
            f"Allowed: {user.allowed_namespaces or 'none'}",
        )


def check_name_prefix(user: Optional[AuthenticatedUser], name: str) -> None:
    """
    Check if the user can use a resource name based on prefix restrictions.

    Args:
        user: The authenticated user (or None if auth is disabled)
        name: The resource name to check

    Raises:
        HTTPException: 403 if name prefix is not allowed
    """
    # If auth is disabled, allow all
    if not _auth_enabled or user is None:
        return

    if not user.can_use_name(name):
        reason = user.get_prefix_denial_reason(name)
        raise HTTPException(
            status_code=403,
            detail=f"Access denied: {reason}",
        )
