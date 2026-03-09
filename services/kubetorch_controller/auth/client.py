"""
Authentication client for kubetorch_controller.

Makes HTTP requests to mgmt_controller's /api/v1/auth/validate endpoint.
"""

from typing import Optional, Tuple

import httpx

from auth.models import AuthenticatedUser


class AuthClient:
    """
    HTTP client for authenticating with mgmt_controller.

    Makes synchronous requests to the auth validation endpoint.
    """

    def __init__(self, auth_endpoint: str, timeout: float = 5.0):
        """
        Initialize auth client.

        Args:
            auth_endpoint: Full URL to auth endpoint (e.g., http://mgmt-controller:8000/api/v1/auth/validate)
            timeout: Request timeout in seconds
        """
        self.auth_endpoint = auth_endpoint.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def validate(
        self, token: str
    ) -> Tuple[bool, Optional[AuthenticatedUser], Optional[str]]:
        """
        Validate a token with mgmt_controller.

        Args:
            token: The token to validate (local kt_xxx or JWT)

        Returns:
            Tuple of (valid, user, error):
            - valid: True if token is valid
            - user: AuthenticatedUser if valid, None otherwise
            - error: Error message if invalid or request failed, None otherwise
        """
        try:
            response = self._client.post(
                self.auth_endpoint,
                json={"token": token},
                headers={"Content-Type": "application/json"},
            )

            if response.status_code != 200:
                return False, None, f"Auth service error: {response.status_code}"

            data = response.json()

            if data.get("valid"):
                user_data = data.get("user", {})
                user = AuthenticatedUser.from_dict(user_data)
                return True, user, None
            else:
                return False, None, data.get("error", "Invalid token")

        except httpx.ConnectError:
            return False, None, "Auth service unavailable"
        except httpx.TimeoutException:
            return False, None, "Auth service timeout"
        except Exception as e:
            return False, None, f"Auth validation failed: {str(e)}"

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
