"""
Authentication models for kubetorch_controller.

Lightweight models for user information returned from mgmt_controller.
"""

from dataclasses import dataclass, field
from typing import Optional


def _matches_prefix(name: str, prefix: str) -> bool:
    """
    Check if a name matches a prefix.

    Prefixes are stored without trailing dash, but matching adds one.
    e.g., prefix "ml" matches "ml-train", "ml-test", etc.
    """
    return name.startswith(f"{prefix}-")


def _matches_any_prefix(name: str, prefixes: list[str]) -> bool:
    """Check if name matches any of the given prefixes."""
    return any(_matches_prefix(name, p) for p in prefixes)


@dataclass
class AuthenticatedUser:
    """
    Represents an authenticated user.

    This is returned from mgmt_controller's /api/v1/auth/validate endpoint.
    """

    user_id: str
    email: Optional[str] = None
    groups: list[str] = field(default_factory=list)
    auth_source: str = "local"  # 'local' or 'oidc'
    token_id: Optional[str] = None
    quotas: dict = field(default_factory=dict)
    allowed_namespaces: list[str] = field(default_factory=list)
    is_admin: bool = False
    metadata: dict = field(default_factory=dict)

    # Prefix restrictions - separate user and group lists for proper precedence
    # Note: Group lists arrive pre-resolved from mgmt_controller:
    # - Within each group: deny overrides allow (already filtered out)
    # - Across groups: allow wins (group_prefix_denylist excludes anything in group_prefix_allowlist)
    user_prefix_allowlist: list[str] = field(default_factory=list)
    user_prefix_denylist: list[str] = field(default_factory=list)
    group_prefix_allowlist: list[str] = field(default_factory=list)
    group_prefix_denylist: list[str] = field(default_factory=list)

    def can_access_namespace(self, namespace: str) -> bool:
        """
        Check if the user can access a given namespace.

        Rules:
        - Admins can access any namespace
        - Empty allowed_namespaces means no restrictions (allow any)
        - Otherwise, namespace must be in allowed_namespaces
        """
        if self.is_admin:
            return True
        if not self.allowed_namespaces:
            return True  # No restrictions
        return namespace in self.allowed_namespaces

    def can_use_name(self, name: str) -> bool:
        """
        Check if the user can use a given resource name based on prefix restrictions.

        Precedence rules:
        1. User denylist - highest priority DENY (overrides everything)
        2. User allowlist - overrides group denylists
        3. Group allowlist - already resolved: if ANY group purely allows, it's here
        4. Group denylist - already resolved: excludes prefixes in group_allowlist
        5. Default - allow if no allowlists exist, otherwise deny

        Note: Group lists arrive pre-resolved from mgmt_controller with proper
        within-group (deny > allow) and across-group (allow wins) precedence.

        Prefixes are stored without trailing dash but matched with one.
        e.g., prefix "ml" matches names starting with "ml-"
        """
        if self.is_admin:
            return True

        # 1. User denylist - highest priority DENY
        if _matches_any_prefix(name, self.user_prefix_denylist):
            return False

        # 2. User allowlist - overrides group denylists
        if _matches_any_prefix(name, self.user_prefix_allowlist):
            return True

        # 3. Group allowlist - if ANY group allows, it's allowed (most permissive)
        if _matches_any_prefix(name, self.group_prefix_allowlist):
            return True

        # 4. Group denylist - only denies if not allowed by any group
        if _matches_any_prefix(name, self.group_prefix_denylist):
            return False

        # 5. Default behavior based on whether allowlists exist
        # If any allowlist exists (user or group), name must match one
        has_allowlist = bool(self.user_prefix_allowlist or self.group_prefix_allowlist)
        if has_allowlist:
            # Allowlists exist but name doesn't match any -> deny
            return False

        # No allowlists exist -> allow by default
        return True

    def get_prefix_denial_reason(self, name: str) -> Optional[str]:
        """
        Get a human-readable reason for why a name is denied.

        Returns None if the name is allowed.
        """
        if self.is_admin:
            return None

        # Check user denylist
        for prefix in self.user_prefix_denylist:
            if _matches_prefix(name, prefix):
                return f"Name '{name}' matches user-denied prefix '{prefix}'"

        # Check if user allowlist allows it
        if _matches_any_prefix(name, self.user_prefix_allowlist):
            return None

        # Check if group allowlist allows it
        if _matches_any_prefix(name, self.group_prefix_allowlist):
            return None

        # Check group denylist
        for prefix in self.group_prefix_denylist:
            if _matches_prefix(name, prefix):
                return f"Name '{name}' matches group-denied prefix '{prefix}'"

        # Check if allowlists require a match
        has_allowlist = bool(self.user_prefix_allowlist or self.group_prefix_allowlist)
        if has_allowlist:
            all_allowed = list(
                set(self.user_prefix_allowlist) | set(self.group_prefix_allowlist)
            )
            return (
                f"Name '{name}' does not match any allowed prefix. "
                f"Allowed prefixes: {all_allowed}"
            )

        return None

    @classmethod
    def from_dict(cls, data: dict) -> "AuthenticatedUser":
        """Create from dictionary (API response)."""
        return cls(
            user_id=data.get("user_id", "unknown"),
            email=data.get("email"),
            groups=data.get("groups", []),
            auth_source=data.get("auth_source", "local"),
            token_id=data.get("token_id"),
            quotas=data.get("quotas", {}),
            allowed_namespaces=data.get("allowed_namespaces", []),
            is_admin=data.get("is_admin", False),
            metadata=data.get("metadata", {}),
            user_prefix_allowlist=data.get("user_prefix_allowlist", []),
            user_prefix_denylist=data.get("user_prefix_denylist", []),
            group_prefix_allowlist=data.get("group_prefix_allowlist", []),
            group_prefix_denylist=data.get("group_prefix_denylist", []),
        )
