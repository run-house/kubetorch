"""
Unit tests for prefix allowlist/denylist logic.

Tests the precedence rules:
1. User denylist - highest priority DENY
2. User allowlist - overrides group denylists
3. Group allowlist - if ANY group allows, it's allowed
4. Group denylist - only denies if no group allows
5. Default - allow if no allowlists exist, otherwise deny
"""


from auth.models import AuthenticatedUser

# =============================================================================
# No Restrictions (Default Allow)
# =============================================================================


class TestNoRestrictions:
    """Test behavior when no prefix restrictions exist."""

    def test_allow_any_name_no_restrictions(self):
        """With no allowlists or denylists, any name is allowed."""
        user = AuthenticatedUser(
            user_id="alice",
            user_prefix_allowlist=[],
            user_prefix_denylist=[],
            group_prefix_allowlist=[],
            group_prefix_denylist=[],
        )
        assert user.can_use_name("anything-goes") is True
        assert user.can_use_name("ml-train") is True
        assert user.can_use_name("prod-server") is True


# =============================================================================
# User Denylist (Highest Priority)
# =============================================================================


class TestUserDenylist:
    """Test that user denylist has highest priority."""

    def test_user_deny_blocks_even_with_group_allow(self):
        """User denylist overrides group allowlist."""
        user = AuthenticatedUser(
            user_id="alice",
            user_prefix_allowlist=[],
            user_prefix_denylist=["ml"],
            group_prefix_allowlist=["ml"],  # Group allows it
            group_prefix_denylist=[],
        )
        assert user.can_use_name("ml-train") is False

    def test_user_deny_blocks_even_with_user_allow(self):
        """If same prefix in both user lists, deny wins."""
        user = AuthenticatedUser(
            user_id="alice",
            user_prefix_allowlist=["ml"],
            user_prefix_denylist=["ml"],
            group_prefix_allowlist=[],
            group_prefix_denylist=[],
        )
        assert user.can_use_name("ml-train") is False

    def test_user_deny_specific_prefix(self):
        """User can deny specific prefix while allowing others."""
        user = AuthenticatedUser(
            user_id="alice",
            user_prefix_allowlist=[],
            user_prefix_denylist=["admin"],
            group_prefix_allowlist=["ml", "admin"],
            group_prefix_denylist=[],
        )
        assert user.can_use_name("ml-train") is True
        assert user.can_use_name("admin-task") is False


# =============================================================================
# User Allowlist (Overrides Group Denies)
# =============================================================================


class TestUserAllowlist:
    """Test that user allowlist overrides group denylists."""

    def test_user_allow_overrides_group_deny(self):
        """User allowlist overrides group denylist."""
        user = AuthenticatedUser(
            user_id="alice",
            user_prefix_allowlist=["ml"],
            user_prefix_denylist=[],
            group_prefix_allowlist=[],
            group_prefix_denylist=["ml"],  # Group denies it
        )
        assert user.can_use_name("ml-train") is True

    def test_user_allow_personal_prefix(self):
        """User can have personal allowed prefix."""
        user = AuthenticatedUser(
            user_id="alice",
            user_prefix_allowlist=["personal"],
            user_prefix_denylist=[],
            group_prefix_allowlist=["ml"],
            group_prefix_denylist=[],
        )
        assert user.can_use_name("personal-project") is True
        assert user.can_use_name("ml-train") is True


# =============================================================================
# Group Allowlist (Most Permissive Across Groups)
# =============================================================================


class TestGroupAllowlist:
    """Test group allowlist behavior."""

    def test_group_allow_permits_name(self):
        """Name matching group allowlist is permitted."""
        user = AuthenticatedUser(
            user_id="alice",
            user_prefix_allowlist=[],
            user_prefix_denylist=[],
            group_prefix_allowlist=["ml", "data"],
            group_prefix_denylist=[],
        )
        assert user.can_use_name("ml-train") is True
        assert user.can_use_name("data-pipeline") is True

    def test_group_allow_overrides_other_group_deny(self):
        """
        If one group allows and another denies, allow wins.

        Note: The group lists arrive pre-resolved from mgmt_controller,
        so if 'ml' is in group_prefix_allowlist, it means at least one
        group purely allows it (and it's been removed from denylist).
        """
        # This simulates the resolved state where Group A allows 'ml'
        # and Group B denies 'ml' - the resolver keeps 'ml' in allowlist only
        user = AuthenticatedUser(
            user_id="alice",
            user_prefix_allowlist=[],
            user_prefix_denylist=[],
            group_prefix_allowlist=["ml"],  # At least one group allows
            group_prefix_denylist=[],  # 'ml' removed because one group allows
        )
        assert user.can_use_name("ml-train") is True


# =============================================================================
# Group Denylist
# =============================================================================


class TestGroupDenylist:
    """Test group denylist behavior."""

    def test_group_deny_blocks_when_no_allow(self):
        """Group denylist blocks when no allowlist permits."""
        user = AuthenticatedUser(
            user_id="alice",
            user_prefix_allowlist=[],
            user_prefix_denylist=[],
            group_prefix_allowlist=[],
            group_prefix_denylist=["prod"],
        )
        assert user.can_use_name("prod-server") is False

    def test_group_deny_allows_non_matching(self):
        """Names not matching denylist are allowed (when no allowlist)."""
        user = AuthenticatedUser(
            user_id="alice",
            user_prefix_allowlist=[],
            user_prefix_denylist=[],
            group_prefix_allowlist=[],
            group_prefix_denylist=["prod"],
        )
        assert user.can_use_name("ml-train") is True
        assert user.can_use_name("dev-server") is True


# =============================================================================
# Allowlist Exists But No Match (Implicit Deny)
# =============================================================================


class TestAllowlistImplicitDeny:
    """Test that having an allowlist implicitly denies non-matching names."""

    def test_group_allowlist_denies_non_matching(self):
        """If group allowlist exists, non-matching names are denied."""
        user = AuthenticatedUser(
            user_id="alice",
            user_prefix_allowlist=[],
            user_prefix_denylist=[],
            group_prefix_allowlist=["ml", "data"],
            group_prefix_denylist=[],
        )
        assert user.can_use_name("ml-train") is True
        assert user.can_use_name("prod-server") is False  # Not in allowlist

    def test_user_allowlist_denies_non_matching(self):
        """If user allowlist exists, non-matching names are denied."""
        user = AuthenticatedUser(
            user_id="alice",
            user_prefix_allowlist=["personal"],
            user_prefix_denylist=[],
            group_prefix_allowlist=[],
            group_prefix_denylist=[],
        )
        assert user.can_use_name("personal-project") is True
        assert user.can_use_name("ml-train") is False  # Not in allowlist


# =============================================================================
# Admin Bypass
# =============================================================================


class TestAdminBypass:
    """Test that admins bypass all prefix restrictions."""

    def test_admin_ignores_user_denylist(self):
        user = AuthenticatedUser(
            user_id="admin",
            is_admin=True,
            user_prefix_denylist=["admin"],
        )
        assert user.can_use_name("admin-task") is True

    def test_admin_ignores_group_denylist(self):
        user = AuthenticatedUser(
            user_id="admin",
            is_admin=True,
            group_prefix_denylist=["prod"],
        )
        assert user.can_use_name("prod-server") is True

    def test_admin_ignores_allowlist_restriction(self):
        user = AuthenticatedUser(
            user_id="admin",
            is_admin=True,
            group_prefix_allowlist=["ml"],  # Only ml allowed for normal users
        )
        assert user.can_use_name("anything-goes") is True
