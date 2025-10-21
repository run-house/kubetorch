from .. import Secret


class GitHubSecret(Secret):
    """
    .. note::
            To create a GitHubSecret, please use the factory method :func:`secret` with ``provider="github"``.
    """

    # values format: {"oauth_token": oath_token}
    _PROVIDER = "github"
    _DEFAULT_PATH = "~/.config/gh"
    _DEFAULT_FILENAMES = ["hosts.yml"]
