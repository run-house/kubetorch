from .. import Secret


class SSHSecret(Secret):
    """
    .. note::
            To create a SSHSecret, please use the factory method :func:`secret` with ``provider="ssh"``.
    """

    _DEFAULT_PATH = "~/.ssh"
    _DEFAULT_FILENAMES = ["id_rsa"]
    _PROVIDER = "ssh"
