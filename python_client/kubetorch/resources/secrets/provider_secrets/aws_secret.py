from .. import Secret


class AWSSecret(Secret):
    """
    .. note::
            To create an AWSSecret, please use the factory method :func:`secret` with ``provider="aws"``.
    """

    _PROVIDER = "aws"
    _DEFAULT_PATH = "~/.aws"
    _DEFAULT_FILENAMES = ["config", "credentials"]
    _DEFAULT_ENV_VARS = {
        "access_key": "AWS_ACCESS_KEY_ID",
        "secret_key": "AWS_SECRET_ACCESS_KEY",
    }
