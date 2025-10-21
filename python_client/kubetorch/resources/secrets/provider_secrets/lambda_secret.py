from .. import Secret


class LambdaSecret(Secret):
    """
    .. note::
            To create a LambdaSecret, please use the factory method :func:`secret` with ``provider="lambda"``.
    """

    # values format: {"api_key": api_key}
    _DEFAULT_PATH = "~/.lambda_cloud"
    _DEFAULT_FILENAMES = ["lambda_keys"]
    _PROVIDER = "lambda"
