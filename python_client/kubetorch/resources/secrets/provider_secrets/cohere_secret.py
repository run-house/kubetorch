from .. import Secret


class CohereSecret(Secret):
    """
    .. note::
            To create an CohereSecret, please use the factory method :func:`secret`
            with ``provider="cohere"``.
    """

    _PROVIDER = "cohere"
    _DEFAULT_ENV_VARS = {"api_key": "COHERE_API_KEY"}
