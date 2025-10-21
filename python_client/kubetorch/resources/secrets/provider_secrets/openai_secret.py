from .. import Secret


class OpenAISecret(Secret):
    """
    .. note::
            To create an OpenAISecret, please use the factory method :func:`secret` with ``provider="openai"``.
    """

    _PROVIDER = "openai"
    _DEFAULT_ENV_VARS = {"api_key": "OPENAI_API_KEY"}
