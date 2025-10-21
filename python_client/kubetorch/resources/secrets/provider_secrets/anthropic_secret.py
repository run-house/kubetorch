from .. import Secret


class AnthropicSecret(Secret):
    """
    .. note::
            To create an AnthropicSecret, please use the factory method :func:`secret`
            with ``provider="anthropic"``.
    """

    _PROVIDER = "anthropic"
    _DEFAULT_ENV_VARS = {"api_key": "ANTHROPIC_API_KEY"}
