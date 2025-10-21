from .. import Secret


class LangChainSecret(Secret):
    """
    .. note::
            To create an LangChainSecret, please use the factory method :func:`secret`
            with ``provider="langchain"``.
    """

    _PROVIDER = "langchain"
    _DEFAULT_ENV_VARS = {"api_key": "LANGCHAIN_API_KEY"}
