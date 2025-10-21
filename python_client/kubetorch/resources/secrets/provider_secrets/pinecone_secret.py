from .. import Secret


class PineconeSecret(Secret):
    """
    .. note::
            To create an PineconeSecret, please use the factory method :func:`secret`
            with ``provider="pinecone"``.
    """

    _PROVIDER = "pinecone"
    _DEFAULT_ENV_VARS = {"api_key": "PINECONE_API_KEY"}
