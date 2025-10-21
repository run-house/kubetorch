from .. import Secret


class AzureSecret(Secret):
    """
    .. note::
            To create an AzureSecret, please use the factory method :func:`secret` with ``provider="azure"``.
    """

    # values format: {"subscription_id": subscription_id}
    _PROVIDER = "azure"
    _DEFAULT_PATH = "~/.azure"
    _DEFAULT_FILENAMES = ["clouds.config"]
    _DEFAULT_ENV_VARS = {"subscription_id": "AZURE_SUBSCRIPTION_ID"}
