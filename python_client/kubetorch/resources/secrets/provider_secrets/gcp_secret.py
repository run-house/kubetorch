from .. import Secret


class GCPSecret(Secret):
    """
    .. note::
            To create a GCPSecret, please use the factory method :func:`secret` with ``provider="gcp"``.
    """

    _PROVIDER = "gcp"
    _DEFAULT_PATH = "~/.config/gcloud"
    _DEFAULT_FILENAMES = ["application_default_credentials.json"]
    _DEFAULT_ENV_VARS = {
        "client_id": "CLIENT_ID",
        "client_secret": "CLIENT_SECRET",
    }
