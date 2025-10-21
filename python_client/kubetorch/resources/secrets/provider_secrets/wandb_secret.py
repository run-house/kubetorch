from .. import Secret


class WandBSecret(Secret):
    """
    .. note::
            To create an WandBSecret, please use the factory method :func:`secret` with ``provider="wandb"``.
    """

    _PROVIDER = "wandb"
    _DEFAULT_ENV_VARS = {"api_key": "WANDB_API_KEY"}
