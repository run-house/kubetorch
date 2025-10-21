from .. import Secret


class HuggingFaceSecret(Secret):
    """
    .. note::
            To create a HuggingFaceSecret, please use the factory method :func:`secret` with
            ``provider="huggingface"``.
    """

    # values format: {"token": hf_token}
    _PROVIDER = "huggingface"
    _DEFAULT_PATH = "~/.cache/huggingface"
    _DEFAULT_FILENAMES = ["token"]
    _DEFAULT_ENV_VARS = ["HF_TOKEN"]

    # Ensure secrets can be used as environment variables
    _MAP_FILENAMES_TO_ENV_VARS = {
        "token": "HF_TOKEN",
    }
