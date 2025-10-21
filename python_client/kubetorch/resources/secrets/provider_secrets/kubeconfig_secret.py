from .. import Secret


class KubeConfigSecret(Secret):
    """
    .. note::
        To create a KubeConfigSecret, please use the factory method :func:`secret` with ``provider=="kubernetes"``.
    """

    _PROVIDER = "kubernetes"
    _DEFAULT_PATH = "~/.kube"
    _DEFAULT_FILENAMES = ["config"]
