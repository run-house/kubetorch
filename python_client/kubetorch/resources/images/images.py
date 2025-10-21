from .image import Image


def debian() -> Image:
    """Return a Kubetorch Debian slim image.

    This uses the default Kubetorch server image, which is built on top of
    a minimal Debian base.
    """
    import kubetorch.serving.constants as serving_constants
    from kubetorch.globals import config

    if config.tracing_enabled:
        image_id = serving_constants.SERVER_IMAGE_WITH_OTEL
    else:
        image_id = serving_constants.SERVER_IMAGE_MINIMAL
    return Image(name="debian", image_id=image_id)


def ubuntu() -> Image:
    """Return a Kubetorch ubuntu image."""
    import kubetorch.serving.constants as serving_constants
    from kubetorch.globals import config

    if config.tracing_enabled:
        image_id = serving_constants.UBUNTU_IMAGE_WITH_OTEL
    else:
        image_id = serving_constants.UBUNTU_IMAGE_MINIMAL
    return Image(name="ubuntu", image_id=image_id)


def python(version: str) -> Image:
    """Return a Python slim base image, e.g. ``python('3.12')``."""
    tag = version.replace(".", "")
    return Image(name=f"python{tag}", image_id=f"python:{version}-slim")


def ray(version: str = "latest") -> Image:
    """Return a Ray base image, defaults to ``ray:latest``."""
    return Image(
        name=f"ray{version if version != 'latest' else ''}".strip(),
        image_id=f"rayproject/ray:{version}",
    )


def pytorch(version: str = "23.12-py3") -> Image:
    """Return an NVIDIA PyTorch base image. Defaults to ``nvcr.io/nvidia/pytorch:23.12-py3``."""
    tag = version.replace(".", "").replace("-", "")
    return Image(name=f"pytorch{tag}", image_id=f"nvcr.io/nvidia/pytorch:{version}")


# Predefined convenience aliases for common versions
Python310 = lambda: python("3.10")
Python311 = lambda: python("3.11")
Python312 = lambda: python("3.12")
Ray = lambda: ray("latest")
Pytorch2312 = lambda: pytorch("23.12-py3")
Debian = lambda: debian()
Ubuntu = lambda: ubuntu()

__all__ = [
    "python",
    "ray",
    "pytorch",
    "debian",
    "ubuntu",
    "Python310",
    "Python311",
    "Python312",
    "Ray",
    "Pytorch2312",
    "Debian",
    "Ubuntu",
]
