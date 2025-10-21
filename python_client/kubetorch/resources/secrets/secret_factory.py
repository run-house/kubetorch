from typing import Dict, Optional

from kubetorch.globals import config

from .secret import Secret


def secret(
    name: Optional[str] = None,
    provider: Optional[str] = None,
    path: Optional[str] = None,
    env_vars: Optional[Dict] = None,
    namespace: Optional[str] = config.namespace,
    override: Optional[bool] = False,
) -> Secret:
    """
    Builds an instance of :class:`Secret`. At most one of `provider`, `path`, or `env_vars` can be provided, to maintain
    one source of truth. For a provider, the values are inferred from the default path or environment variables for that
    provider. To load a secret by name, provide its name and namespace.

    Args:
        namespace (str, optional): Namespace to load the secret from, if we create a secret from name. Default: "default".
        name (str, optional): Name to assign the resource. If none is provided, resource name defaults to the
            provider name.
        provider (str, optional): Provider corresponding to the secret (e.g. "aws", "gcp"). To see all supported provider
            types, run ``kt.Secret.builtin_providers(as_str=True)``.
        path (str, optional): Path where the secret values are held.
        env_vars (Dict, optional): Dictionary mapping secret keys to the corresponding
            environment variable key.
        override (Bool, optional): If True, override the secret's values in Kubernetes if a secret with the same name already exists.

    Returns:
        Secret: The resulting secret object.

    Examples:

    .. code-block:: python

        import kubetorch as kt

        local_secret = kt.secret(name="in_memory_secret", values={"secret_key": "secret_val"})
        aws_secret = kt.secret(provider="aws")
        gcp_secret = kt.secret(name="my-gcp-secret", path="~/.gcp/credentials")
        lambda_secret = kt.secret(name= "my-lambda-secret", env_vars={"api_key": "LAMBDA_API_KEY"})
    """

    # env_vars or path or provider are provided
    valid_input = sum([bool(x) for x in [provider, path, env_vars]]) == 1 or (
        provider and path
    )
    valid_from_name_input = (
        sum([bool(x) for x in [provider, path, env_vars]]) == 0 and name
    )

    if not (valid_from_name_input or valid_input):
        raise ValueError(
            "You must provide exactly one of: `provider`, `path`, or `env_vars`. Alternatively, you may provide `name` to load a secret from name."
        )

    if valid_input:
        if provider:
            return Secret.from_provider(
                provider=provider, name=name, path=path, override=override
            )
        elif path and not provider:  # the case where provider + path are provided are
            return Secret.from_path(path=path, name=name, override=override)
        elif env_vars:
            return Secret.from_env(env_vars=env_vars, name=name, override=override)
    else:
        return Secret.from_name(name=name, namespace=namespace)
