from typing import Union

from .anthropic_secret import AnthropicSecret
from .aws_secret import AWSSecret
from .azure_secret import AzureSecret
from .cohere_secret import CohereSecret
from .gcp_secret import GCPSecret
from .github_secret import GitHubSecret
from .huggingface_secret import HuggingFaceSecret
from .kubeconfig_secret import KubeConfigSecret
from .lambda_secret import LambdaSecret
from .langchain_secret import LangChainSecret
from .openai_secret import OpenAISecret
from .pinecone_secret import PineconeSecret
from .ssh_secret import SSHSecret
from .wandb_secret import WandBSecret

_str_to_provider_class = {
    # File and/or Env secrets
    "aws": AWSSecret,
    "azure": AzureSecret,
    "gcp": GCPSecret,
    "github": GitHubSecret,
    "huggingface": HuggingFaceSecret,
    "kubernetes": KubeConfigSecret,
    "lambda": LambdaSecret,
    # SSH secrets
    "ssh": SSHSecret,
    # API key secrets
    "anthropic": AnthropicSecret,
    "cohere": CohereSecret,
    "langchain": LangChainSecret,
    "openai": OpenAISecret,
    "pinecone": PineconeSecret,
    "wandb": WandBSecret,
}

_path_to_provider_class = {
    "~/.aws": AWSSecret,
    "~/.azure": AzureSecret,
    "~/.config/gcloud": GCPSecret,
    "~/.config/gh": GitHubSecret,
    "~/.cache/huggingface": HuggingFaceSecret,
    "~/.kube": KubeConfigSecret,
    "~/.lambda_cloud": LambdaSecret,
    "~/.ssh": SSHSecret,
}

_secret_to_env_vars = {
    "aws": {"access_key": "AWS_ACCESS_KEY_ID", "secret_key": "AWS_SECRET_ACCESS_KEY"},
    "azure": {"subscription_id": "AZURE_SUBSCRIPTION_ID"},
    "gcp": {"client_id": "CLIENT_ID", "client_secret": "CLIENT_SECRET"},
    "github": {},
    "huggingface": ["HF_TOKEN"],
    "anthropic": {"api_key": "ANTHROPIC_API_KEY"},
    "cohere": {"api_key": "COHERE_API_KEY"},
    "langchain": {"api_key": "LANGCHAIN_API_KEY"},
    "openai": {"api_key": "OPENAI_API_KEY"},
    "pinecone": {"api_key": "PINECONE_API_KEY"},
    "wandb": {"api_key": "WANDB_API_KEY"},
}


def _check_if_provider_secret(provider_str: str = None, provider_env_vars: dict = None):
    import os

    # user passed provider name or path to secrets values
    if provider_str:
        if provider_str in _str_to_provider_class.keys():
            return _str_to_provider_class[provider_str]

        full_default_paths_to_provider = {
            os.path.abspath(path): secret
            for path, secret in _path_to_provider_class.items()
        }
        provided_full_path = os.path.abspath(provider_str)
        return full_default_paths_to_provider.get(provided_full_path, None)

    # user passed secrets keys to env vars mapping
    elif provider_env_vars:
        for provider_name, default_provider_env_vars in _secret_to_env_vars.items():
            if provider_env_vars == default_provider_env_vars:
                return _str_to_provider_class[provider_name]
        return None


def _get_provider_class(provider_info: Union[str, dict]):
    provider_secret = (
        _check_if_provider_secret(provider_str=provider_info)
        if isinstance(provider_info, str)
        else _check_if_provider_secret(provider_env_vars=provider_info)
    )
    return provider_secret
