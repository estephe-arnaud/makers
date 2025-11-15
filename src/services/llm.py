"""
LLM Factory - Simple and efficient provider-agnostic LLM creation.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Type

from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import settings

logger = logging.getLogger(__name__)

# Temperature constants
DEFAULT_LLM_TEMPERATURE = 0.0
SYNTHESIS_LLM_TEMPERATURE = 0.5


@dataclass(frozen=True)
class ProviderConfig:
    """Immutable provider configuration."""
    chat_model_class: Type[BaseLanguageModel]
    model_setting: str
    api_key_setting: str | None = None
    api_key_param: str | None = None
    base_url_setting: str | None = None
    extra_config: dict | None = None


# Provider registry
PROVIDERS = {
    "openai": ProviderConfig(
        chat_model_class=ChatOpenAI,
        model_setting="DEFAULT_OPENAI_GENERATIVE_MODEL",
        api_key_setting="OPENAI_API_KEY",
        api_key_param="api_key",
    ),
    "huggingface_api": ProviderConfig(
        chat_model_class=ChatHuggingFace,
        model_setting="HUGGINGFACE_REPO_ID",
        api_key_setting="HUGGINGFACE_API_KEY",
        api_key_param="huggingfacehub_api_token",
        extra_config={"max_new_tokens": 1024},
    ),
    "ollama": ProviderConfig(
        chat_model_class=ChatOllama,
        model_setting="OLLAMA_GENERATIVE_MODEL_NAME",
        base_url_setting="OLLAMA_BASE_URL",
    ),
    "groq": ProviderConfig(
        chat_model_class=ChatGroq,
        model_setting="GROQ_MODEL_NAME",
        api_key_setting="GROQ_API_KEY",
        api_key_param="groq_api_key",
    ),
    "google": ProviderConfig(
        chat_model_class=ChatGoogleGenerativeAI,
        model_setting="GOOGLE_GEMINI_MODEL_NAME",
        api_key_setting="GOOGLE_API_KEY",
        api_key_param="google_api_key",
    ),
}


def get_llm(
    temperature: Optional[float] = None,
    model_provider_override: Optional[str] = None,
    model_name_override: Optional[str] = None,
) -> BaseLanguageModel:
    """
    Create and configure an LLM instance.
    
    Args:
        temperature: Optional temperature (defaults to DEFAULT_LLM_TEMPERATURE)
        model_provider_override: Optional provider override
        model_name_override: Optional model name override
        
    Returns:
        Configured LLM instance
        
    Raises:
        ValueError: If configuration is missing or invalid
    """
    provider = (model_provider_override or settings.DEFAULT_LLM_MODEL_PROVIDER).lower()
    temperature = temperature or DEFAULT_LLM_TEMPERATURE

    if provider not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(f"Unsupported provider: {provider}. Available: {available}")

    config = PROVIDERS[provider]
    
    # Validate required settings
    if config.api_key_setting and not getattr(settings, config.api_key_setting, None):
        raise ValueError(f"{config.api_key_setting.replace('_', ' ').title()} is missing")
    if config.base_url_setting and not getattr(settings, config.base_url_setting, None):
        raise ValueError(f"{config.base_url_setting.replace('_', ' ').title()} is missing")
    
    # Get model name
    model_name = model_name_override or getattr(settings, config.model_setting, None)
    if not model_name:
        raise ValueError(f"Model name missing for provider: {provider}")

    # Build config dict
    chat_config = {
        "model": model_name,
        "temperature": temperature,
    }
    if config.api_key_setting and config.api_key_param:
        chat_config[config.api_key_param] = getattr(settings, config.api_key_setting)
    if config.base_url_setting:
        chat_config["base_url"] = getattr(settings, config.base_url_setting)
    if config.extra_config:
        chat_config.update(config.extra_config)

    logger.info(f"Initializing {provider} LLM: model={model_name}, temperature={temperature}")
    return config.chat_model_class(**chat_config)

