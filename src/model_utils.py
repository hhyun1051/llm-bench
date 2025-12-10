"""모델 초기화 관련 유틸리티"""
import os
from typing import Dict, Any
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

from src.constants import (
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    OPENROUTER_DEFAULT_BASE_URL,
    ENV_OPENROUTER_API_KEY,
    PROVIDER_OPENROUTER
)


def create_model(model_config: Dict[str, Any]):
    """
    모델 설정으로부터 LangChain 모델 인스턴스 생성

    Args:
        model_config: 모델 설정 딕셔너리 (provider, model, temperature 등)

    Returns:
        초기화된 LangChain 모델 인스턴스

    Raises:
        ValueError: 지원하지 않는 provider인 경우
    """
    provider = model_config.get('provider')
    model_name = model_config.get('model')
    temperature = model_config.get('temperature', DEFAULT_TEMPERATURE)
    max_tokens = model_config.get('max_tokens', DEFAULT_MAX_TOKENS)

    # OpenRouter는 OpenAI 호환 API 사용
    if provider == PROVIDER_OPENROUTER:
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=model_config.get('base_url', OPENROUTER_DEFAULT_BASE_URL),
            api_key=os.getenv(ENV_OPENROUTER_API_KEY)
        )

    # 기타 provider는 init_chat_model 사용
    return init_chat_model(
        model=model_name,
        model_provider=provider,
        temperature=temperature,
        max_tokens=max_tokens
    )
