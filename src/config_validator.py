"""설정 파일 검증 모듈"""
from typing import Dict, Any, List
from pathlib import Path
import yaml

from src.logger import get_logger
from src.constants import (
    PROVIDER_OPENAI,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE,
    PROVIDER_OPENROUTER
)


logger = get_logger(__name__)

SUPPORTED_PROVIDERS = {
    PROVIDER_OPENAI,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE,
    PROVIDER_OPENROUTER
}


class ConfigValidationError(Exception):
    """설정 검증 오류"""
    pass


def validate_model_config(model_name: str, model_config: Dict[str, Any]) -> List[str]:
    """
    개별 모델 설정 검증

    Args:
        model_name: 모델 이름
        model_config: 모델 설정 딕셔너리

    Returns:
        검증 오류 메시지 리스트 (빈 리스트면 검증 성공)
    """
    errors = []

    # 필수 필드 확인
    if 'provider' not in model_config:
        errors.append(f"{model_name}: 'provider' 필드가 없습니다")
    elif model_config['provider'] not in SUPPORTED_PROVIDERS:
        errors.append(
            f"{model_name}: 지원하지 않는 provider '{model_config['provider']}'. "
            f"지원: {', '.join(SUPPORTED_PROVIDERS)}"
        )

    if 'model' not in model_config:
        errors.append(f"{model_name}: 'model' 필드가 없습니다")

    # 선택 필드 타입 확인
    if 'temperature' in model_config:
        temp = model_config['temperature']
        if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
            errors.append(f"{model_name}: 'temperature'는 0~2 사이의 숫자여야 합니다 (현재: {temp})")

    if 'max_tokens' in model_config:
        max_tok = model_config['max_tokens']
        if not isinstance(max_tok, int) or max_tok <= 0:
            errors.append(f"{model_name}: 'max_tokens'는 양의 정수여야 합니다 (현재: {max_tok})")

    # OpenRouter 전용 필드 확인
    if model_config.get('provider') == PROVIDER_OPENROUTER:
        if 'base_url' in model_config and not isinstance(model_config['base_url'], str):
            errors.append(f"{model_name}: 'base_url'은 문자열이어야 합니다")

    return errors


def validate_config_file(config_path: str) -> Dict[str, Any]:
    """
    설정 파일 검증 및 로드

    Args:
        config_path: 설정 파일 경로

    Returns:
        검증된 설정 딕셔너리

    Raises:
        ConfigValidationError: 검증 실패 시
        FileNotFoundError: 파일이 없을 시
    """
    # 파일 존재 확인
    if not Path(config_path).exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

    # YAML 파싱
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigValidationError(f"YAML 파싱 오류: {e}")

    # 최상위 구조 확인
    if not isinstance(config, dict):
        raise ConfigValidationError("설정 파일은 딕셔너리 형식이어야 합니다")

    if 'models' not in config:
        raise ConfigValidationError("'models' 필드가 없습니다")

    if not isinstance(config['models'], dict):
        raise ConfigValidationError("'models' 필드는 딕셔너리여야 합니다")

    if not config['models']:
        raise ConfigValidationError("최소 하나의 모델이 정의되어야 합니다")

    # 각 모델 설정 검증
    all_errors = []
    for model_name, model_config in config['models'].items():
        if not isinstance(model_config, dict):
            all_errors.append(f"{model_name}: 모델 설정은 딕셔너리여야 합니다")
            continue

        errors = validate_model_config(model_name, model_config)
        all_errors.extend(errors)

    # 오류가 있으면 예외 발생
    if all_errors:
        error_msg = "설정 파일 검증 실패:\n  - " + "\n  - ".join(all_errors)
        raise ConfigValidationError(error_msg)

    logger.info(f"설정 파일 검증 성공: {len(config['models'])}개 모델 로드됨")
    return config


def load_and_validate_config(config_path: str) -> Dict[str, Any]:
    """
    설정 파일 로드 및 검증 (간편 함수)

    Args:
        config_path: 설정 파일 경로

    Returns:
        검증된 설정 딕셔너리

    Raises:
        ConfigValidationError: 검증 실패 시
        FileNotFoundError: 파일이 없을 시
    """
    return validate_config_file(config_path)
