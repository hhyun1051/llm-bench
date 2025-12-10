"""모델 실행 모듈"""
import time
from typing import Dict, Any, Optional, Union
from langchain_core.messages import HumanMessage

from src.model_utils import create_model
from src.config_validator import load_and_validate_config


class ModelRunner:
    """LLM 모델을 실행하는 클래스"""

    def __init__(self, config: Union[str, Dict[str, Any]] = "config/models.yaml", langfuse_integration=None):
        """
        Args:
            config: 모델 설정 (파일 경로 또는 설정 딕셔너리)
            langfuse_integration: LangfuseIntegration 인스턴스 (선택)
        """
        # config가 문자열이면 파일 경로로 간주하여 로드
        if isinstance(config, str):
            self.config = load_and_validate_config(config)
        else:
            self.config = config
        self.langfuse = langfuse_integration

    def run_prompt(self, model_name: str, prompt: str) -> Dict[str, Any]:
        """
        특정 모델로 프롬프트 실행

        Args:
            model_name: 모델 이름 (config의 키)
            prompt: 실행할 프롬프트

        Returns:
            {
                "response": 응답 텍스트,
                "metadata": {
                    "response_time_ms": 응답 시간(밀리초),
                    "input_tokens": 입력 토큰 수 (가능한 경우),
                    "output_tokens": 출력 토큰 수 (가능한 경우),
                    "total_tokens": 총 토큰 수 (가능한 경우)
                }
            }
        """
        if model_name not in self.config['models']:
            raise ValueError(f"모델을 찾을 수 없습니다: {model_name}")

        model_config = self.config['models'][model_name]

        # 공통 유틸리티로 모델 생성
        model = create_model(model_config)

        # 실행 시간 측정
        start_time = time.time()

        try:
            # 메시지 형식으로 호출
            response = model.invoke([HumanMessage(content=prompt)])

            end_time = time.time()
            response_time_ms = int((end_time - start_time) * 1000)

            # 메타데이터 추출
            metadata = {
                "response_time_ms": response_time_ms,
            }

            # 사용량 정보가 있으면 추가 (response_metadata에 있을 수 있음)
            if hasattr(response, 'response_metadata') and response.response_metadata:
                usage = response.response_metadata.get('usage', {})
                if usage:
                    metadata["input_tokens"] = usage.get('input_tokens') or usage.get('prompt_tokens')
                    metadata["output_tokens"] = usage.get('output_tokens') or usage.get('completion_tokens')
                    metadata["total_tokens"] = usage.get('total_tokens')

            return {
                "response": response.content,
                "metadata": metadata
            }

        except Exception as e:
            end_time = time.time()
            response_time_ms = int((end_time - start_time) * 1000)

            return {
                "response": None,
                "error": str(e),
                "metadata": {
                    "response_time_ms": response_time_ms
                }
            }

    def list_available_models(self) -> list:
        """사용 가능한 모델 목록 반환"""
        return list(self.config['models'].keys())

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """특정 모델의 설정 정보 반환"""
        return self.config['models'].get(model_name)
