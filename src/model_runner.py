"""모델 실행 모듈"""
import os
import time
from typing import Dict, Any, Optional
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import yaml


class ModelRunner:
    """LLM 모델을 실행하는 클래스"""

    def __init__(self, config_path: str = "config/models.yaml"):
        """
        Args:
            config_path: 모델 설정 파일 경로
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

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

        # OpenRouter는 별도 처리 (OpenAI 호환 API 사용)
        if model_config['provider'] == 'openrouter':
            model = ChatOpenAI(
                model=model_config['model'],
                temperature=model_config.get('temperature', 0.7),
                max_tokens=model_config.get('max_tokens', 2000),
                base_url=model_config.get('base_url', 'https://openrouter.ai/api/v1'),
                api_key=os.getenv('OPENROUTER_API_KEY')
            )
        else:
            # init_chat_model로 모델 초기화
            model = init_chat_model(
                model=model_config['model'],
                model_provider=model_config['provider'],
                temperature=model_config.get('temperature', 0.7),
                max_tokens=model_config.get('max_tokens', 2000)
            )

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
