"""Langfuse 통합 모듈"""
import os
from typing import Dict, Any, Optional
from datetime import datetime

from src.logger import get_logger
from src.constants import (
    ENV_LANGFUSE_PUBLIC_KEY,
    ENV_LANGFUSE_SECRET_KEY,
    ENV_LANGFUSE_HOST,
    LANGFUSE_DEFAULT_HOST
)


class LangfuseIntegration:
    """Langfuse 추적 및 분석 통합 클래스"""

    def __init__(self, enabled: bool = True):
        """
        Args:
            enabled: Langfuse 사용 여부
        """
        self.logger = get_logger(__name__)
        self.enabled = enabled and self._check_credentials()
        self.langfuse = None

        if self.enabled:
            try:
                from langfuse import Langfuse
                self.langfuse = Langfuse(
                    public_key=os.getenv(ENV_LANGFUSE_PUBLIC_KEY),
                    secret_key=os.getenv(ENV_LANGFUSE_SECRET_KEY),
                    host=os.getenv(ENV_LANGFUSE_HOST, LANGFUSE_DEFAULT_HOST)
                )
                self.logger.info("Langfuse 연결 성공")
            except Exception as e:
                self.logger.warning(f"Langfuse 초기화 실패: {e}")
                self.enabled = False

    def _check_credentials(self) -> bool:
        """Langfuse 자격증명 확인"""
        public_key = os.getenv(ENV_LANGFUSE_PUBLIC_KEY)
        secret_key = os.getenv(ENV_LANGFUSE_SECRET_KEY)

        if not public_key or not secret_key:
            return False
        return True

    def create_trace(
        self,
        name: str,
        benchmark_type: str,
        metadata: Dict[str, Any] = None
    ) -> Optional[str]:
        """
        새로운 trace 생성

        Args:
            name: Trace 이름
            benchmark_type: 벤치마크 타입 (query or function-call)
            metadata: 추가 메타데이터

        Returns:
            Trace ID (string) 또는 None
        """
        if not self.enabled:
            return None

        try:
            # Langfuse SDK의 올바른 메서드 사용
            trace_id = self.langfuse.create_trace_id()

            # Trace는 첫 번째 generation/event에서 자동 생성됨
            # 여기서는 trace_id만 생성하고 반환
            return trace_id
        except Exception as e:
            self.logger.warning(f"Trace 생성 실패: {e}")
            return None

    def log_query_result(
        self,
        trace_id: str,
        model_name: str,
        prompt_name: str,
        prompt: str,
        response: str,
        metadata: Dict[str, Any],
        error: Optional[str] = None
    ):
        """
        일반 질의 결과 로깅

        Args:
            trace_id: Trace ID
            model_name: 모델 이름
            prompt_name: 프롬프트 이름
            prompt: 프롬프트 내용
            response: 응답 내용
            metadata: 메타데이터 (응답 시간, 토큰 등)
            error: 에러 메시지 (있는 경우)
        """
        if not self.enabled:
            return

        try:
            from langfuse.types import TraceContext

            # Generation 시작 및 종료
            generation = self.langfuse.start_generation(
                trace_context=TraceContext(trace_id=trace_id),
                name=f"{prompt_name}_{model_name}",
                model=model_name,
                input=prompt,
                metadata={
                    "prompt_name": prompt_name,
                    "response_time_ms": metadata.get("response_time_ms"),
                    "error": error
                }
            )

            # Generation 업데이트 및 종료
            usage_dict = {}
            if metadata.get("input_tokens"):
                usage_dict["input"] = metadata.get("input_tokens")
            if metadata.get("output_tokens"):
                usage_dict["output"] = metadata.get("output_tokens")
            if metadata.get("total_tokens"):
                usage_dict["total"] = metadata.get("total_tokens")

            if response and not error:
                generation.update(output=response)
            if usage_dict:
                generation.update(usage_details=usage_dict)

            generation.end()

            # 에러가 없으면 성공 스코어
            if not error:
                self.langfuse.create_score(
                    trace_id=trace_id,
                    name="success",
                    value=1.0
                )

        except Exception as e:
            self.logger.warning(f"Query 결과 로깅 실패: {e}")

    def log_function_call_result(
        self,
        trace_id: str,
        model_name: str,
        scenario_name: str,
        prompt: str,
        response: str,
        tool_calls: list,
        evaluation: Dict[str, Any],
        metadata: Dict[str, Any],
        error: Optional[str] = None
    ):
        """
        펑션 콜링 결과 로깅

        Args:
            trace_id: Trace ID
            model_name: 모델 이름
            scenario_name: 시나리오 이름
            prompt: 프롬프트
            response: 최종 응답
            tool_calls: 호출된 툴 목록
            evaluation: 평가 결과
            metadata: 메타데이터
            error: 에러 메시지 (있는 경우)
        """
        if not self.enabled:
            return

        try:
            from langfuse.types import TraceContext

            # Generation 시작 및 종료
            generation = self.langfuse.start_generation(
                trace_context=TraceContext(trace_id=trace_id),
                name=f"{scenario_name}_{model_name}",
                model=model_name,
                input=prompt,
                metadata={
                    "scenario_name": scenario_name,
                    "response_time_ms": metadata.get("response_time_ms"),
                    "num_tool_calls": metadata.get("num_tool_calls"),
                    "tool_calls": tool_calls,
                    "error": error
                }
            )

            # Generation 업데이트 및 종료
            if response and not error:
                generation.update(output=response)

            generation.end()

            # 평가 스코어 로깅
            if not error and evaluation.get("evaluated"):
                # 전체 정확도 스코어
                self.langfuse.create_score(
                    trace_id=trace_id,
                    name="accuracy",
                    value=evaluation.get("score", 0),
                    comment=evaluation.get("message", "")
                )

                # Tool 선택 정확도
                self.langfuse.create_score(
                    trace_id=trace_id,
                    name="correct_tool",
                    value=1.0 if evaluation.get("correct_tool") else 0.0
                )

                # 파라미터 정확도
                self.langfuse.create_score(
                    trace_id=trace_id,
                    name="correct_args",
                    value=1.0 if evaluation.get("correct_args") else 0.0
                )

        except Exception as e:
            self.logger.warning(f"Function call 결과 로깅 실패: {e}")

    def create_session(self, session_name: str) -> Optional[str]:
        """
        벤치마크 세션 생성

        Args:
            session_name: 세션 이름 (보통 timestamp)

        Returns:
            Session ID
        """
        if not self.enabled:
            return None

        try:
            # Langfuse는 자동으로 세션을 그룹화하므로
            # session_name을 반환
            return session_name
        except Exception as e:
            self.logger.warning(f"Session 생성 실패: {e}")
            return None

    def flush(self):
        """모든 로그를 Langfuse로 전송"""
        if self.enabled and self.langfuse:
            try:
                self.langfuse.flush()
            except Exception as e:
                self.logger.warning(f"Flush 실패: {e}")

    def get_dashboard_url(self) -> Optional[str]:
        """Langfuse 대시보드 URL 반환"""
        if not self.enabled:
            return None

        host = os.getenv(ENV_LANGFUSE_HOST, LANGFUSE_DEFAULT_HOST)
        return f"{host}/project"
