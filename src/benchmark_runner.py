"""벤치마크 실행 로직"""
from typing import Dict, Any, List
from datetime import datetime

from src.constants import SEPARATOR_LONG, SEPARATOR_SHORT


class BenchmarkRunner:
    """벤치마크 실행을 담당하는 클래스"""

    def __init__(self, model_runner, function_runner=None, langfuse=None):
        """
        Args:
            model_runner: ModelRunner 인스턴스
            function_runner: FunctionCallRunner 인스턴스 (선택)
            langfuse: LangfuseIntegration 인스턴스 (선택)
        """
        self.model_runner = model_runner
        self.function_runner = function_runner
        self.langfuse = langfuse

    def _log_query_to_langfuse(
        self,
        trace_id: str,
        model_name: str,
        prompt_name: str,
        prompt: str,
        result: Dict[str, Any]
    ):
        """
        일반 질의 결과를 Langfuse에 로깅 (헬퍼 메서드)

        Args:
            trace_id: Trace ID
            model_name: 모델 이름
            prompt_name: 프롬프트 이름
            prompt: 프롬프트 내용
            result: 실행 결과 딕셔너리
        """
        if not trace_id or not self.langfuse:
            return

        error = result.get('error')
        response = result.get('response', '')
        metadata = result.get('metadata', {})

        self.langfuse.log_query_result(
            trace_id=trace_id,
            model_name=model_name,
            prompt_name=prompt_name,
            prompt=prompt,
            response=response,
            metadata=metadata,
            error=error
        )

    def _log_function_call_to_langfuse(
        self,
        trace_id: str,
        model_name: str,
        scenario_name: str,
        prompt: str,
        result: Dict[str, Any]
    ):
        """
        펑션 콜링 결과를 Langfuse에 로깅 (헬퍼 메서드)

        Args:
            trace_id: Trace ID
            model_name: 모델 이름
            scenario_name: 시나리오 이름
            prompt: 프롬프트
            result: 실행 결과 딕셔너리
        """
        if not trace_id or not self.langfuse:
            return

        error = result.get('error')
        response = result.get('response', '')
        tool_calls = result.get('tool_calls', [])
        evaluation = result.get('evaluation', {})
        metadata = result.get('metadata', {})

        self.langfuse.log_function_call_result(
            trace_id=trace_id,
            model_name=model_name,
            scenario_name=scenario_name,
            prompt=prompt,
            response=response,
            tool_calls=tool_calls,
            evaluation=evaluation,
            metadata=metadata,
            error=error
        )

    def run_query_benchmarks(
        self,
        models: List[str],
        prompts: Dict[str, str],
        session_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        일반 질의 벤치마크 실행

        Args:
            models: 실행할 모델 목록
            prompts: {프롬프트명: 프롬프트 내용} 딕셔너리
            session_id: Langfuse 세션 ID

        Returns:
            벤치마크 결과 리스트
        """
        results = []

        print(f"\n{SEPARATOR_LONG}")
        print("일반 질의 벤치마크")
        print(f"{SEPARATOR_LONG}")

        for idx, (prompt_name, prompt_text) in enumerate(prompts.items(), 1):
            print(f"\n[{idx}/{len(prompts)}] 프롬프트: {prompt_name}")
            print(f"{SEPARATOR_SHORT}")

            benchmark_result = {
                "prompt_name": prompt_name,
                "prompt": prompt_text,
                "results": []
            }

            # Langfuse trace 생성
            trace_id = None
            if self.langfuse and self.langfuse.enabled:
                trace_id = self.langfuse.create_trace(
                    name=f"query_{prompt_name}",
                    benchmark_type="query",
                    metadata={"prompt_name": prompt_name, "session_id": session_id}
                )

            # 각 모델에 대해 실행
            for model_name in models:
                result = self._run_single_query(
                    model_name,
                    prompt_name,
                    prompt_text,
                    trace_id
                )
                benchmark_result["results"].append(result)

            results.append(benchmark_result)

        return results

    def _run_single_query(
        self,
        model_name: str,
        prompt_name: str,
        prompt_text: str,
        trace_id: str = None
    ) -> Dict[str, Any]:
        """단일 모델로 질의 실행"""
        print(f"  실행 중: {model_name}... ", end="", flush=True)

        try:
            result = self.model_runner.run_prompt(model_name, prompt_text)

            if 'error' in result:
                print(f"❌ 오류: {result['error']}")
            else:
                response_time = result['metadata'].get('response_time_ms', 0)
                tokens = result['metadata'].get('total_tokens', 'N/A')
                print(f"✓ 완료 ({response_time}ms, {tokens} tokens)")

            # 결과 딕셔너리 생성
            model_result = {
                "model": model_name,
                **result
            }

            # Langfuse 로깅 (헬퍼 메서드 사용)
            self._log_query_to_langfuse(
                trace_id=trace_id,
                model_name=model_name,
                prompt_name=prompt_name,
                prompt=prompt_text,
                result=result
            )

            return model_result

        except Exception as e:
            print(f"❌ 예외 발생: {e}")
            return {
                "model": model_name,
                "error": str(e),
                "metadata": {}
            }

    def run_function_call_benchmarks(
        self,
        models: List[str],
        scenarios: Dict[str, Dict[str, Any]],
        session_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        펑션 콜링 벤치마크 실행

        Args:
            models: 실행할 모델 목록
            scenarios: {시나리오명: 시나리오 정보} 딕셔너리
            session_id: Langfuse 세션 ID

        Returns:
            벤치마크 결과 리스트
        """
        if not self.function_runner:
            raise ValueError("FunctionCallRunner가 초기화되지 않았습니다")

        results = []

        print(f"\n{SEPARATOR_LONG}")
        print("펑션 콜링 벤치마크")
        print(f"{SEPARATOR_LONG}")

        for idx, (scenario_name, scenario) in enumerate(scenarios.items(), 1):
            print(f"\n[{idx}/{len(scenarios)}] 시나리오: {scenario_name}")
            print(f"설명: {scenario['description']}")
            print(f"프롬프트: {scenario['prompt']}")
            print(f"{SEPARATOR_SHORT}")

            benchmark_result = {
                "scenario_name": scenario_name,
                "description": scenario['description'],
                "prompt": scenario['prompt'],
                "tools": scenario['tools'],
                "expected_tool_calls": scenario.get('expected_tool_calls', []),
                "results": []
            }

            # Langfuse trace 생성
            trace_id = None
            if self.langfuse and self.langfuse.enabled:
                trace_id = self.langfuse.create_trace(
                    name=f"function_call_{scenario_name}",
                    benchmark_type="function-call",
                    metadata={
                        "scenario_name": scenario_name,
                        "tools": scenario['tools'],
                        "session_id": session_id
                    }
                )

            # 각 모델에 대해 실행
            for model_name in models:
                result = self._run_single_function_call(
                    model_name,
                    scenario_name,
                    scenario,
                    trace_id
                )
                benchmark_result["results"].append(result)

            results.append(benchmark_result)

        # 모델별 평균 점수 계산 및 Langfuse에 기록
        # 약간의 지연을 주어 요약이 마지막에 표시되도록 함
        import time
        time.sleep(0.5)
        self._log_model_summary_scores(results, models, session_id)

        return results

    def _log_model_summary_scores(
        self,
        results: List[Dict[str, Any]],
        models: List[str],
        session_id: str = None
    ):
        """
        모델별 평균 점수를 계산하고 Langfuse에 기록

        Args:
            results: 벤치마크 결과 리스트
            models: 실행한 모델 목록
            session_id: Langfuse 세션 ID
        """
        if not self.langfuse or not self.langfuse.enabled:
            return

        # 모델별 점수 집계
        model_scores = {model: [] for model in models}
        model_times = {model: [] for model in models}

        for benchmark_result in results:
            for result in benchmark_result.get("results", []):
                model = result.get("model")
                evaluation = result.get("evaluation", {})
                metadata = result.get("metadata", {})

                if evaluation.get("evaluated"):
                    score = evaluation.get("score", 0)
                    model_scores[model].append(score)

                response_time = metadata.get("response_time_ms", 0)
                if response_time > 0:
                    model_times[model].append(response_time)

        # 각 모델에 대한 요약 trace 생성 및 스코어 기록
        for model in models:
            scores = model_scores.get(model, [])
            times = model_times.get(model, [])

            if not scores:
                continue

            avg_score = sum(scores) / len(scores)
            avg_time = sum(times) / len(times) if times else 0

            # 모델 요약 trace 생성
            summary_trace_id = self.langfuse.create_trace(
                name=f"[Summary] {model}",
                benchmark_type="function-call-summary",
                metadata={
                    "model": model,
                    "session_id": session_id,
                    "num_scenarios": len(scores),
                    "avg_response_time_ms": int(avg_time),
                    "avg_score": avg_score
                }
            )

            if summary_trace_id:
                # 모델 요약 generation 생성 (trace name 설정용)
                from langfuse.types import TraceContext

                summary_gen = self.langfuse.langfuse.start_generation(
                    trace_context=TraceContext(trace_id=summary_trace_id),
                    name=f"[Summary] {model}",
                    model=model,
                    input=f"Function calling benchmark summary for {model}",
                    metadata={
                        "num_scenarios": len(scores),
                        "avg_score": avg_score,
                        "avg_response_time_ms": int(avg_time)
                    }
                )

                summary_gen.update(
                    output=f"Model {model} completed {len(scores)} scenarios with average score {avg_score:.2f}"
                )
                summary_gen.end()

                # 평균 점수 기록
                self.langfuse.langfuse.create_score(
                    trace_id=summary_trace_id,
                    name="avg_accuracy",
                    value=avg_score,
                    comment=f"Average score across {len(scores)} scenarios"
                )

                # 평균 응답 시간 기록
                if avg_time > 0:
                    self.langfuse.langfuse.create_score(
                        trace_id=summary_trace_id,
                        name="avg_response_time",
                        value=avg_time,
                        comment=f"Average response time in milliseconds"
                    )

    def _run_single_function_call(
        self,
        model_name: str,
        scenario_name: str,
        scenario: Dict[str, Any],
        trace_id: str = None
    ) -> Dict[str, Any]:
        """단일 모델로 펑션 콜 실행"""
        print(f"  실행 중: {model_name}... ", end="", flush=True)

        try:
            result = self.function_runner.run_scenario(
                model_name=model_name,
                prompt=scenario['prompt'],
                tools=scenario['tool_objects'],
                expected_tool_calls=scenario.get('expected_tool_calls')
            )

            if 'error' in result:
                print(f"❌ 오류: {result['error']}")
            else:
                response_time = result['metadata'].get('response_time_ms', 0)
                num_calls = result['metadata'].get('num_tool_calls', 0)
                evaluation = result.get('evaluation', {})
                score = evaluation.get('score', 0)

                status = "✓"
                if evaluation.get('evaluated'):
                    if score == 1.0:
                        status = "✓✓"
                    elif score >= 0.5:
                        status = "✓~"
                    else:
                        status = "✗"

                print(f"{status} 완료 ({response_time}ms, {num_calls} calls, score: {score:.1f})")

            # 결과 딕셔너리 생성
            model_result = {
                "model": model_name,
                **result
            }

            # Langfuse 로깅 (헬퍼 메서드 사용)
            self._log_function_call_to_langfuse(
                trace_id=trace_id,
                model_name=model_name,
                scenario_name=scenario_name,
                prompt=scenario['prompt'],
                result=result
            )

            return model_result

        except Exception as e:
            print(f"❌ 예외 발생: {e}")
            return {
                "model": model_name,
                "error": str(e),
                "metadata": {}
            }
