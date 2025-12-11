"""Langfuse Dataset Run 실행 모듈

Langfuse Dataset을 기반으로 벤치마크를 실행하고 자동 평가를 수행하는 모듈입니다.
함수 호출 벤치마크와 일반 질의 벤치마크를 모두 지원합니다.

사용법:
    # 함수 호출 벤치마크 (기본)
    python -m src.langfuse_runner gpt-4o-mini

    # 일반 질의 벤치마크
    python -m src.langfuse_runner gpt-4o-mini query_benchmark

    # 시스템 프롬프트 지정
    python -m src.langfuse_runner gpt-4o-mini --system-prompt function-calling-system-v1

    # 시스템 프롬프트 특정 버전 지정
    python -m src.langfuse_runner gpt-4o-mini --system-prompt function-calling-system-v1 --system-prompt-version 2
"""
import time
import yaml
from typing import Dict, Any, List
from pathlib import Path

from langfuse import Langfuse, Evaluation

from src.function_call_runner import FunctionCallRunner
from src.model_utils import create_model
from src.logger import get_logger
from src.constants import DEFAULT_SYSTEM_PROMPT


# Config 캐시 (매번 로드하지 않도록)
_config_cache = None


def _load_config() -> Dict[str, Any]:
    """모델 설정 로드 (캐싱)"""
    global _config_cache
    if _config_cache is None:
        config_path = Path(__file__).parent.parent / "config" / "models.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            _config_cache = yaml.safe_load(f)
    return _config_cache


# Tool 함수 매핑 (동적 import)
def get_tool_functions():
    """사용 가능한 모든 tool 함수를 딕셔너리로 반환"""
    from function_calls.tools import (
        get_weather,
        calculate,
        search_database,
        send_email,
        get_current_time,
        convert_currency
    )

    return {
        "get_weather": get_weather,
        "calculate": calculate,
        "search_database": search_database,
        "send_email": send_email,
        "get_current_time": get_current_time,
        "convert_currency": convert_currency
    }


def run_general_query_task(
    item,
    model_name: str = "gpt-4o-mini",
    system_prompt: str = None
) -> Dict[str, Any]:
    """
    Langfuse Dataset Item에 대해 일반 질의 실행

    LLM에 프롬프트를 전달하고 응답을 받습니다.

    Args:
        item: Langfuse Dataset Item
            - item.input: {"query": str, "type": "general_query"}
        model_name: 사용할 모델 이름
        system_prompt: 시스템 프롬프트 (None이면 기본값 사용)

    Returns:
        실행 결과 딕셔너리 (response, metadata 포함)
    """
    logger = get_logger(__name__)

    # Config 로드 (캐싱됨)
    config = _load_config()

    # Input 파싱
    query = item.input.get("query", "")
    prompt_name = item.metadata.get("prompt_name", "unknown") if hasattr(item, 'metadata') else "unknown"

    logger.info(f"실행 중: {prompt_name}")
    logger.info(f"쿼리: {query}")

    # 시스템 프롬프트 결정
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    # 모델 초기화
    try:
        model = create_model(config['models'][model_name])
    except Exception as e:
        logger.error(f"모델 초기화 오류: {e}")
        return {
            "error": f"모델 초기화 오류: {str(e)}",
            "metadata": {"response_time_ms": 0}
        }

    start_time = time.time()

    try:
        # 메시지 구성
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        # LLM 호출
        response = model.invoke(messages)

        end_time = time.time()
        response_time_ms = int((end_time - start_time) * 1000)

        # 응답 텍스트 추출
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)

        logger.info(f"응답 완료 ({response_time_ms}ms)")

        # 결과 구성
        return {
            "response": response_text,
            "metadata": {
                "response_time_ms": response_time_ms,
                "prompt_name": prompt_name
            }
        }

    except Exception as e:
        end_time = time.time()
        response_time_ms = int((end_time - start_time) * 1000)
        logger.error(f"일반 질의 실행 오류: {e}")

        return {
            "error": str(e),
            "metadata": {
                "response_time_ms": response_time_ms
            }
        }


def run_function_calling_task(
    item,
    model_name: str = "gpt-4o-mini",
    system_prompt: str = None
) -> Dict[str, Any]:
    """
    Langfuse Dataset Item에 대해 함수 호출 벤치마크 실행

    LangChain Agent를 사용하여 실제로 함수를 호출하고,
    결과를 평가하여 반환합니다.

    Args:
        item: Langfuse Dataset Item
            - item.input: {"query": str, "tools": List[str], "description": str}
            - item.expected_output: {"tool_calls": List[Dict]}
        model_name: 사용할 모델 이름
        system_prompt: 시스템 프롬프트 (None이면 기본값 사용)

    Returns:
        실행 결과 딕셔너리 (response, tool_calls, evaluation, metadata 포함)
    """
    logger = get_logger(__name__)

    # Config 로드 (캐싱됨)
    config = _load_config()

    # Input 파싱 (query 또는 prompt 필드 지원)
    prompt = item.input.get("query") or item.input.get("prompt", "")
    tool_names = item.input.get("tools", [])
    description = item.input.get("description", "")

    logger.info(f"실행 중: {description}")
    logger.info(f"프롬프트: {prompt}")
    logger.info(f"도구: {tool_names}")

    # Tool 함수 가져오기
    tool_map = get_tool_functions()
    tools = []
    for tool_name in tool_names:
        if tool_name in tool_map:
            tools.append(tool_map[tool_name])
        else:
            logger.warning(f"Tool을 찾을 수 없음: {tool_name}")

    if not tools:
        return {
            "error": "사용 가능한 tool이 없습니다",
            "tool_calls": [],
            "evaluation": {"evaluated": False}
        }

    # Expected output 파싱
    expected_tool_calls = item.expected_output.get("tool_calls", [])

    # Function Call Runner 실행
    try:
        runner = FunctionCallRunner(config=config)
        result = runner.run_scenario(
            model_name=model_name,
            prompt=prompt,
            tools=tools,
            expected_tool_calls=expected_tool_calls,
            system_prompt=system_prompt
        )

        # 결과 구성
        output = {
            "response": result.get("response", ""),
            "tool_calls": result.get("tool_calls", []),
            "evaluation": result.get("evaluation", {}),
            "metadata": result.get("metadata", {})
        }

        if "error" in result:
            output["error"] = result["error"]

        return output

    except Exception as e:
        logger.error(f"실행 오류: {e}")
        return {
            "error": str(e),
            "tool_calls": [],
            "evaluation": {"evaluated": False}
        }


def accuracy_evaluator(*, input, output, expected_output, **kwargs) -> Evaluation:
    """
    함수 호출 정확도 평가 함수

    Langfuse Dataset Run에서 자동으로 호출되어 각 결과를 평가합니다.

    Args:
        input: Dataset item의 input
        output: run_function_calling_task의 반환값
        expected_output: Dataset item의 expected_output

    Returns:
        Evaluation 객체 (점수와 코멘트 포함)
    """
    logger = get_logger(__name__)

    # 에러가 있으면 0점
    if "error" in output:
        return Evaluation(
            name="accuracy",
            value=0.0,
            comment=f"❌ 실행 오류: {output['error']}"
        )

    # Expected tool calls 확인
    expected_calls = expected_output.get("tool_calls", [])
    if not expected_calls:
        return Evaluation(
            name="accuracy",
            value=1.0,
            comment="✓ 예상 tool call 없음 (평가 불가)"
        )

    # Actual tool calls 확인
    actual_calls = output.get("tool_calls", [])
    if not actual_calls:
        return Evaluation(
            name="accuracy",
            value=0.0,
            comment="✗ Tool이 호출되지 않음"
        )

    # 첫 번째 tool call 비교
    expected_call = expected_calls[0]
    actual_call = actual_calls[0]

    expected_tool = expected_call.get("tool")
    actual_tool = actual_call.get("name")  # LangChain에서는 'name' 필드 사용

    expected_args = expected_call.get("expected_args", {})
    actual_args = actual_call.get("args", {})

    # Tool 이름 비교
    correct_tool = expected_tool == actual_tool

    # 인자 비교 (대소문자 무시)
    correct_args = True
    failed_params = []

    for key, expected_value in expected_args.items():
        actual_value = actual_args.get(key)
        if str(actual_value).lower() != str(expected_value).lower():
            correct_args = False
            failed_params.append(f"{key}:expected={expected_value},actual={actual_value}")

    # 점수 계산 (기존 벤치마크와 동일한 로직)
    score = 0.0
    if correct_tool:
        score += 0.5
    if correct_args:
        score += 0.5

    # 코멘트 생성
    tool_status = "✓" if correct_tool else "✗"
    args_status = "✓" if correct_args else "✗"

    comment_parts = [
        f"Tool: {tool_status} ({actual_tool})",
        f"Args: {args_status}"
    ]

    if failed_params:
        comment_parts.append(f"Failed: {', '.join(failed_params)}")

    comment = " | ".join(comment_parts)

    logger.info(f"평가 완료 - Score: {score:.1f}, {comment}")

    return Evaluation(
        name="accuracy",
        value=score,
        comment=comment
    )


def correct_tool_evaluator(*, input, output, expected_output, **kwargs) -> Evaluation:
    """Tool 선택 정확도만 평가"""
    if "error" in output or not output.get("tool_calls"):
        return Evaluation(name="correct_tool", value=0.0)

    expected_calls = expected_output.get("tool_calls", [])
    if not expected_calls:
        return Evaluation(name="correct_tool", value=1.0)

    expected_tool = expected_calls[0].get("tool")
    actual_tool = output["tool_calls"][0].get("name")

    value = 1.0 if expected_tool == actual_tool else 0.0
    return Evaluation(name="correct_tool", value=value)


def correct_args_evaluator(*, input, output, expected_output, **kwargs) -> Evaluation:
    """인자 정확도만 평가"""
    if "error" in output or not output.get("tool_calls"):
        return Evaluation(name="correct_args", value=0.0)

    expected_calls = expected_output.get("tool_calls", [])
    if not expected_calls:
        return Evaluation(name="correct_args", value=1.0)

    expected_args = expected_calls[0].get("expected_args", {})
    actual_args = output["tool_calls"][0].get("args", {})

    correct = all(
        str(actual_args.get(k, "")).lower() == str(v).lower()
        for k, v in expected_args.items()
    )

    return Evaluation(name="correct_args", value=1.0 if correct else 0.0)


class LangfuseExperimentRunner:
    """Langfuse Dataset 실험 실행 클래스"""

    def __init__(self, dataset_name: str = "function_calling_benchmark"):
        """
        Args:
            dataset_name: 실행할 Dataset 이름
        """
        self.dataset_name = dataset_name
        self.logger = get_logger(__name__)
        self.langfuse = Langfuse()

    def _detect_dataset_type(self, dataset) -> str:
        """
        Dataset 타입을 감지합니다.

        첫 번째 아이템의 input.type 필드를 확인하여
        'general_query' 또는 'function_calling'을 반환합니다.

        Args:
            dataset: Langfuse Dataset 객체

        Returns:
            "query" 또는 "function_calling"
        """
        try:
            # Dataset의 첫 번째 아이템 가져오기
            items = list(dataset.items)
            if not items:
                self.logger.warning("Dataset이 비어있습니다. 기본값 'function_calling' 사용")
                return "function_calling"

            first_item = items[0]
            item_type = first_item.input.get("type", "")

            if item_type == "general_query":
                return "query"
            else:
                # tools 필드가 있으면 function_calling
                if "tools" in first_item.input:
                    return "function_calling"
                else:
                    # 기본값
                    return "function_calling"

        except Exception as e:
            self.logger.warning(f"Dataset 타입 감지 실패: {e}. 기본값 'function_calling' 사용")
            return "function_calling"

    def run_experiment(
        self,
        model_name: str = "gpt-4o-mini",
        run_name: str = None,
        run_description: str = None,
        use_evaluators: bool = True,
        max_concurrency: int = 1,
        system_prompt_name: str = None,
        system_prompt_version: int = None
    ) -> Any:
        """
        Dataset에 대해 실험 실행

        Args:
            model_name: 테스트할 모델
            run_name: 실행 이름 (없으면 자동 생성)
            run_description: 실행 설명
            use_evaluators: 자동 평가 사용 여부
            max_concurrency: 동시 실행 개수
            system_prompt_name: Langfuse에서 가져올 시스템 프롬프트 이름
            system_prompt_version: 시스템 프롬프트 버전 (None이면 production)

        Returns:
            실험 결과
        """
        # Dataset 로드
        try:
            dataset = self.langfuse.get_dataset(self.dataset_name)
        except Exception as e:
            self.logger.error(f"Dataset을 찾을 수 없음: {e}")
            raise

        # Dataset 타입 감지 (첫 번째 아이템의 input.type 확인)
        dataset_type = self._detect_dataset_type(dataset)
        self.logger.info(f"Dataset 타입: {dataset_type}")

        # 시스템 프롬프트 로드 (지정된 경우)
        system_prompt = None
        if system_prompt_name:
            try:
                prompt_obj = self.langfuse.get_prompt(
                    system_prompt_name,
                    version=system_prompt_version
                )
                system_prompt = prompt_obj.prompt
                self.logger.info(f"시스템 프롬프트 로드: {system_prompt_name} (버전: {prompt_obj.version})")
            except Exception as e:
                self.logger.error(f"시스템 프롬프트를 찾을 수 없음: {e}")
                raise

        # Run name 생성
        if not run_name:
            from datetime import datetime
            run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if not run_description:
            benchmark_type_str = "일반 질의" if dataset_type == "query" else "함수 호출"
            run_description = f"{benchmark_type_str} 벤치마크 with {model_name}"

        self.logger.info(f"실험 시작: {run_name}")
        self.logger.info(f"Dataset: {self.dataset_name}")
        self.logger.info(f"모델: {model_name}")

        # Dataset 타입에 따라 Evaluators 및 Task 함수 설정
        if dataset_type == "query":
            # 일반 질의는 evaluator 없음 (정답이 없으므로)
            evaluators = []

            # Task 함수
            def task(item):
                return run_general_query_task(
                    item,
                    model_name=model_name,
                    system_prompt=system_prompt
                )

            benchmark_type = "general_query"
        else:
            # 함수 호출 벤치마크
            evaluators = []
            if use_evaluators:
                evaluators = [
                    accuracy_evaluator,
                    correct_tool_evaluator,
                    correct_args_evaluator
                ]

            # Task 함수
            def task(item):
                return run_function_calling_task(
                    item,
                    model_name=model_name,
                    system_prompt=system_prompt
                )

            benchmark_type = "function_calling"

        # 실험 실행
        metadata = {
            "model": model_name,
            "benchmark_type": benchmark_type
        }
        if system_prompt_name:
            metadata["system_prompt_name"] = system_prompt_name
            if system_prompt_version:
                metadata["system_prompt_version"] = system_prompt_version

        result = dataset.run_experiment(
            name=run_name,
            description=run_description,
            task=task,
            evaluators=evaluators,
            max_concurrency=max_concurrency,
            metadata=metadata
        )

        self.logger.info("실험 완료!")
        print("\n" + "=" * 70)
        print("실험 결과 요약")
        print("=" * 70)
        print(result.format())
        print("=" * 70)

        return result

    def list_runs(self):
        """Dataset의 모든 실행 목록 조회"""
        # Langfuse SDK에 runs 조회 API가 있다면 사용
        # 없으면 UI에서 확인하도록 안내
        self.logger.info(f"Langfuse UI에서 '{self.dataset_name}' Dataset Runs를 확인하세요")
        self.logger.info("URL: https://cloud.langfuse.com")


def main():
    """메인 실행 함수"""
    import sys
    import argparse

    print("=" * 70)
    print("Langfuse Dataset Experiment Runner")
    print("=" * 70)

    # 커맨드라인 인자 파싱
    parser = argparse.ArgumentParser(
        description="Langfuse Dataset을 사용한 함수 호출 벤치마크"
    )
    parser.add_argument(
        "model_name",
        nargs="?",
        default="gpt-4o-mini",
        help="테스트할 모델 이름"
    )
    parser.add_argument(
        "dataset_name",
        nargs="?",
        default="function_calling_benchmark",
        help="사용할 Dataset 이름"
    )
    parser.add_argument(
        "--system-prompt",
        dest="system_prompt_name",
        help="Langfuse에서 가져올 시스템 프롬프트 이름"
    )
    parser.add_argument(
        "--system-prompt-version",
        dest="system_prompt_version",
        type=int,
        help="시스템 프롬프트 버전 (없으면 production)"
    )

    args = parser.parse_args()

    print(f"\n모델: {args.model_name}")
    print(f"Dataset: {args.dataset_name}")
    if args.system_prompt_name:
        version_str = f"v{args.system_prompt_version}" if args.system_prompt_version else "production"
        print(f"시스템 프롬프트: {args.system_prompt_name} ({version_str})")
    print("")

    try:
        # 실험 실행
        runner = LangfuseExperimentRunner(dataset_name=args.dataset_name)
        result = runner.run_experiment(
            model_name=args.model_name,
            system_prompt_name=args.system_prompt_name,
            system_prompt_version=args.system_prompt_version
        )

        print("\n✅ 실험이 성공적으로 완료되었습니다!")
        print("\nLangfuse 대시보드에서 결과를 확인하세요:")
        print("→ https://cloud.langfuse.com")

    except Exception as e:
        print(f"\n❌ 실험 실행 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
