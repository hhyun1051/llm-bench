"""LLM 벤치마크 메인 스크립트 (리팩토링 버전)"""
import argparse
from datetime import datetime
from pathlib import Path
import sys
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 상위 디렉토리를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompt_loader import PromptLoader
from src.model_runner import ModelRunner
from src.function_call_loader import FunctionCallLoader
from src.function_call_runner import FunctionCallRunner
from src.langfuse_integration import LangfuseIntegration
from src.benchmark_runner import BenchmarkRunner
from src.config_validator import load_and_validate_config, ConfigValidationError
from src.logger import setup_logger
from src.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_PROMPTS_DIR,
    SEPARATOR_LONG
)


def parse_arguments() -> argparse.Namespace:
    """커맨드 라인 인자 파싱"""
    parser = argparse.ArgumentParser(description="LLM 벤치마크 도구")
    parser.add_argument(
        "--models",
        type=str,
        help="테스트할 모델 (쉼표로 구분, 예: gpt-4o,claude-3.5-sonnet). 생략 시 모든 모델 실행"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        help="실행할 프롬프트 (쉼표로 구분). 생략 시 모든 프롬프트 실행"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="사용 가능한 모델 목록 출력"
    )
    parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="사용 가능한 프롬프트 목록 출력"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"모델 설정 파일 경로 (기본: {DEFAULT_CONFIG_PATH})"
    )
    parser.add_argument(
        "--prompts-dir",
        type=str,
        default=DEFAULT_PROMPTS_DIR,
        help=f"프롬프트 디렉토리 경로 (기본: {DEFAULT_PROMPTS_DIR})"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["query", "function-call", "all"],
        default="query",
        help="벤치마크 타입: query (일반 질의), function-call (펑션 콜링), all (둘 다)"
    )
    parser.add_argument(
        "--functions",
        type=str,
        help="실행할 펑션 콜링 시나리오 (쉼표로 구분). 생략 시 모든 시나리오 실행"
    )
    parser.add_argument(
        "--list-functions",
        action="store_true",
        help="사용 가능한 펑션 콜링 시나리오 목록 출력"
    )

    return parser.parse_args()


def initialize_components(args: argparse.Namespace):
    """컴포넌트 초기화"""
    # 설정 파일 검증 및 로드
    try:
        config = load_and_validate_config(args.config)
    except (ConfigValidationError, FileNotFoundError) as e:
        raise ValueError(f"설정 파일 오류: {e}")

    # Langfuse 초기화 (항상 활성화)
    langfuse = LangfuseIntegration(enabled=True)

    # 모델 러너 초기화 (이미 로드된 config 딕셔너리 전달)
    model_runner = ModelRunner(config, langfuse_integration=langfuse)

    # 타입에 따라 필요한 로더만 초기화
    prompt_loader = None
    function_loader = None
    function_runner = None

    if args.type in ["query", "all"]:
        prompt_loader = PromptLoader(args.prompts_dir)

    if args.type in ["function-call", "all"]:
        function_loader = FunctionCallLoader()
        function_runner = FunctionCallRunner(config, langfuse_integration=langfuse)

    return {
        'config': config,
        'langfuse': langfuse,
        'model_runner': model_runner,
        'prompt_loader': prompt_loader,
        'function_loader': function_loader,
        'function_runner': function_runner
    }


def handle_list_commands(args: argparse.Namespace, components: dict) -> bool:
    """목록 출력 커맨드 처리. 처리했으면 True 반환"""
    if args.list_models:
        print("사용 가능한 모델:")
        for model in components['model_runner'].list_available_models():
            info = components['model_runner'].get_model_info(model)
            print(f"  - {model} ({info['provider']}: {info['model']})")
        return True

    if args.list_prompts:
        if components['prompt_loader']:
            print("사용 가능한 프롬프트:")
            for prompt in components['prompt_loader'].list_prompts():
                print(f"  - {prompt}")
        else:
            print("프롬프트 로더가 초기화되지 않았습니다. --type query 또는 --type all로 실행하세요.")
        return True

    if args.list_functions:
        if components['function_loader']:
            print("사용 가능한 펑션 콜링 시나리오:")
            for scenario in components['function_loader'].list_scenarios():
                print(f"  - {scenario}")
        else:
            print("펑션 콜링 로더가 초기화되지 않았습니다. --type function-call 또는 --type all로 실행하세요.")
        return True

    return False


def determine_targets(args: argparse.Namespace, components: dict):
    """실행할 모델, 프롬프트, 시나리오 결정"""
    # 실행할 모델 결정
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        models = components['model_runner'].list_available_models()

    # 실행할 프롬프트/시나리오 결정
    prompts = {}
    scenarios = {}

    if args.type in ["query", "all"]:
        prompt_loader = components['prompt_loader']
        if args.prompts:
            prompt_names = [p.strip() for p in args.prompts.split(",")]
            prompts = {name: prompt_loader.load_prompt(name) for name in prompt_names}
        else:
            prompts = prompt_loader.load_all_prompts()

    if args.type in ["function-call", "all"]:
        function_loader = components['function_loader']
        if args.functions:
            scenario_names = [s.strip() for s in args.functions.split(",")]
            scenarios = {name: function_loader.load_scenario(name) for name in scenario_names}
        else:
            scenarios = function_loader.load_all_scenarios()

    return models, prompts, scenarios


def print_benchmark_header(args: argparse.Namespace, models: list, prompts: dict, scenarios: dict):
    """벤치마크 시작 헤더 출력"""
    print(f"\n{SEPARATOR_LONG}")
    print("LLM 벤치마크 시작")
    print(f"{SEPARATOR_LONG}")
    print(f"타입: {args.type}")
    print(f"모델: {', '.join(models)}")
    if prompts:
        print(f"프롬프트: {', '.join(prompts.keys())}")
    if scenarios:
        print(f"펑션 콜링 시나리오: {', '.join(scenarios.keys())}")
    print(f"{SEPARATOR_LONG}\n")


def main():
    """메인 함수"""
    # 로거 설정
    setup_logger()

    args = parse_arguments()

    # 초기화
    try:
        components = initialize_components(args)
    except Exception as e:
        print(f"초기화 오류: {e}")
        return 1

    # 목록 출력 옵션 처리
    if handle_list_commands(args, components):
        return 0

    # 실행 대상 결정
    models, prompts, scenarios = determine_targets(args, components)

    # 헤더 출력
    print_benchmark_header(args, models, prompts, scenarios)

    # Langfuse 세션 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    langfuse = components['langfuse']
    session_id = langfuse.create_session(f"benchmark_{timestamp}") if langfuse.enabled else None

    # 벤치마크 러너 생성
    runner = BenchmarkRunner(
        model_runner=components['model_runner'],
        function_runner=components['function_runner'],
        langfuse=langfuse
    )

    # 일반 질의 벤치마크
    if prompts:
        runner.run_query_benchmarks(models, prompts, session_id)

    # 펑션 콜링 벤치마크
    if scenarios:
        runner.run_function_call_benchmarks(models, scenarios, session_id)

    # Langfuse 플러시
    if langfuse.enabled:
        print(f"\n{SEPARATOR_LONG}")
        print("Langfuse로 데이터 전송 중...")
        langfuse.flush()
        print("✓ Langfuse 전송 완료")

        dashboard_url = langfuse.get_dashboard_url()
        if dashboard_url:
            print(f"  대시보드: {dashboard_url}")
        print(f"{SEPARATOR_LONG}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
