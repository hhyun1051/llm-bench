"""LLM 벤치마크 메인 스크립트"""
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


def main():
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
        default="config/models.yaml",
        help="모델 설정 파일 경로 (기본: config/models.yaml)"
    )
    parser.add_argument(
        "--prompts-dir",
        type=str,
        default="prompts",
        help="프롬프트 디렉토리 경로 (기본: prompts)"
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
    parser.add_argument(
        "--enable-langfuse",
        action="store_true",
        help="Langfuse 추적 활성화 (환경변수에 LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY 필요)"
    )

    args = parser.parse_args()

    # 초기화
    try:
        import yaml
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Langfuse 초기화
        langfuse = LangfuseIntegration(enabled=args.enable_langfuse)

        model_runner = ModelRunner(args.config, langfuse_integration=langfuse)

        # 타입에 따라 필요한 로더만 초기화
        prompt_loader = None
        function_loader = None
        function_runner = None

        if args.type in ["query", "all"]:
            prompt_loader = PromptLoader(args.prompts_dir)

        if args.type in ["function-call", "all"]:
            function_loader = FunctionCallLoader()
            function_runner = FunctionCallRunner(config, langfuse_integration=langfuse)

    except Exception as e:
        print(f"초기화 오류: {e}")
        return 1

    # 목록 출력 옵션
    if args.list_models:
        print("사용 가능한 모델:")
        for model in model_runner.list_available_models():
            info = model_runner.get_model_info(model)
            print(f"  - {model} ({info['provider']}: {info['model']})")
        return 0

    if args.list_prompts:
        if prompt_loader:
            print("사용 가능한 프롬프트:")
            for prompt in prompt_loader.list_prompts():
                print(f"  - {prompt}")
        else:
            print("프롬프트 로더가 초기화되지 않았습니다. --type query 또는 --type all로 실행하세요.")
        return 0

    if args.list_functions:
        if function_loader:
            print("사용 가능한 펑션 콜링 시나리오:")
            for scenario in function_loader.list_scenarios():
                print(f"  - {scenario}")
        else:
            print("펑션 콜링 로더가 초기화되지 않았습니다. --type function-call 또는 --type all로 실행하세요.")
        return 0

    # 실행할 모델 결정
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        models = model_runner.list_available_models()

    # 실행할 프롬프트/시나리오 결정
    prompts = {}
    scenarios = {}

    if args.type in ["query", "all"]:
        if args.prompts:
            prompt_names = [p.strip() for p in args.prompts.split(",")]
            prompts = {name: prompt_loader.load_prompt(name) for name in prompt_names}
        else:
            prompts = prompt_loader.load_all_prompts()

    if args.type in ["function-call", "all"]:
        if args.functions:
            scenario_names = [s.strip() for s in args.functions.split(",")]
            scenarios = {name: function_loader.load_scenario(name) for name in scenario_names}
        else:
            scenarios = function_loader.load_all_scenarios()

    # 벤치마크 실행
    print(f"\n{'='*60}")
    print(f"LLM 벤치마크 시작")
    print(f"{'='*60}")
    print(f"타입: {args.type}")
    print(f"모델: {', '.join(models)}")
    if prompts:
        print(f"프롬프트: {', '.join(prompts.keys())}")
    if scenarios:
        print(f"펑션 콜링 시나리오: {', '.join(scenarios.keys())}")
    print(f"{'='*60}\n")

    # Langfuse 세션 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = langfuse.create_session(f"benchmark_{timestamp}") if langfuse.enabled else None

    # 일반 질의 벤치마크
    if prompts:
        print(f"\n{'='*60}")
        print("일반 질의 벤치마크")
        print(f"{'='*60}")

        for idx, (prompt_name, prompt_text) in enumerate(prompts.items(), 1):
            print(f"\n[{idx}/{len(prompts)}] 프롬프트: {prompt_name}")
            print(f"{'-'*60}")

            benchmark_result = {
                "prompt_name": prompt_name,
                "prompt": prompt_text,
                "results": []
            }

            # Langfuse trace 생성
            trace_id = None
            if langfuse.enabled:
                trace_id = langfuse.create_trace(
                    name=f"query_{prompt_name}",
                    benchmark_type="query",
                    metadata={"prompt_name": prompt_name, "session_id": session_id}
                )

            # 각 모델에 대해
            for model_name in models:
                print(f"  실행 중: {model_name}... ", end="", flush=True)

                try:
                    result = model_runner.run_prompt(model_name, prompt_text)

                    if 'error' in result:
                        print(f"❌ 오류: {result['error']}")
                        benchmark_result["results"].append({
                            "model": model_name,
                            "error": result['error'],
                            "metadata": result.get('metadata', {})
                        })

                        # Langfuse 로깅
                        if trace_id:
                            langfuse.log_query_result(
                                trace_id=trace_id,
                                model_name=model_name,
                                prompt_name=prompt_name,
                                prompt=prompt_text,
                                response="",
                                metadata=result.get('metadata', {}),
                                error=result['error']
                            )
                    else:
                        response_time = result['metadata'].get('response_time_ms', 0)
                        tokens = result['metadata'].get('total_tokens', 'N/A')
                        print(f"✓ 완료 ({response_time}ms, {tokens} tokens)")

                        benchmark_result["results"].append({
                            "model": model_name,
                            "response": result['response'],
                            "metadata": result['metadata']
                        })

                        # Langfuse 로깅
                        if trace_id:
                            langfuse.log_query_result(
                                trace_id=trace_id,
                                model_name=model_name,
                                prompt_name=prompt_name,
                                prompt=prompt_text,
                                response=result['response'],
                                metadata=result['metadata']
                            )

                except Exception as e:
                    print(f"❌ 예외 발생: {e}")
                    benchmark_result["results"].append({
                        "model": model_name,
                        "error": str(e),
                        "metadata": {}
                    })

    # 펑션 콜링 벤치마크
    if scenarios:
        print(f"\n{'='*60}")
        print("펑션 콜링 벤치마크")
        print(f"{'='*60}")

        for idx, (scenario_name, scenario) in enumerate(scenarios.items(), 1):
            print(f"\n[{idx}/{len(scenarios)}] 시나리오: {scenario_name}")
            print(f"설명: {scenario['description']}")
            print(f"프롬프트: {scenario['prompt']}")
            print(f"{'-'*60}")

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
            if langfuse.enabled:
                trace_id = langfuse.create_trace(
                    name=f"function_call_{scenario_name}",
                    benchmark_type="function-call",
                    metadata={
                        "scenario_name": scenario_name,
                        "tools": scenario['tools'],
                        "session_id": session_id
                    }
                )

            # 각 모델에 대해
            for model_name in models:
                print(f"  실행 중: {model_name}... ", end="", flush=True)

                try:
                    result = function_runner.run_scenario(
                        model_name=model_name,
                        prompt=scenario['prompt'],
                        tools=scenario['tool_objects'],
                        expected_tool_calls=scenario.get('expected_tool_calls')
                    )

                    if 'error' in result:
                        print(f"❌ 오류: {result['error']}")
                        benchmark_result["results"].append({
                            "model": model_name,
                            "error": result['error'],
                            "metadata": result.get('metadata', {})
                        })

                        # Langfuse 로깅
                        if trace_id:
                            langfuse.log_function_call_result(
                                trace_id=trace_id,
                                model_name=model_name,
                                scenario_name=scenario_name,
                                prompt=scenario['prompt'],
                                response="",
                                tool_calls=[],
                                evaluation={},
                                metadata=result.get('metadata', {}),
                                error=result['error']
                            )
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

                        benchmark_result["results"].append({
                            "model": model_name,
                            "response": result['response'],
                            "tool_calls": result['tool_calls'],
                            "metadata": result['metadata'],
                            "evaluation": result['evaluation']
                        })

                        # Langfuse 로깅
                        if trace_id:
                            langfuse.log_function_call_result(
                                trace_id=trace_id,
                                model_name=model_name,
                                scenario_name=scenario_name,
                                prompt=scenario['prompt'],
                                response=result['response'],
                                tool_calls=result['tool_calls'],
                                evaluation=result['evaluation'],
                                metadata=result['metadata']
                            )

                except Exception as e:
                    print(f"❌ 예외 발생: {e}")
                    benchmark_result["results"].append({
                        "model": model_name,
                        "error": str(e),
                        "metadata": {}
                    })

    # Langfuse 플러시 (모든 로그 전송)
    if langfuse.enabled:
        print(f"\n{'='*60}")
        print("Langfuse로 데이터 전송 중...")
        langfuse.flush()
        print(f"✓ Langfuse 전송 완료")

        dashboard_url = langfuse.get_dashboard_url()
        if dashboard_url:
            print(f"  대시보드: {dashboard_url}")
        print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())