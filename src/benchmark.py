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
from src.result_processor import ResultProcessor


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
        "--results-dir",
        type=str,
        default="results",
        help="결과 저장 디렉토리 경로 (기본: results)"
    )

    args = parser.parse_args()

    # 초기화
    try:
        model_runner = ModelRunner(args.config)
        prompt_loader = PromptLoader(args.prompts_dir)
        result_processor = ResultProcessor(args.results_dir)
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
        print("사용 가능한 프롬프트:")
        for prompt in prompt_loader.list_prompts():
            print(f"  - {prompt}")
        return 0

    # 실행할 모델 결정
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        models = model_runner.list_available_models()

    # 실행할 프롬프트 결정
    if args.prompts:
        prompt_names = [p.strip() for p in args.prompts.split(",")]
        prompts = {name: prompt_loader.load_prompt(name) for name in prompt_names}
    else:
        prompts = prompt_loader.load_all_prompts()

    # 벤치마크 실행
    print(f"\n{'='*60}")
    print(f"LLM 벤치마크 시작")
    print(f"{'='*60}")
    print(f"모델: {', '.join(models)}")
    print(f"프롬프트: {', '.join(prompts.keys())}")
    print(f"{'='*60}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "models": models,
        "benchmarks": []
    }

    # 각 프롬프트에 대해
    for idx, (prompt_name, prompt_text) in enumerate(prompts.items(), 1):
        print(f"\n[{idx}/{len(prompts)}] 프롬프트: {prompt_name}")
        print(f"{'-'*60}")

        benchmark_result = {
            "prompt_name": prompt_name,
            "prompt": prompt_text,
            "results": []
        }

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
                else:
                    response_time = result['metadata'].get('response_time_ms', 0)
                    tokens = result['metadata'].get('total_tokens', 'N/A')
                    print(f"✓ 완료 ({response_time}ms, {tokens} tokens)")

                    benchmark_result["results"].append({
                        "model": model_name,
                        "response": result['response'],
                        "metadata": result['metadata']
                    })

            except Exception as e:
                print(f"❌ 예외 발생: {e}")
                benchmark_result["results"].append({
                    "model": model_name,
                    "error": str(e),
                    "metadata": {}
                })

        results["benchmarks"].append(benchmark_result)

    # 결과 저장
    print(f"\n{'='*60}")
    print("결과 저장 중...")
    output_dir = result_processor.save_results(results, timestamp)
    print(f"✓ 결과 저장 완료: {output_dir}")
    print(f"  - JSON: {output_dir}/results.json")
    print(f"  - Markdown: {output_dir}/summary.md")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())