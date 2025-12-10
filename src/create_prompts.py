"""Langfuse Prompts 생성 스크립트

Langfuse UI의 'Prompts → Run Experiment' 기능을 사용하기 위해
프롬프트 템플릿을 생성합니다.
"""
from langfuse import Langfuse
from src.logger import get_logger


def create_function_calling_prompt():
    """함수 호출용 프롬프트 템플릿 생성"""
    langfuse = Langfuse()
    logger = get_logger(__name__)

    prompt_name = "function-calling-benchmark"

    # 프롬프트 템플릿 (변수: {{query}})
    prompt_template = """{{query}}

You have access to the following tools. Use them if needed to answer the question."""

    try:
        # Langfuse에 프롬프트 생성
        langfuse.create_prompt(
            name=prompt_name,
            prompt=prompt_template,
            config={
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": 2000
            },
            labels=["benchmark", "function-calling"]
        )

        logger.info(f"✅ Prompt '{prompt_name}' 생성 완료")
        print(f"✅ Prompt '{prompt_name}' 생성 완료")
        print(f"\n프롬프트 템플릿:")
        print(prompt_template)
        print(f"\n변수: {{{{query}}}}")

    except Exception as e:
        if "already exists" in str(e).lower():
            logger.info(f"ℹ️  Prompt '{prompt_name}' 이미 존재")
            print(f"ℹ️  Prompt '{prompt_name}' 이미 존재")
        else:
            logger.error(f"❌ Prompt 생성 실패: {e}")
            raise

    langfuse.flush()

    return prompt_name


def main():
    """메인 실행"""
    print("=" * 70)
    print("Langfuse Prompts 생성")
    print("=" * 70)
    print()

    try:
        prompt_name = create_function_calling_prompt()

        print("\n" + "=" * 70)
        print("✅ 완료!")
        print("=" * 70)
        print(f"\n다음 단계:")
        print(f"1. Langfuse 대시보드 접속: https://cloud.langfuse.com")
        print(f"2. 'Prompts' 메뉴에서 '{prompt_name}' 확인")
        print(f"3. 'Datasets' 메뉴에서 'function_calling_benchmark' 선택")
        print(f"4. 'Run Experiment' 버튼 클릭")
        print(f"5. Prompt: '{prompt_name}' 선택")
        print(f"6. Dataset 변수 매핑: query → query")
        print(f"7. 실행!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
