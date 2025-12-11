"""일반 질의 프롬프트를 Langfuse Dataset으로 동기화

Langfuse UI에 등록된 프롬프트나 로컬 프롬프트 파일을
Dataset으로 변환하여 실험 비교를 가능하게 합니다.

사용법:
    python -m src.query_dataset_sync
"""
from typing import Dict, Any, List
from pathlib import Path

from langfuse import Langfuse

from src.logger import get_logger


class QueryDatasetSync:
    """일반 질의 프롬프트 Dataset 동기화 클래스"""

    def __init__(self, dataset_name: str = "query_benchmark"):
        """
        Args:
            dataset_name: 생성할 Dataset 이름
        """
        self.dataset_name = dataset_name
        self.logger = get_logger(__name__)
        self.langfuse = Langfuse()

    def sync_from_prompts_directory(self):
        """
        prompts/ 디렉토리의 텍스트 파일을 Dataset으로 동기화
        """
        prompts_dir = Path(__file__).parent.parent / "prompts"

        if not prompts_dir.exists():
            self.logger.warning(f"프롬프트 디렉토리가 없습니다: {prompts_dir}")
            return

        self.logger.info(f"프롬프트 디렉토리 스캔 중: {prompts_dir}")

        # Dataset 생성 (이미 존재하면 무시됨)
        try:
            self.langfuse.create_dataset(name=self.dataset_name)
            self.logger.info(f"✅ Dataset '{self.dataset_name}' 생성")
        except Exception as e:
            self.logger.info(f"Dataset '{self.dataset_name}' 이미 존재 (또는 생성 오류: {e})")

        # .txt 파일 찾기
        prompt_files = list(prompts_dir.glob("*.txt"))

        if not prompt_files:
            self.logger.warning(f"프롬프트 파일이 없습니다: {prompts_dir}/*.txt")
            return

        self.logger.info(f"발견된 프롬프트 파일: {len(prompt_files)}개")

        # 각 파일을 Dataset Item으로 생성
        for prompt_file in prompt_files:
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompt_content = f.read().strip()

                prompt_name = prompt_file.stem  # 확장자 제외한 파일명

                self._create_dataset_item(
                    name=prompt_name,
                    query=prompt_content
                )

                self.logger.info(f"  ✓ {prompt_name}")

            except Exception as e:
                self.logger.error(f"  ✗ {prompt_file.name}: {e}")

        print("\n" + "=" * 60)
        print(f"✅ Dataset '{self.dataset_name}' 동기화 완료")
        print(f"   총 {len(prompt_files)}개 프롬프트")
        print("=" * 60)

    def sync_from_langfuse_prompts(self, prompt_names: List[str]):
        """
        Langfuse에 등록된 프롬프트를 가져와서 Dataset으로 동기화

        Args:
            prompt_names: Langfuse에 등록된 프롬프트 이름 리스트
        """
        self.logger.info(f"Langfuse 프롬프트 가져오기 시작")

        # Dataset 생성
        try:
            self.langfuse.create_dataset(name=self.dataset_name)
            self.logger.info(f"✅ Dataset '{self.dataset_name}' 생성")
        except Exception as e:
            self.logger.info(f"Dataset '{self.dataset_name}' 이미 존재")

        # 각 프롬프트를 Langfuse에서 가져와서 Dataset Item으로 생성
        success_count = 0
        for prompt_name in prompt_names:
            try:
                # Langfuse에서 프롬프트 가져오기 (production 버전)
                prompt_obj = self.langfuse.get_prompt(prompt_name)
                prompt_content = prompt_obj.prompt

                self._create_dataset_item(
                    name=prompt_name,
                    query=prompt_content
                )
                self.logger.info(f"  ✓ {prompt_name} (버전: {prompt_obj.version})")
                success_count += 1

            except Exception as e:
                self.logger.error(f"  ✗ {prompt_name}: {e}")

        print("\n" + "=" * 60)
        print(f"✅ Dataset '{self.dataset_name}' 동기화 완료")
        print(f"   총 {success_count}/{len(prompt_names)}개 프롬프트")
        print("=" * 60)

    def sync_from_manual_list(self, prompts: List[Dict[str, str]]):
        """
        수동으로 정의한 프롬프트 리스트를 Dataset으로 동기화

        Args:
            prompts: [{"name": str, "query": str}, ...] 형태의 리스트
        """
        self.logger.info(f"수동 프롬프트 리스트 동기화 시작")

        # Dataset 생성
        try:
            self.langfuse.create_dataset(name=self.dataset_name)
            self.logger.info(f"✅ Dataset '{self.dataset_name}' 생성")
        except Exception as e:
            self.logger.info(f"Dataset '{self.dataset_name}' 이미 존재")

        # 각 프롬프트를 Dataset Item으로 생성
        for prompt in prompts:
            try:
                self._create_dataset_item(
                    name=prompt["name"],
                    query=prompt["query"]
                )
                self.logger.info(f"  ✓ {prompt['name']}")
            except Exception as e:
                self.logger.error(f"  ✗ {prompt['name']}: {e}")

        print("\n" + "=" * 60)
        print(f"✅ Dataset '{self.dataset_name}' 동기화 완료")
        print(f"   총 {len(prompts)}개 프롬프트")
        print("=" * 60)

    def _create_dataset_item(self, name: str, query: str):
        """
        단일 프롬프트를 Dataset Item으로 생성

        Args:
            name: 프롬프트 이름
            query: 프롬프트 내용
        """
        # Input 구성
        input_data = {
            "query": query,
            "type": "general_query"  # 함수 호출과 구분
        }

        # Expected output은 없음 (일반 질의는 정답이 없으므로)
        expected_output = {}

        # Metadata
        metadata = {
            "prompt_name": name,
            "benchmark_type": "query"
        }

        # Dataset Item 생성
        self.langfuse.create_dataset_item(
            dataset_name=self.dataset_name,
            input=input_data,
            expected_output=expected_output,
            metadata=metadata
        )


def main():
    """메인 실행 함수"""
    import sys

    print("=" * 60)
    print("일반 질의 프롬프트 Dataset 동기화")
    print("=" * 60)

    # Dataset 이름 (커맨드라인 인자로 받을 수도 있음)
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "query_benchmark"

    syncer = QueryDatasetSync(dataset_name=dataset_name)

    # Langfuse에 등록된 프롬프트 이름 리스트
    prompt_names = ["짜장면벤치", "괭벤치"]

    # Langfuse에서 프롬프트를 가져와서 Dataset으로 동기화
    syncer.sync_from_langfuse_prompts(prompt_names)

    print("\n다음 명령으로 벤치마크 실행:")
    print(f"python -m src.langfuse_runner gpt-4o-mini {dataset_name}")


if __name__ == "__main__":
    main()
