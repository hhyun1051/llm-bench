"""Langfuse Dataset 동기화 모듈

로컬 시나리오 파일들을 Langfuse Dataset으로 업로드하여
UI에서 실험을 실행할 수 있도록 합니다.
"""
import os
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from src.function_call_loader import FunctionCallLoader
from src.logger import get_logger


class DatasetSync:
    """Langfuse Dataset 동기화 클래스"""

    def __init__(self, dataset_name: str = "function_calling_benchmark"):
        """
        Args:
            dataset_name: Langfuse에 생성할 Dataset 이름
        """
        self.dataset_name = dataset_name
        self.logger = get_logger(__name__)
        self.langfuse = None

        # Langfuse 초기화
        self._init_langfuse()

    def _init_langfuse(self):
        """Langfuse 클라이언트 초기화"""
        try:
            from langfuse import Langfuse
            self.langfuse = Langfuse()
            self.logger.info("Langfuse 연결 성공")
        except Exception as e:
            self.logger.error(f"Langfuse 초기화 실패: {e}")
            raise

    def sync_scenarios(self) -> Dict[str, Any]:
        """
        로컬 시나리오를 Langfuse Dataset으로 동기화

        Returns:
            동기화 결과 정보
        """
        loader = FunctionCallLoader()
        scenarios = loader.load_all_scenarios()

        self.logger.info(f"로드된 시나리오: {len(scenarios)}개")

        # Dataset 생성 또는 가져오기
        dataset = self._create_or_get_dataset()

        # 시나리오를 Dataset Item으로 추가
        synced_count = 0
        skipped_count = 0

        for scenario_name, scenario in scenarios.items():
            try:
                self._create_dataset_item(scenario_name, scenario)
                synced_count += 1
                self.logger.info(f"✓ Synced: {scenario_name}")
            except Exception as e:
                skipped_count += 1
                self.logger.warning(f"✗ Skipped {scenario_name}: {e}")

        # Flush to ensure all items are sent
        self.langfuse.flush()

        result = {
            "dataset_name": self.dataset_name,
            "total_scenarios": len(scenarios),
            "synced": synced_count,
            "skipped": skipped_count,
            "timestamp": datetime.now().isoformat()
        }

        return result

    def _create_or_get_dataset(self):
        """Dataset 생성 또는 기존 Dataset 가져오기"""
        try:
            # Dataset 생성 시도
            self.langfuse.create_dataset(
                name=self.dataset_name,
                description="LLM 벤치마크 - 함수 호출 시나리오 테스트 세트",
                metadata={
                    "source": "llm-bench",
                    "type": "function_calling",
                    "version": "1.0",
                    "created_at": datetime.now().isoformat()
                }
            )
            self.logger.info(f"✓ Dataset '{self.dataset_name}' 생성됨")
        except Exception as e:
            # 이미 존재하는 경우
            self.logger.info(f"Dataset '{self.dataset_name}' 이미 존재 (기존 항목에 추가됨)")

        return self.dataset_name

    def _create_dataset_item(self, scenario_name: str, scenario: Dict[str, Any]):
        """
        단일 시나리오를 Dataset Item으로 생성

        Args:
            scenario_name: 시나리오 이름
            scenario: 시나리오 정보
        """
        # Input 구성 - Langfuse Prompts와 호환되도록 변수 형태로 저장
        # prompt는 메타데이터로 이동하고, 대신 query 변수 사용
        input_data = {
            "query": scenario["prompt"],  # 프롬프트를 query 변수로 저장
            "tools": scenario["tools"],
            "description": scenario.get("description", "")
        }

        # Expected output 구성
        expected_output = {
            "tool_calls": scenario.get("expected_tool_calls", [])
        }

        # Metadata
        metadata = {
            "scenario_name": scenario_name,
            "source_file": f"function_calls/scenarios/{scenario_name}.yaml",
            "original_prompt": scenario["prompt"]  # 원본 프롬프트 보관
        }

        # Dataset Item 생성
        self.langfuse.create_dataset_item(
            dataset_name=self.dataset_name,
            input=input_data,
            expected_output=expected_output,
            metadata=metadata
        )

    def list_dataset_items(self) -> list:
        """
        현재 Dataset의 모든 아이템 조회

        Returns:
            Dataset 아이템 리스트
        """
        try:
            dataset = self.langfuse.get_dataset(self.dataset_name)
            items = list(dataset.items)
            self.logger.info(f"Dataset '{self.dataset_name}'에 {len(items)}개 아이템 존재")
            return items
        except Exception as e:
            self.logger.error(f"Dataset 조회 실패: {e}")
            return []


def main():
    """메인 실행 함수"""
    import sys

    # 로거 설정
    logger = get_logger(__name__)

    print("=" * 70)
    print("Langfuse Dataset 동기화")
    print("=" * 70)

    # Dataset 이름 (커맨드라인 인자 지원)
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "function_calling_benchmark"

    try:
        # 동기화 실행
        syncer = DatasetSync(dataset_name=dataset_name)

        print(f"\n📤 시나리오를 '{dataset_name}' Dataset으로 업로드 중...\n")

        result = syncer.sync_scenarios()

        # 결과 출력
        print("\n" + "=" * 70)
        print("동기화 완료!")
        print("=" * 70)
        print(f"Dataset 이름: {result['dataset_name']}")
        print(f"총 시나리오: {result['total_scenarios']}")
        print(f"동기화됨: {result['synced']}")
        print(f"스킵됨: {result['skipped']}")
        print(f"시간: {result['timestamp']}")

        # Dataset 아이템 목록 출력
        print(f"\n📋 Dataset 아이템 확인 중...")
        items = syncer.list_dataset_items()

        if items:
            print(f"\n현재 Dataset에 등록된 아이템:")
            for i, item in enumerate(items, 1):
                scenario_name = item.metadata.get("scenario_name", "Unknown")
                prompt = item.input.get("prompt", "")[:50]
                print(f"  {i}. {scenario_name}: {prompt}...")

        # Langfuse UI 안내
        print(f"\n" + "=" * 70)
        print("✅ 다음 단계:")
        print("=" * 70)
        print("1. Langfuse 대시보드 접속:")
        print("   → https://cloud.langfuse.com")
        print(f"\n2. 'Datasets' 메뉴에서 '{dataset_name}' 확인")
        print("\n3. Dataset을 선택하고 'Run Experiment' 버튼 클릭")
        print("\n4. 또는 SDK로 실험 실행:")
        print(f"   → python src/langfuse_runner.py")
        print("=" * 70)

    except Exception as e:
        logger.error(f"동기화 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
