"""프롬프트 로딩 모듈"""
import os
from pathlib import Path
from typing import Dict, List


class PromptLoader:
    """프롬프트 파일을 로드하는 클래스"""

    def __init__(self, prompts_dir: str = "prompts"):
        """
        Args:
            prompts_dir: 프롬프트 파일이 있는 디렉토리 경로
        """
        self.prompts_dir = Path(prompts_dir)
        if not self.prompts_dir.exists():
            raise ValueError(f"프롬프트 디렉토리를 찾을 수 없습니다: {prompts_dir}")

    def load_all_prompts(self) -> Dict[str, str]:
        """
        모든 .txt 파일을 로드하여 딕셔너리로 반환

        Returns:
            {파일명(확장자 제외): 프롬프트 내용} 형식의 딕셔너리
        """
        prompts = {}

        for file_path in self.prompts_dir.glob("*.txt"):
            prompt_name = file_path.stem  # 확장자 제외한 파일명
            with open(file_path, 'r', encoding='utf-8') as f:
                prompts[prompt_name] = f.read().strip()

        if not prompts:
            raise ValueError(f"프롬프트 파일을 찾을 수 없습니다: {self.prompts_dir}")

        return prompts

    def load_prompt(self, prompt_name: str) -> str:
        """
        특정 프롬프트 파일 로드

        Args:
            prompt_name: 파일명 (확장자 제외)

        Returns:
            프롬프트 내용
        """
        file_path = self.prompts_dir / f"{prompt_name}.txt"

        if not file_path.exists():
            raise ValueError(f"프롬프트 파일을 찾을 수 없습니다: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def list_prompts(self) -> List[str]:
        """
        사용 가능한 모든 프롬프트 이름 목록 반환

        Returns:
            프롬프트 이름 리스트
        """
        return [f.stem for f in self.prompts_dir.glob("*.txt")]
