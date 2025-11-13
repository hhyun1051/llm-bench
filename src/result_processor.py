"""결과 저장 및 처리 모듈"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


class ResultProcessor:
    """벤치마크 결과를 저장하고 포맷하는 클래스"""

    def __init__(self, results_dir: str = "results"):
        """
        Args:
            results_dir: 결과를 저장할 디렉토리 경로
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_results(self, results: Dict[str, Any], timestamp: str = None) -> str:
        """
        결과를 JSON과 Markdown 형식으로 저장

        Args:
            results: 벤치마크 결과 데이터
            timestamp: 타임스탬프 (없으면 자동 생성)

        Returns:
            결과가 저장된 디렉토리 경로
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 타임스탬프별 디렉토리 생성
        output_dir = self.results_dir / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON 저장
        json_path = output_dir / "results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # Markdown 저장
        md_path = output_dir / "summary.md"
        markdown_content = self._generate_markdown(results)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        return str(output_dir)

    def _generate_markdown(self, results: Dict[str, Any]) -> str:
        """결과를 Markdown 형식으로 변환"""
        lines = []
        lines.append("# LLM Benchmark 결과\n")
        lines.append(f"**실행 시각:** {results.get('timestamp', 'N/A')}\n")
        lines.append(f"**실행 모델:** {', '.join(results.get('models', []))}\n")
        lines.append(f"**프롬프트 수:** {len(results.get('benchmarks', []))}\n")
        lines.append("\n---\n")

        # 각 벤치마크별 결과
        for benchmark in results.get('benchmarks', []):
            prompt_name = benchmark['prompt_name']
            lines.append(f"\n## {prompt_name}\n")
            lines.append(f"**프롬프트:**\n```\n{benchmark['prompt'][:200]}{'...' if len(benchmark['prompt']) > 200 else ''}\n```\n")

            # 모델별 응답
            for model_result in benchmark['results']:
                model_name = model_result['model']
                lines.append(f"\n### {model_name}\n")

                if 'error' in model_result:
                    lines.append(f"**오류 발생:** {model_result['error']}\n")
                else:
                    lines.append(f"**응답:**\n```\n{model_result['response']}\n```\n")

                # 메타데이터
                metadata = model_result.get('metadata', {})
                lines.append(f"\n**메타데이터:**\n")
                lines.append(f"- 응답 시간: {metadata.get('response_time_ms', 'N/A')}ms\n")

                if 'input_tokens' in metadata:
                    lines.append(f"- 입력 토큰: {metadata['input_tokens']}\n")
                if 'output_tokens' in metadata:
                    lines.append(f"- 출력 토큰: {metadata['output_tokens']}\n")
                if 'total_tokens' in metadata:
                    lines.append(f"- 총 토큰: {metadata['total_tokens']}\n")

            lines.append("\n---\n")

        # 통계 요약
        lines.append("\n## 통계 요약\n")
        lines.append(self._generate_statistics(results))

        return "".join(lines)

    def _generate_statistics(self, results: Dict[str, Any]) -> str:
        """통계 정보 생성"""
        lines = []
        stats_by_model = {}

        for benchmark in results.get('benchmarks', []):
            for model_result in benchmark['results']:
                model_name = model_result['model']
                if model_name not in stats_by_model:
                    stats_by_model[model_name] = {
                        'total_time': 0,
                        'total_tokens': 0,
                        'count': 0,
                        'errors': 0
                    }

                if 'error' in model_result:
                    stats_by_model[model_name]['errors'] += 1
                else:
                    metadata = model_result.get('metadata', {})
                    stats_by_model[model_name]['total_time'] += metadata.get('response_time_ms', 0)
                    stats_by_model[model_name]['total_tokens'] += metadata.get('total_tokens', 0)
                    stats_by_model[model_name]['count'] += 1

        lines.append("| 모델 | 평균 응답 시간 | 총 토큰 | 에러 수 |\n")
        lines.append("|------|---------------|---------|--------|\n")

        for model_name, stats in stats_by_model.items():
            avg_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
            lines.append(f"| {model_name} | {avg_time:.0f}ms | {stats['total_tokens']} | {stats['errors']} |\n")

        return "".join(lines)