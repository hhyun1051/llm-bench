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
        lines.append(f"**벤치마크 타입:** {results.get('type', 'N/A')}\n")
        lines.append(f"**실행 모델:** {', '.join(results.get('models', []))}\n")

        query_count = len(results.get('query_benchmarks', []))
        function_count = len(results.get('function_call_benchmarks', []))

        if query_count > 0:
            lines.append(f"**일반 질의 수:** {query_count}\n")
        if function_count > 0:
            lines.append(f"**펑션 콜링 시나리오 수:** {function_count}\n")

        lines.append("\n---\n")

        # 일반 질의 벤치마크 결과
        query_benchmarks = results.get('query_benchmarks', [])
        if query_benchmarks:
            lines.append("\n# 일반 질의 벤치마크\n")

            for benchmark in query_benchmarks:
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

        # 펑션 콜링 벤치마크 결과
        function_benchmarks = results.get('function_call_benchmarks', [])
        if function_benchmarks:
            lines.append("\n# 펑션 콜링 벤치마크\n")

            for benchmark in function_benchmarks:
                scenario_name = benchmark['scenario_name']
                lines.append(f"\n## {scenario_name}\n")
                lines.append(f"**설명:** {benchmark['description']}\n")
                lines.append(f"**프롬프트:** {benchmark['prompt']}\n")
                lines.append(f"**사용 가능한 Tools:** {', '.join(benchmark['tools'])}\n")

                expected = benchmark.get('expected_tool_calls', [])
                if expected:
                    lines.append(f"**예상 Tool 호출:** {expected[0].get('tool', 'N/A')}\n")

                lines.append("\n")

                # 모델별 결과
                for model_result in benchmark['results']:
                    model_name = model_result['model']
                    lines.append(f"\n### {model_name}\n")

                    if 'error' in model_result:
                        lines.append(f"**오류 발생:** {model_result['error']}\n")
                    else:
                        # Tool 호출 정보
                        tool_calls = model_result.get('tool_calls', [])
                        if tool_calls:
                            lines.append(f"**호출된 Tools:**\n")
                            for tc in tool_calls:
                                lines.append(f"- `{tc['name']}` with args: `{tc['args']}`\n")
                        else:
                            lines.append(f"**호출된 Tools:** 없음\n")

                        lines.append(f"\n**최종 응답:**\n```\n{model_result['response']}\n```\n")

                        # 평가 결과
                        evaluation = model_result.get('evaluation', {})
                        if evaluation.get('evaluated'):
                            lines.append(f"\n**평가 결과:**\n")
                            lines.append(f"- 점수: {evaluation.get('score', 0):.1f} / 1.0\n")
                            lines.append(f"- 올바른 Tool 선택: {'✓' if evaluation.get('correct_tool') else '✗'}\n")
                            lines.append(f"- 올바른 파라미터: {'✓' if evaluation.get('correct_args') else '✗'}\n")

                            details = evaluation.get('details', {})
                            if details:
                                lines.append(f"\n**상세:**\n")
                                for key, value in details.items():
                                    lines.append(f"- {key}: {value}\n")

                    # 메타데이터
                    metadata = model_result.get('metadata', {})
                    lines.append(f"\n**메타데이터:**\n")
                    lines.append(f"- 응답 시간: {metadata.get('response_time_ms', 'N/A')}ms\n")
                    lines.append(f"- Tool 호출 수: {metadata.get('num_tool_calls', 0)}\n")

                lines.append("\n---\n")

        # 통계 요약
        lines.append("\n# 통계 요약\n")
        lines.append(self._generate_statistics(results))

        return "".join(lines)

    def _generate_statistics(self, results: Dict[str, Any]) -> str:
        """통계 정보 생성"""
        lines = []
        stats_by_model = {}

        # 일반 질의 통계
        for benchmark in results.get('query_benchmarks', []):
            for model_result in benchmark['results']:
                model_name = model_result['model']
                if model_name not in stats_by_model:
                    stats_by_model[model_name] = {
                        'query_total_time': 0,
                        'query_total_tokens': 0,
                        'query_count': 0,
                        'query_errors': 0,
                        'fc_total_time': 0,
                        'fc_total_score': 0.0,
                        'fc_count': 0,
                        'fc_errors': 0
                    }

                if 'error' in model_result:
                    stats_by_model[model_name]['query_errors'] += 1
                else:
                    metadata = model_result.get('metadata', {})
                    stats_by_model[model_name]['query_total_time'] += metadata.get('response_time_ms', 0)
                    stats_by_model[model_name]['query_total_tokens'] += metadata.get('total_tokens', 0)
                    stats_by_model[model_name]['query_count'] += 1

        # 펑션 콜링 통계
        for benchmark in results.get('function_call_benchmarks', []):
            for model_result in benchmark['results']:
                model_name = model_result['model']
                if model_name not in stats_by_model:
                    stats_by_model[model_name] = {
                        'query_total_time': 0,
                        'query_total_tokens': 0,
                        'query_count': 0,
                        'query_errors': 0,
                        'fc_total_time': 0,
                        'fc_total_score': 0.0,
                        'fc_count': 0,
                        'fc_errors': 0
                    }

                if 'error' in model_result:
                    stats_by_model[model_name]['fc_errors'] += 1
                else:
                    metadata = model_result.get('metadata', {})
                    stats_by_model[model_name]['fc_total_time'] += metadata.get('response_time_ms', 0)
                    stats_by_model[model_name]['fc_count'] += 1

                    evaluation = model_result.get('evaluation', {})
                    if evaluation.get('evaluated'):
                        stats_by_model[model_name]['fc_total_score'] += evaluation.get('score', 0)

        # 일반 질의 통계 테이블
        if results.get('query_benchmarks'):
            lines.append("\n## 일반 질의 통계\n\n")
            lines.append("| 모델 | 평균 응답 시간 | 총 토큰 | 에러 수 |\n")
            lines.append("|------|---------------|---------|--------|\n")

            for model_name, stats in stats_by_model.items():
                if stats['query_count'] > 0 or stats['query_errors'] > 0:
                    avg_time = stats['query_total_time'] / stats['query_count'] if stats['query_count'] > 0 else 0
                    lines.append(f"| {model_name} | {avg_time:.0f}ms | {stats['query_total_tokens']} | {stats['query_errors']} |\n")

        # 펑션 콜링 통계 테이블
        if results.get('function_call_benchmarks'):
            lines.append("\n## 펑션 콜링 통계\n\n")
            lines.append("| 모델 | 평균 응답 시간 | 평균 정확도 | 에러 수 |\n")
            lines.append("|------|---------------|------------|--------|\n")

            for model_name, stats in stats_by_model.items():
                if stats['fc_count'] > 0 or stats['fc_errors'] > 0:
                    avg_time = stats['fc_total_time'] / stats['fc_count'] if stats['fc_count'] > 0 else 0
                    avg_score = stats['fc_total_score'] / stats['fc_count'] if stats['fc_count'] > 0 else 0
                    lines.append(f"| {model_name} | {avg_time:.0f}ms | {avg_score:.2f} / 1.0 | {stats['fc_errors']} |\n")

        return "".join(lines)