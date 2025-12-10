"""펑션 콜링 실행 모듈"""
import os
import time
from typing import Dict, Any, List, Callable
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI


class FunctionCallRunner:
    """Agent 기반 펑션 콜링을 실행하는 클래스"""

    def __init__(self, config: Dict[str, Any], langfuse_integration=None):
        """
        Args:
            config: 모델 설정 딕셔너리
            langfuse_integration: LangfuseIntegration 인스턴스 (선택)
        """
        self.config = config
        self.langfuse = langfuse_integration

    def run_scenario(
        self,
        model_name: str,
        prompt: str,
        tools: List[Callable],
        expected_tool_calls: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Agent로 펑션 콜링 실행

        Args:
            model_name: 모델 이름
            prompt: 실행할 프롬프트
            tools: 사용 가능한 tool 함수 리스트
            expected_tool_calls: 예상되는 tool 호출 정보 (평가용)

        Returns:
            {
                "response": 최종 응답,
                "tool_calls": 호출된 툴 정보,
                "metadata": 메타데이터,
                "evaluation": 평가 결과
            }
        """
        if model_name not in self.config['models']:
            raise ValueError(f"모델을 찾을 수 없습니다: {model_name}")

        model_config = self.config['models'][model_name]

        # 모델 초기화
        try:
            # OpenRouter는 별도 처리
            if model_config['provider'] == 'openrouter':
                model = ChatOpenAI(
                    model=model_config['model'],
                    temperature=model_config.get('temperature', 0.7),
                    max_tokens=model_config.get('max_tokens', 2000),
                    base_url=model_config.get('base_url', 'https://openrouter.ai/api/v1'),
                    api_key=os.getenv('OPENROUTER_API_KEY')
                )
            else:
                model = init_chat_model(
                    model=model_config['model'],
                    model_provider=model_config['provider'],
                    temperature=model_config.get('temperature', 0.7),
                    max_tokens=model_config.get('max_tokens', 2000)
                )

            # Agent 생성
            agent = create_agent(
                model=model,
                tools=tools,
                system_prompt="You are a helpful assistant that uses tools when needed."
            )

        except Exception as e:
            return {
                "error": f"Agent 초기화 오류: {str(e)}",
                "metadata": {"response_time_ms": 0}
            }

        start_time = time.time()

        try:
            # Agent 실행
            result = agent.invoke({
                "messages": [{"role": "user", "content": prompt}]
            })

            end_time = time.time()
            response_time_ms = int((end_time - start_time) * 1000)

            # Tool 호출 정보 추출
            tool_calls = self._extract_tool_calls(result)

            # 평가
            evaluation = self._evaluate_tool_calls(tool_calls, expected_tool_calls)

            # 최종 응답 추출
            final_response = self._extract_final_response(result)

            return {
                "response": final_response,
                "tool_calls": tool_calls,
                "metadata": {
                    "response_time_ms": response_time_ms,
                    "num_tool_calls": len(tool_calls),
                    "num_messages": len(result.get('messages', []))
                },
                "evaluation": evaluation
            }

        except Exception as e:
            end_time = time.time()
            response_time_ms = int((end_time - start_time) * 1000)

            return {
                "error": str(e),
                "metadata": {
                    "response_time_ms": response_time_ms
                }
            }

    def _extract_tool_calls(self, result: Dict) -> List[Dict]:
        """
        결과에서 tool 호출 정보 추출

        Args:
            result: Agent 실행 결과

        Returns:
            Tool 호출 정보 리스트
        """
        tool_calls = []

        if 'messages' not in result:
            return tool_calls

        for msg in result['messages']:
            # AIMessage에 tool_calls 속성이 있는 경우
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({
                        'name': tc.get('name') or tc.get('function', {}).get('name'),
                        'args': tc.get('args') or tc.get('function', {}).get('arguments', {})
                    })

            # 또는 additional_kwargs에 tool_calls가 있는 경우
            elif hasattr(msg, 'additional_kwargs') and 'tool_calls' in msg.additional_kwargs:
                for tc in msg.additional_kwargs['tool_calls']:
                    import json
                    args = tc.get('function', {}).get('arguments', '{}')
                    if isinstance(args, str):
                        args = json.loads(args)
                    tool_calls.append({
                        'name': tc.get('function', {}).get('name'),
                        'args': args
                    })

        return tool_calls

    def _extract_final_response(self, result: Dict) -> str:
        """
        최종 응답 텍스트 추출

        Args:
            result: Agent 실행 결과

        Returns:
            최종 응답 텍스트
        """
        if 'messages' in result and result['messages']:
            # 마지막 메시지가 최종 응답
            last_msg = result['messages'][-1]
            if hasattr(last_msg, 'content'):
                return last_msg.content
            return str(last_msg)

        return str(result)

    def _evaluate_tool_calls(
        self,
        actual_calls: List[Dict],
        expected_calls: List[Dict]
    ) -> Dict[str, Any]:
        """
        Tool 호출 평가

        Args:
            actual_calls: 실제 호출된 tool 정보
            expected_calls: 예상되는 tool 호출 정보

        Returns:
            평가 결과 딕셔너리
        """
        if not expected_calls:
            return {
                "evaluated": False,
                "message": "No expected tool calls to evaluate"
            }

        evaluation = {
            "evaluated": True,
            "correct_tool": False,
            "correct_args": False,
            "score": 0.0,
            "details": {}
        }

        if not actual_calls:
            evaluation["message"] = "No tool calls were made"
            return evaluation

        # 첫 번째 tool call 평가 (간단한 버전)
        actual = actual_calls[0]
        expected = expected_calls[0]

        # Tool 이름 확인
        expected_tool = expected.get('tool')
        if actual['name'] == expected_tool:
            evaluation['correct_tool'] = True
            evaluation['score'] += 0.5
            evaluation['details']['tool_name'] = f"✓ Correct: {actual['name']}"
        else:
            evaluation['details']['tool_name'] = f"✗ Expected: {expected_tool}, Got: {actual['name']}"

        # Arguments 확인
        expected_args = expected.get('expected_args', {})
        if expected_args:
            matched_args = []
            mismatched_args = []

            for key, expected_value in expected_args.items():
                actual_value = actual['args'].get(key)

                # 문자열 비교 (대소문자 무시)
                if actual_value is not None:
                    if str(actual_value).lower() == str(expected_value).lower():
                        matched_args.append(f"{key}={actual_value}")
                    else:
                        mismatched_args.append(f"{key}: expected '{expected_value}', got '{actual_value}'")
                else:
                    mismatched_args.append(f"{key}: missing (expected '{expected_value}')")

            if len(matched_args) == len(expected_args):
                evaluation['correct_args'] = True
                evaluation['score'] += 0.5
                evaluation['details']['args'] = f"✓ All args correct: {', '.join(matched_args)}"
            else:
                evaluation['details']['args'] = f"✗ Args mismatch: {', '.join(mismatched_args)}"
        else:
            # 예상 args가 없으면 자동으로 정답 처리
            evaluation['correct_args'] = True
            evaluation['score'] += 0.5
            evaluation['details']['args'] = "✓ No specific args expected"

        # 평가 메시지
        if evaluation['score'] == 1.0:
            evaluation['message'] = "Perfect match"
        elif evaluation['score'] >= 0.5:
            evaluation['message'] = "Partial match"
        else:
            evaluation['message'] = "No match"

        return evaluation
