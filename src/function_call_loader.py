"""펑션 콜링 시나리오 로딩 모듈"""
import yaml
from pathlib import Path
from typing import Dict, List, Any
import importlib
import sys


class FunctionCallLoader:
    """펑션 콜링 시나리오를 로드하는 클래스"""

    def __init__(self, scenarios_dir: str = "function_calls/scenarios"):
        """
        Args:
            scenarios_dir: 시나리오 파일이 있는 디렉토리 경로
        """
        self.scenarios_dir = Path(scenarios_dir)
        if not self.scenarios_dir.exists():
            raise ValueError(f"시나리오 디렉토리를 찾을 수 없습니다: {scenarios_dir}")

        # function_calls 모듈을 import 경로에 추가
        root_dir = Path(__file__).parent.parent
        if str(root_dir) not in sys.path:
            sys.path.insert(0, str(root_dir))

        # tools 모듈 import
        self.tools_module = importlib.import_module("function_calls.tools")

    def load_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """
        특정 시나리오 로드

        Args:
            scenario_name: 시나리오 파일명 (확장자 제외)

        Returns:
            시나리오 정보 딕셔너리
        """
        file_path = self.scenarios_dir / f"{scenario_name}.yaml"

        if not file_path.exists():
            raise ValueError(f"시나리오 파일을 찾을 수 없습니다: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            scenario = yaml.safe_load(f)

        # tool 함수 객체 가져오기
        tools = []
        for tool_name in scenario['tools']:
            if not hasattr(self.tools_module, tool_name):
                raise ValueError(f"Tool 함수를 찾을 수 없습니다: {tool_name}")
            tool_func = getattr(self.tools_module, tool_name)
            tools.append(tool_func)

        scenario['tool_objects'] = tools
        return scenario

    def load_all_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """
        모든 시나리오 로드

        Returns:
            {시나리오명: 시나리오 정보} 형식의 딕셔너리
        """
        scenarios = {}

        for file_path in self.scenarios_dir.glob("*.yaml"):
            scenario_name = file_path.stem
            scenarios[scenario_name] = self.load_scenario(scenario_name)

        if not scenarios:
            raise ValueError(f"시나리오 파일을 찾을 수 없습니다: {self.scenarios_dir}")

        return scenarios

    def list_scenarios(self) -> List[str]:
        """
        사용 가능한 시나리오 목록 반환

        Returns:
            시나리오 이름 리스트
        """
        return [f.stem for f in self.scenarios_dir.glob("*.yaml")]
