# LLM Bench

다양한 LLM 모델에 대해 동일한 프롬프트를 실행하고 결과를 비교하는 벤치마크 도구입니다.

## 특징

- 여러 LLM 모델 동시 테스트 (OpenAI, Anthropic, Google 등)
- 프롬프트 파일 기반 관리
- 응답 시간 및 토큰 사용량 측정
- JSON 및 Markdown 형식으로 결과 저장
- 간단한 CLI 인터페이스

## 설치

1. 의존성 설치:
```bash
pip install -r requirements.txt
```

2. 환경 변수 설정:

`.env` 파일을 생성하고 API 키를 입력하세요:

```bash
# .env 파일 내용
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
GOOGLE_API_KEY=your-google-api-key-here
OPENROUTER_API_KEY=your-openrouter-api-key-here
```

또는 환경 변수로 직접 설정:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GOOGLE_API_KEY="your-google-api-key"
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

## 디렉토리 구조

```
llm-bench/
├── config/
│   └── models.yaml          # 모델 설정
├── prompts/
│   ├── 짜장면벤치.txt
│   └── 괭벤치.txt
├── results/                 # 결과 저장 디렉토리
│   └── {timestamp}/
│       ├── results.json
│       └── summary.md
├── src/
│   ├── prompt_loader.py
│   ├── model_runner.py
│   ├── result_processor.py
│   └── benchmark.py
├── requirements.txt
└── README.md
```

## 사용법

### 기본 사용 (모든 모델, 모든 프롬프트)

```bash
cd llm-bench
python src/benchmark.py
```

### 특정 모델 선택

```bash
python src/benchmark.py --models gpt-4o,claude-3.5-sonnet
```

### 특정 프롬프트 선택

```bash
python src/benchmark.py --prompts 짜장면벤치
```

### 모델과 프롬프트 동시 선택

```bash
python src/benchmark.py --models gpt-4o-mini,claude-3-haiku --prompts 괭벤치
```

### 사용 가능한 모델 목록 확인

```bash
python src/benchmark.py --list-models
```

### 사용 가능한 프롬프트 목록 확인

```bash
python src/benchmark.py --list-prompts
```

## 모델 설정

[config/models.yaml](config/models.yaml) 파일에서 모델을 추가/수정할 수 있습니다:

### 기본 모델 추가
```yaml
models:
  my-custom-model:
    provider: openai
    model: gpt-4o
    temperature: 0.7
    max_tokens: 2000
```

### OpenRouter 모델 추가
OpenRouter를 사용하면 다양한 LLM 모델을 하나의 API로 사용할 수 있습니다:

```yaml
models:
  kimi-k2-thinking:
    provider: openrouter
    model: moonshotai/kimi-k2-thinking
    temperature: 0.7
    max_tokens: 2000
    base_url: https://openrouter.ai/api/v1
```

**지원 provider:**
- `openai`: OpenAI 모델 (GPT-4, GPT-4o 등)
- `anthropic`: Anthropic 모델 (Claude 3.5 Sonnet, Haiku 등)
- `google-genai`: Google 모델 (Gemini Pro 등)
- `openrouter`: OpenRouter 통합 (다양한 모델 지원)

## 프롬프트 추가

`prompts/` 디렉토리에 `.txt` 파일을 추가하면 자동으로 인식됩니다:

```bash
echo "Your prompt here" > prompts/new-benchmark.txt
```

## 결과 확인

결과는 `results/{timestamp}/` 디렉토리에 저장됩니다:

- `results.json`: 전체 결과 데이터 (JSON)
- `summary.md`: 읽기 쉬운 요약 (Markdown)

## 예시 출력

```
============================================================
LLM 벤치마크 시작
============================================================
모델: gpt-4o, claude-3.5-sonnet
프롬프트: 짜장면벤치, 괭벤치
============================================================

[1/2] 프롬프트: 짜장면벤치
------------------------------------------------------------
  실행 중: gpt-4o... ✓ 완료 (1234ms, 450 tokens)
  실행 중: claude-3.5-sonnet... ✓ 완료 (987ms, 380 tokens)

[2/2] 프롬프트: 괭벤치
------------------------------------------------------------
  실행 중: gpt-4o... ✓ 완료 (567ms, 120 tokens)
  실행 중: claude-3.5-sonnet... ✓ 완료 (432ms, 95 tokens)

============================================================
결과 저장 완료: results/20251112_103000
  - JSON: results/20251112_103000/results.json
  - Markdown: results/20251112_103000/summary.md
============================================================
```

## 라이센스

MIT
