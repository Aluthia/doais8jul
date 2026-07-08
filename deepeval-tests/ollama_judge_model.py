"""
Custom deepeval judge LLM that talks to the same Ollama backend used by the
llm-multiroute service (see llm-multiroute/app/service/ai_service.py), instead
of calling out to OpenAI.

Uses the same environment variables as llm-multiroute:
  - OLLAMA_BASE_URL (default: https://ollama.com)
  - OLLAMA_API_KEY
  - OLLAMA_TEMPERATURE (default: 0.7)

The specific model used to *judge* eval results is configurable separately via
OLLAMA_MODEL_JUDGE so it doesn't have to match any single task model. It
defaults to the largest/most capable model already configured for the project
(gemma3:12b, the same one used for intent detection) since judging quality
benefits from a stronger model.
"""

import os

import httpx
from deepeval.models import DeepEvalBaseLLM


class OllamaJudgeModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
    ):
        self.model = model or os.getenv("OLLAMA_MODEL_JUDGE", "gemma3:12b")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
        self.api_key = api_key if api_key is not None else os.getenv("OLLAMA_API_KEY", "")
        self.temperature = (
            temperature if temperature is not None else float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
        )
        super().__init__(self.model)

    def load_model(self):
        return self.model

    def _headers(self) -> dict:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, prompt: str) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": self.temperature,
        }

    def generate(self, prompt: str) -> str:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{self.base_url}/api/chat",
                headers=self._headers(),
                json=self._payload(prompt),
            )
            response.raise_for_status()
            return response.json()["message"]["content"]

    async def a_generate(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                headers=self._headers(),
                json=self._payload(prompt),
            )
            response.raise_for_status()
            return response.json()["message"]["content"]

    def get_model_name(self) -> str:
        return f"Ollama:{self.model}"
