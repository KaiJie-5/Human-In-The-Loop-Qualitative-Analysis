from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol

import httpx


class OllamaError(RuntimeError):
    """Base error safe to present without logging protected request content."""


class OllamaConnectionError(OllamaError):
    pass


class OllamaTimeoutError(OllamaError):
    pass


class OllamaHTTPError(OllamaError):
    def __init__(self, message: str, *, status_code: int, raw_body: str):
        super().__init__(message)
        self.status_code = status_code
        self.raw_body = raw_body


class OllamaResponseError(OllamaError):
    def __init__(self, message: str, *, raw_body: str):
        super().__init__(message)
        self.raw_body = raw_body


@dataclass(frozen=True, slots=True)
class ModelInfo:
    name: str
    digest: str
    modified_at: str | None = None
    context_length: int | None = None


@dataclass(frozen=True, slots=True)
class OllamaResponse:
    raw_body: str
    payload: dict[str, Any]
    content: str
    model: str
    metadata: dict[str, Any]


class OllamaClient(Protocol):
    def list_models(self) -> list[ModelInfo]: ...

    def show_model(self, model: str) -> ModelInfo: ...

    def generate_structured(
        self,
        *,
        model: str,
        prompt: str,
        schema: dict[str, Any],
        options: dict[str, Any],
        seed: int,
    ) -> OllamaResponse: ...


class HttpOllamaClient:
    def __init__(
        self,
        base_url: str,
        *,
        health_timeout_seconds: float = 5,
        generation_timeout_seconds: float = 900,
    ):
        self.base_url = base_url.rstrip("/")
        self.health_timeout_seconds = health_timeout_seconds
        self.generation_timeout_seconds = generation_timeout_seconds

    def list_models(self) -> list[ModelInfo]:
        payload = self._request_json("GET", "/api/tags", timeout=self.health_timeout_seconds)
        models = payload.get("models")
        if not isinstance(models, list):
            raise OllamaResponseError("Ollama model list is malformed.", raw_body=json.dumps(payload))
        result: list[ModelInfo] = []
        for item in models:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                continue
            result.append(
                ModelInfo(
                    name=item["name"],
                    digest=str(item.get("digest", "")),
                    modified_at=str(item["modified_at"]) if item.get("modified_at") else None,
                )
            )
        return sorted(result, key=lambda model: model.name.casefold())

    def show_model(self, model: str) -> ModelInfo:
        if not model.strip():
            raise ValueError("Model name is required.")
        payload = self._request_json(
            "POST", "/api/show", timeout=self.health_timeout_seconds, json_body={"model": model}
        )
        details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
        model_info = payload.get("model_info") if isinstance(payload.get("model_info"), dict) else {}
        context_lengths: list[int] = []
        for key, value in model_info.items():
            if not str(key).endswith(".context_length"):
                continue
            try:
                parsed_context = int(value)
            except (TypeError, ValueError):
                continue
            if parsed_context > 0:
                context_lengths.append(parsed_context)
        digest = str(payload.get("digest") or details.get("digest") or "")
        try:
            tags = self._request_json("GET", "/api/tags", timeout=self.health_timeout_seconds)
            models = tags.get("models") if isinstance(tags.get("models"), list) else []
            for item in models:
                if isinstance(item, dict) and item.get("name") == model and item.get("digest"):
                    digest = str(item["digest"])
                    break
        except OllamaError:
            pass
        if not digest:
            digest = "show:" + hashlib.sha256(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
        return ModelInfo(
            name=model,
            digest=digest,
            modified_at=payload.get("modified_at"),
            context_length=max(context_lengths) if context_lengths else None,
        )

    def generate_structured(
        self,
        *,
        model: str,
        prompt: str,
        schema: dict[str, Any],
        options: dict[str, Any],
        seed: int,
    ) -> OllamaResponse:
        request = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "format": schema,
            "options": {**options, "seed": seed},
        }
        raw, payload = self._request("POST", "/api/chat", self.generation_timeout_seconds, request)
        message = payload.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise OllamaResponseError("Ollama response has no assistant content.", raw_body=raw)
        metadata = {
            key: payload.get(key)
            for key in (
                "created_at", "done", "done_reason", "total_duration", "load_duration",
                "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration",
            )
            if key in payload
        }
        return OllamaResponse(
            raw_body=raw,
            payload=payload,
            content=message["content"],
            model=str(payload.get("model", model)),
            metadata=metadata,
        )

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        timeout: float,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _, payload = self._request(method, path, timeout, json_body)
        return payload

    def _request(
        self,
        method: str,
        path: str,
        timeout: float,
        json_body: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any]]:
        try:
            with httpx.Client(timeout=httpx.Timeout(timeout, connect=min(timeout, 5))) as client:
                response = client.request(method, f"{self.base_url}{path}", json=json_body)
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError("Ollama request timed out; the review remains saved.") from exc
        except httpx.RequestError as exc:
            raise OllamaConnectionError(
                "Could not connect to Ollama; check the base URL and local service."
            ) from exc
        raw = response.text
        if response.is_error:
            detail = ""
            try:
                value = response.json()
                detail = str(value.get("error", "")) if isinstance(value, dict) else ""
            except ValueError:
                pass
            message = f"Ollama returned HTTP {response.status_code}"
            if detail:
                message += f": {detail}"
            raise OllamaHTTPError(message, status_code=response.status_code, raw_body=raw)
        try:
            payload = response.json()
        except ValueError as exc:
            raise OllamaResponseError("Ollama returned invalid JSON.", raw_body=raw) from exc
        if not isinstance(payload, dict):
            raise OllamaResponseError("Ollama response must be a JSON object.", raw_body=raw)
        return raw, payload
