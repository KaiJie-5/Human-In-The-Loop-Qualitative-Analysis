from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


@dataclass(frozen=True, slots=True)
class AppConfig:
    repository_root: Path
    database_path: Path
    export_directory: Path
    ollama_base_url: str = "http://localhost:11434"
    health_timeout_seconds: float = 5.0
    generation_timeout_seconds: float = 900.0
    default_context_before: int = 20
    default_context_after: int = 20
    maximum_context_turns: int = 100
    temperature: float = 0.4
    top_p: float = 0.9
    output_tokens: int = 5000
    context_tokens: int = 65536
    maximum_pair_attempts: int = 3


def load_config(
    repository_root: Path | None = None,
    config_path: Path | None = None,
) -> AppConfig:
    root = (repository_root or Path.cwd()).resolve()
    path = config_path or root / "local_config.toml"
    payload: dict[str, Any] = {}
    if path.is_file():
        with path.open("rb") as handle:
            payload = tomllib.load(handle)

    paths = _section(payload, "paths")
    ollama = _section(payload, "ollama")
    review = _section(payload, "review")
    database = _relative_to(root, paths.get("database", "runtime/hitl.sqlite3"))
    exports = _relative_to(root, paths.get("exports", "exports"))
    config = AppConfig(
        repository_root=root,
        database_path=database,
        export_directory=exports,
        ollama_base_url=str(ollama.get("base_url", "http://localhost:11434")).rstrip("/"),
        health_timeout_seconds=float(ollama.get("health_timeout_seconds", 5)),
        generation_timeout_seconds=float(ollama.get("generation_timeout_seconds", 900)),
        default_context_before=int(review.get("default_context_before", 20)),
        default_context_after=int(review.get("default_context_after", 20)),
        maximum_context_turns=int(review.get("maximum_context_turns", 100)),
        temperature=float(review.get("temperature", 0.4)),
        top_p=float(review.get("top_p", 0.9)),
        output_tokens=int(review.get("output_tokens", 5000)),
        context_tokens=int(review.get("context_tokens", 65536)),
        maximum_pair_attempts=int(review.get("maximum_pair_attempts", 3)),
    )
    _validate(config)
    return config


def _section(payload: dict[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section {name!r} must be a table.")
    return value


def _relative_to(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _validate(config: AppConfig) -> None:
    if not config.ollama_base_url.startswith(("http://", "https://")):
        raise ValueError("Ollama base URL must begin with http:// or https://.")
    if config.maximum_context_turns < 20:
        raise ValueError("maximum_context_turns must be at least 20.")
    for name in ("default_context_before", "default_context_after"):
        value = getattr(config, name)
        if not 0 <= value <= config.maximum_context_turns:
            raise ValueError(f"{name} is outside the configured context bounds.")
    if not 0 <= config.temperature <= 2:
        raise ValueError("temperature must be between 0 and 2.")
    if not 0 < config.top_p <= 1:
        raise ValueError("top_p must be greater than 0 and at most 1.")
    if config.output_tokens <= 0 or config.context_tokens <= 0 or config.maximum_pair_attempts <= 0:
        raise ValueError("Context tokens, output tokens, and maximum attempts must be positive.")
    if config.output_tokens >= config.context_tokens:
        raise ValueError("output_tokens must be smaller than context_tokens.")
