"""Ollama helpers for local VLM / LLM calls with JSON parsing and disk cache."""
from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from .config import LLM_DIR, OLLAMA_HOST, OLLAMA_TIMEOUT, TEXT_MODEL, VLM_MODEL, ensure_dirs


class OllamaError(RuntimeError):
    pass


def ollama_available(timeout: float = 2.0) -> bool:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False


def wait_for_ollama(seconds: int = 60) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if ollama_available():
            return True
        time.sleep(1.0)
    return False


def _b64_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first JSON object out of a model response."""
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def generate(
    prompt: str,
    image_path: Optional[Path] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    force_json: bool = True,
    cache_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Call Ollama /api/generate. Caches the raw response under outputs/llm/."""
    ensure_dirs()
    if cache_key:
        cache_path = LLM_DIR / f"{cache_key}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())

    if not ollama_available():
        raise OllamaError(
            "Ollama is not running at 127.0.0.1:11434. "
            "Start it (see README) and pull llama3.2-vision."
        )

    payload: Dict[str, Any] = {
        "model": model or (VLM_MODEL if image_path else TEXT_MODEL),
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": temperature, "num_predict": 700},
    }
    if force_json:
        payload["format"] = "json"
    if image_path is not None:
        payload["images"] = [_b64_image(Path(image_path))]

    try:
        r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT)
        r.raise_for_status()
    except requests.RequestException as exc:
        raise OllamaError(f"Ollama generate failed: {exc}") from exc

    body = r.json()
    raw = body.get("response", "")
    parsed = _extract_json(raw)
    record = {
        "model": payload["model"],
        "temperature": temperature,
        "prompt": prompt,
        "image": str(image_path) if image_path else None,
        "raw": raw,
        "json": parsed,
        "total_duration_ns": body.get("total_duration"),
    }
    if cache_key:
        (LLM_DIR / f"{cache_key}.json").write_text(json.dumps(record, indent=2))
    return record
