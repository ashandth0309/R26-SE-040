"""Language-model routing: local first, optional cloud, always a fallback.

* OFFLINE mode: no model is called; the dialogue manager's own rules and
  fallbacks answer.
* LOCAL: an on-robot model (Ollama on localhost).
* The cloud adapter is used only if the user has enabled online mode AND
  the connectivity breaker allows it.  API keys come from the environment
  at call time; there is no key in code or config.  If none is set the
  provider reports itself unavailable.
* Each provider gets the same trimmed conversation, so switching providers
  mid-conversation keeps context.  The whole attempt is bounded by one
  time budget, so a slow provider cannot stall the voice loop.
* Only recent chat turns are sent.  Stored memory, audio and transcripts
  of earlier sessions never are.

The HTTP adapters are written against the documented request shapes of
Ollama and OpenAI-compatible servers and are UNVERIFIED against live
servers.  Transports are injectable and tests use fakes.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from buddy_core.voice.connectivity import ConnectivityManager

SYSTEM_PROMPT = (
    "You are Buddy, a friendly robot dog companion. Reply in one to three "
    "short spoken sentences, plain text only. You cannot move the robot or "
    "do anything in the world. If you think a movement could help, you "
    "may add one line at the end exactly like 'ACTION: go forward' using "
    "only: go forward, go back, turn left, turn right. Never claim to "
    "have done it. Do not give medical, legal or financial instructions."
)


class LlmUnavailable(RuntimeError):
    """No provider could answer."""


class RouterMode(str, Enum):
    """Which providers may be used."""

    OFFLINE = "offline"
    LOCAL = "local"
    AUTO = "auto"


Transport = Callable[[str, dict, dict, float], dict]


def urllib_transport(url: str, payload: dict, headers: dict,
                     timeout: float) -> dict:
    """POST JSON and return parsed JSON (stdlib only)."""

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json", **headers})

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class OllamaProvider:
    """Local Ollama server."""

    def __init__(self, model: str = "llama3.2:1b",
                 base_url: str = "http://127.0.0.1:11434",
                 transport: Transport = urllib_transport,
                 max_tokens: int = 120) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.transport = transport
        self.max_tokens = max_tokens

    def complete(self, messages: list[dict], timeout_s: float) -> str:
        data = self.transport(
            f"{self.base_url}/api/chat",
            {"model": self.model, "messages": messages, "stream": False,
             "options": {"num_predict": self.max_tokens}},
            {}, timeout_s)

        return str(data["message"]["content"])


class OpenAICompatProvider:
    """Any OpenAI-compatible chat endpoint (cloud or self-hosted)."""

    def __init__(self, base_url: str, model: str,
                 api_key_env: str = "BUDDY_LLM_API_KEY",
                 transport: Transport = urllib_transport,
                 max_tokens: int = 120,
                 environ: dict | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.transport = transport
        self.max_tokens = max_tokens
        self._environ = environ if environ is not None else os.environ

    def complete(self, messages: list[dict], timeout_s: float) -> str:
        key = self._environ.get(self.api_key_env, "")

        if not key:
            raise LlmUnavailable(f"{self.api_key_env} is not set")

        data = self.transport(
            f"{self.base_url}/chat/completions",
            {"model": self.model, "messages": messages,
             "max_tokens": self.max_tokens},
            {"Authorization": f"Bearer {key}"}, timeout_s)

        return str(data["choices"][0]["message"]["content"])


class _Provider(Protocol):
    def complete(self, messages: list[dict], timeout_s: float) -> str:
        """Return reply text."""


@dataclass
class ProviderEntry:
    """A provider and how it is treated."""

    name: str
    kind: str                # "local" or "cloud"
    provider: _Provider
    timeout_s: float = 5.0


def trim_messages(messages: list[dict], max_turns: int = 8,
                  max_chars: int = 1500) -> list[dict]:
    """Last few turns within a size budget, prefixed by the system prompt."""

    recent = messages[-max_turns:]
    out: list[dict] = []
    total = 0

    for msg in reversed(recent):
        total += len(msg.get("content", ""))

        if total > max_chars and out:
            break

        out.append(msg)

    out.reverse()

    return [{"role": "system", "content": SYSTEM_PROMPT}] + out


@dataclass
class LlmRouter:
    """Implements the ``LlmProvider`` interface over several backends."""

    entries: list[ProviderEntry] = field(default_factory=list)
    mode: RouterMode = RouterMode.OFFLINE
    connectivity: ConnectivityManager | None = None
    cloud_enabled: Callable[[], bool] = lambda: False
    clock: Callable[[], float] = time.monotonic
    last_provider: str = ""
    last_errors: list = field(default_factory=list)

    def complete(self, messages: list[dict], timeout_s: float) -> str:
        """Answer using the first provider that works within budget."""

        self.last_errors = []
        self.last_provider = ""

        if self.mode is RouterMode.OFFLINE:
            raise LlmUnavailable("offline mode")

        deadline = self.clock() + timeout_s
        trimmed = trim_messages(messages)

        for entry in self.entries:
            if entry.kind == "cloud" and not self._cloud_ok():
                continue

            remaining = deadline - self.clock()

            if remaining < 0.3:
                self.last_errors.append("time budget used up")
                break

            try:
                text = self._call(entry, trimmed, remaining)
            except Exception as exc:
                self.last_errors.append(f"{entry.name}: {exc}")
                continue

            self.last_provider = entry.name

            return text

        raise LlmUnavailable("; ".join(self.last_errors) or "no provider")

    def _cloud_ok(self) -> bool:
        if self.mode is not RouterMode.AUTO or not self.cloud_enabled():
            return False

        return self.connectivity is None or \
            self.connectivity.allow_request()

    def _call(self, entry: ProviderEntry, messages: list[dict],
              remaining: float) -> str:
        timeout = min(remaining, entry.timeout_s)

        if entry.kind == "cloud" and self.connectivity is not None:
            return self.connectivity.call(
                lambda: entry.provider.complete(messages, timeout),
                budget_s=remaining)

        return entry.provider.complete(messages, timeout)
