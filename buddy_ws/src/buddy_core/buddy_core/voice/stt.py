"""Speech recognition adapters.

The engine talks to a small ``Recognizer`` interface so that Vosk can be
swapped, faked in tests, or replaced by an online provider.  ``VoskRecognizer``
is written against the documented Vosk API and is UNVERIFIED until it runs
on the Pi with the real model.

Language note: the shipped model is ``vosk-model-small-en-us``.  It does not
recognise Tamil.  Tamil needs a separate Tamil model (not bundled); until
one is configured the language manager must treat Tamil as unsupported
offline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Transcript:
    """A finished recognition result."""

    text: str
    confidence: float | None
    language: str = "en"


class Recognizer(Protocol):
    """Streaming recogniser."""

    def accept(self, pcm16: bytes) -> Transcript | None:
        """Feed audio; return a transcript when an utterance is final."""

    def partial(self) -> str:
        """Current partial text."""

    def finalize(self) -> Transcript | None:
        """Force the end of the utterance and return what was heard."""

    def reset(self) -> None:
        """Discard internal state."""


class VoskRecognizer:
    """Wrap a Vosk model, optionally with a restricted grammar."""

    def __init__(
        self,
        model,
        sample_rate: int = 16000,
        grammar: list[str] | None = None,
        language: str = "en",
        recognizer_factory=None,
    ) -> None:
        self._model = model
        self._rate = sample_rate
        self._grammar = grammar
        self.language = language
        self._factory = recognizer_factory
        self._rec = None
        self.reset()

    def _build(self):
        if self._factory is not None:
            return self._factory(self._model, self._rate, self._grammar)

        from vosk import KaldiRecognizer

        if self._grammar:
            rec = KaldiRecognizer(
                self._model, self._rate, json.dumps(self._grammar),
            )
        else:
            rec = KaldiRecognizer(self._model, self._rate)

        rec.SetWords(True)

        return rec

    def reset(self) -> None:
        self._rec = self._build()

    def accept(self, pcm16: bytes) -> Transcript | None:
        if not self._rec.AcceptWaveform(pcm16):
            return None

        return self._parse(self._rec.Result())

    def finalize(self) -> Transcript | None:
        try:
            return self._parse(self._rec.FinalResult())
        except Exception:
            return None

    def _parse(self, raw: str) -> Transcript | None:
        data = json.loads(raw)
        text = str(data.get("text", "")).strip()

        if not text or text == "[unk]":
            return None

        words = data.get("result") or []
        confs = [w["conf"] for w in words if "conf" in w]
        confidence = sum(confs) / len(confs) if confs else None

        return Transcript(text, confidence, self.language)

    def partial(self) -> str:
        try:
            return str(json.loads(self._rec.PartialResult())
                       .get("partial", ""))
        except Exception:
            return ""


class ScriptedRecognizer:
    """Test double: returns queued transcripts, one per ``accept`` call."""

    def __init__(
        self,
        results: list[Transcript | None] | None = None,
        finals: list[Transcript | None] | None = None,
    ):
        self.results = list(results or [])
        self.finals = list(finals or [])
        self.fed = 0
        self.resets = 0
        self._partial = ""

    def push(self, transcript: Transcript | None) -> None:
        """Queue the next result."""

        self.results.append(transcript)

    def accept(self, pcm16: bytes) -> Transcript | None:
        self.fed += 1

        if self.results:
            return self.results.pop(0)

        return None

    def partial(self) -> str:
        return self._partial

    def set_partial(self, text: str) -> None:
        """Set what ``partial`` reports."""

        self._partial = text

    def finalize(self) -> Transcript | None:
        if self.finals:
            return self.finals.pop(0)

        return None

    def reset(self) -> None:
        self.resets += 1
        self._partial = ""
