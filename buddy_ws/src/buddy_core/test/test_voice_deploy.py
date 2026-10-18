"""Tests for the deployment helpers (benchmark, scripts, ignore rules)."""

import json
import re
import subprocess
from pathlib import Path

from buddy_core.voice import benchmark

REPO = Path(__file__).resolve().parents[4]


def test_machine_info_reports_pi_flag_honestly():
    info = benchmark.machine_info()
    assert isinstance(info["is_raspberry_pi"], bool)
    assert "machine" in info


def test_report_has_all_sections_and_no_model_runs_without_paths():
    report = benchmark.run()
    for key in ("vad", "features", "parse", "dialogue_rules",
                "stop_path", "targets_missed", "note"):
        assert key in report
    assert "real_models" not in report
    assert report["stop_path"]["n"] > 0
    json.dumps(report)


def test_targets_flag_slow_results():
    fake = {
        "vad": {"realtime_factor": 9.0},
        "features": {"realtime_factor": 9.0},
        "stop_path": {"p95_ms": 999.0},
    }
    assert len(benchmark.check_targets(fake)) == 3


def test_missing_stop_path_counts_as_a_miss():
    fake = {
        "vad": {"realtime_factor": 0.0},
        "features": {"realtime_factor": 0.0},
        "stop_path": {"n": 0},
    }
    assert benchmark.check_targets(fake)


def test_startup_script_syntax_and_guards():
    script = REPO / "scripts" / "start_buddy_companion.sh"
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    text = script.read_text()
    assert "BUDDY_VOSK_MODEL" in text
    assert "buddy_voice$" in text


def test_no_secrets_in_deployment_files():
    for name in ("buddy_companion.service", "voice_requirements.txt"):
        text = (REPO / "deployment" / name).read_text().lower()
        assert "api_key=" not in text
        assert not re.search(r"\bsk-[a-z0-9]{16,}", text)


def test_model_binaries_are_ignored():
    rules = (REPO / ".gitignore").read_text()
    assert "*.onnx" in rules
    assert "vosk-model-" in rules
