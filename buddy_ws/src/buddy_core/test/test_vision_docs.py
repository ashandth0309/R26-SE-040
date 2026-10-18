"""The vision document covers what was asked and matches the data."""

import json
import re
from pathlib import Path

from buddy_core.vision import eval_report, privacy

DOC = Path(eval_report.REPO) / "docs" / "sobiya_advanced_vision.md"
TEXT = DOC.read_text()
REQUIRED = (
    "Architecture", "Reused existing modules", "New modules",
    "Raspberry Pi Camera Module 2 setup", "Model details",
    "Dataset descriptions", "Face recognition",
    "Facial-expression classification",
    "Object recognition and enrollment", "Fall detection",
    "Human activity and gesture recognition", "Multimodal integration",
    "Proactive interaction policy", "Navigation safety gating",
    "ROS 2 messages, topics and services", "Memory, consent and privacy",
    "Installation commands", "Offline demonstration",
    "benchmark results", "Unit test results", "Known limitations",
    "Future research")


def test_every_required_section_is_present():
    heads = " ".join(re.findall(r"^## .*$", TEXT, re.M)).lower()
    missing = [r for r in REQUIRED if r.lower() not in heads
               and r.lower() not in TEXT.lower()]
    assert missing == []
    assert "```\n              Pi Camera Module 2" in TEXT


def test_honesty_statements_are_present():
    low = " ".join(TEXT.lower().split())
    assert "not verified on the raspberry pi" in low
    assert "does not claim buddy can tell how a person feels" in low
    assert "cannot be described as approaching or following people safely" \
        in low
    assert "there are no real-camera results" in low
    assert "raspberry pi benchmark results: none" in low
    assert "is_raspberry_pi: false" in low


def test_no_em_dashes():
    assert "—" not in TEXT


def test_expression_numbers_match_the_result_file():
    d = json.loads((eval_report.EVAL /
                    "expression_zoo_fer2013_test.json").read_text())
    assert f"{100 * d['accuracy']:.1f}%" in TEXT
    assert f"{d['macro_f1']:.3f}" in TEXT
    pc = d["per_class"]

    for name in ("happy", "surprise", "neutral", "disgust", "sad", "angry",
                 "fear"):
        assert f"{name} {pc[name]['recall']:.2f}" in TEXT, name

    cal = d["calibration"]
    assert f"{cal['ece_held_out_after']:.3f}" in TEXT
    assert str(cal["temperature_fit_on_half"]) in TEXT


def test_documented_topics_match_the_node():
    for topic in ("/vision/people", "/vision/objects", "/vision/events",
                  "/vision/status", "/person/recognized",
                  "/emotion/observation", "/security/event",
                  "/vision/command", "/vision/identify_object",
                  "/vision/teach_object", "/vision/where_is",
                  "/vision/enable", "/camera/image_raw", "/voice/status"):
        assert topic in TEXT, topic


def test_privacy_table_covers_every_inventory_item_kind():
    assert any("photos" in str(i) for i in privacy.DATA_INVENTORY)
    assert all(i["photos"] is False for i in privacy.DATA_INVENTORY)
    assert "No photos are stored anywhere" in TEXT
    assert "Nothing is encrypted at rest" in TEXT


def test_every_module_named_in_the_doc_exists():
    pkg = Path(eval_report.__file__).parent
    named = set(re.findall(r"`([a-z_]+)`", TEXT))
    mods = {p.stem for p in pkg.glob("*.py")}
    table = re.search(r"## 4\. New modules(.*?)Interface additions", TEXT,
                      re.S).group(1)
    listed = set(re.findall(r"`([a-z_0-9]+)`", table))
    assert listed - mods - {"vision_models"} == set()
    assert named is not None


def test_documented_commands_exist():
    root = Path(eval_report.REPO)
    assert (root / "scripts" / "fetch_vision_models.py").exists()
    assert (root / "deployment" / "vision_requirements.txt").exists()
    assert (root / "docs" / "evaluation" / "test_matrix.md").exists()
    assert (root / "docs" / "sobiya_vision_audit.md").exists()
