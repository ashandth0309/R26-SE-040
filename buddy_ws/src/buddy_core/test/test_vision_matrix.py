"""Phase 23: the 30-area test matrix is real, honest and consistent."""

import json
import re
from pathlib import Path

import buddy_core.vision as vision_pkg
from buddy_core.vision import eval_report

TESTS = Path(__file__).resolve().parent
MATRIX = json.loads((eval_report.EVAL / "test_matrix.json").read_text())
LEVELS = {"mock", "synthetic_logic", "real_model_smoke", "dataset_measured",
          "static_check", "stub_only", "tool_only", "hardware_required"}


def test_exactly_the_thirty_required_areas_in_order():
    ids = [a["id"] for a in MATRIX["areas"]]
    assert ids == list(range(1, 31))
    names = [a["name"] for a in MATRIX["areas"]]
    assert names[0] == "Camera acquisition"
    assert names[28] == "Raspberry Pi CPU and memory limits"
    assert names[29] == "End-to-end conversational latency"


def test_every_listed_test_really_exists():
    missing = []

    for a in MATRIX["areas"]:
        for ref in a["tests"]:
            fname, name = ref.split("::")
            path = TESTS / fname

            if not path.exists() or not re.search(
                    rf"^def {re.escape(name)}\(", path.read_text(), re.M):
                missing.append(ref)

    assert missing == []


def test_every_area_states_limits_and_a_manual_procedure():
    for a in MATRIX["areas"]:
        assert a["evidence"] in LEVELS, a["id"]
        assert a["tests"], a["id"]

        for key in ("measured", "not_verified", "manual_procedure"):
            assert len(a[key].strip()) > 20, (a["id"], key)


def test_no_area_claims_a_real_camera_result():
    assert MATRIX["real_camera_results"] == "none"
    blob = json.dumps(MATRIX).lower()

    for banned in ("tested on the pi", "pi camera result",
                   "physically tested"):
        assert banned not in blob


def test_hardware_only_areas_have_no_pi_numbers():
    for a in MATRIX["areas"]:
        if a["evidence"] == "hardware_required":
            assert "No Pi number" in a["measured"] or \
                "No latency" in a["measured"]
            assert not re.search(r"\d+(\.\d+)?\s*(ms|fps|%|mb)\b",
                                 a["measured"].lower())


def test_only_dataset_measured_area_is_expression_uncertainty():
    ds = [a["id"] for a in MATRIX["areas"]
          if a["evidence"] == "dataset_measured"]
    assert ds == [12]


def test_no_em_dashes_in_matrix_or_report():
    assert "—" not in json.dumps(MATRIX, ensure_ascii=False)
    assert "—" not in eval_report.build_report()


def test_report_contains_confusion_matrices_and_per_class_metrics():
    text = eval_report.build_report()
    assert "Real-camera results: none" in text
    assert text.count("true \\ predicted") >= 4      # 7-class and groups, x2
    assert text.count("| class | precision | recall | F1 | support |") >= 4
    assert "Neutral faces called negative" in text
    assert "Coverage that is missing" in text and "skin tone" in text


def test_report_numbers_match_the_result_files():
    data = json.loads(
        (eval_report.EVAL / "expression_zoo_fer2013_test.json").read_text())
    text = eval_report.build_report()
    assert f"{100 * data['accuracy']:.1f}%" in text
    total = sum(sum(r) for r in data["confusion"])
    assert total == data["images_scored"]


def test_committed_markdown_report_is_current():
    path = eval_report.EVAL / "test_matrix.md"
    assert path.exists(), "run: python -m buddy_core.vision.eval_report"
    assert path.read_text() == eval_report.build_report()


def test_runtime_vision_code_has_no_network_imports():
    pat = re.compile(r"^\s*(import|from)\s+(urllib|requests|socket|http|"
                     r"ftplib|aiohttp|websocket)", re.M)
    offenders = []

    for p in Path(vision_pkg.__file__).parent.glob("*.py"):
        if pat.search(p.read_text()):
            offenders.append(p.name)

    assert offenders == ["models.py"], offenders      # the download tool
