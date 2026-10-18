"""Offline demo runs without a camera, ROS, network or saved data."""

import io
import json

import cv2
import numpy as np

from buddy_core.vision import demo


def args_for(tmp_path, **kw):
    img = np.full((120, 160, 3), 90, np.uint8)
    path = str(tmp_path / "a.png")
    cv2.imwrite(path, img)
    ns = dict(image=[path], video=None, repeat=5, fps=10.0,
              profile="balanced", models_dir=str(tmp_path / "no_models"),
              threads=1, max_frames=0)
    ns.update(kw)

    return type("A", (), ns)


def test_runs_with_no_models_and_reports_what_is_missing(tmp_path):
    out = io.StringIO()
    summary = demo.run(args_for(tmp_path), out)
    lines = [json.loads(x) for x in out.getvalue().splitlines()]
    assert summary["frames"] == 5 and len(lines) == 6
    assert summary["unavailable_stages"], "missing models must be reported"
    assert "nothing was saved" in summary["note"]


def test_video_file_input_and_no_files_written(tmp_path):
    path = str(tmp_path / "v.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
                        (160, 120))

    for _ in range(6):
        w.write(np.full((120, 160, 3), 60, np.uint8))

    w.release()
    ns = args_for(tmp_path, image=None, video=path)
    before = set(tmp_path.iterdir())
    out = io.StringIO()
    s = demo.run(ns, out)
    assert s["frames"] == 6
    assert set(tmp_path.iterdir()) == before


def test_bad_input_exits_cleanly(tmp_path):
    import pytest

    with pytest.raises(SystemExit):
        demo.run(args_for(tmp_path, image=[str(tmp_path / "none.png")]),
                 io.StringIO())
