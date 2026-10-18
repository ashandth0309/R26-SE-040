"""camera_node under strict ROS stubs (no real ROS 2 involved)."""

import importlib

import numpy as np
import pytest

import ros_stub_vision as stub
from buddy_core.vision.frames import make_frame


@pytest.fixture(scope="module", autouse=True)
def ros_stubs():
    stub.install()
    yield
    stub.uninstall()


@pytest.fixture
def cn(tmp_path):
    stub.base.Node.overrides = {"lock_path": str(tmp_path / "l.lock"),
                                "autostart": False, "fps": 10.0,
                                "width": 64, "height": 48}
    stub.base.Node.existing_nodes = []
    module = importlib.import_module("buddy_core.vision.camera_node")

    return importlib.reload(module)


def test_image_fields_are_correct_bgr8():
    img = np.zeros((4, 6, 3), np.uint8)
    f = make_frame(img, 1, 1234.9999999996, 1.0)
    fields = __import__("buddy_core.vision.camera_node",
                        fromlist=["x"]).frame_to_image_fields(f)
    assert (fields["width"], fields["height"], fields["step"]) == (6, 4, 18)
    assert fields["encoding"] == "bgr8" and len(fields["data"]) == 72
    assert 0 <= fields["nanosec"] < 1_000_000_000
    assert fields["sec"] in (1234, 1235)


def test_node_interfaces_match_the_camera_contract(cn):
    node = cn.CameraNode()
    assert set(node.pubs) == {"/camera/image_raw", "/diagnostics"}
    assert "/camera/enable" in node.services


def test_published_image_is_a_real_message_with_frame_data(cn):
    node = cn.CameraNode()
    node.service.step()
    node._publish_image()
    sent = node.pubs["/camera/image_raw"].sent
    assert len(sent) == 1
    msg = sent[0]
    assert (msg.width, msg.height, msg.encoding) == (64, 48, "bgr8")
    assert len(msg.data) == 64 * 48 * 3 and msg.header.frame_id
    node._publish_image()  # nothing new, nothing sent
    assert len(node.pubs["/camera/image_raw"].sent) == 1


def test_privacy_service_turns_camera_off_and_stops_publishing(cn):
    node = cn.CameraNode()
    node.service.step()
    _, callback = node.services["/camera/enable"]
    resp = callback(stub.base.SetBool.Request(False),
                    stub.base.SetBool.Response())
    assert resp.success and resp.message == "camera off"
    assert not node.service.indicator
    assert not node.service.step()
    resp = callback(stub.base.SetBool.Request(True),
                    stub.base.SetBool.Response())
    assert resp.message == "camera on"
    assert node.service.step()


def test_diagnostics_levels_follow_state(cn):
    node = cn.CameraNode()
    node._publish_diagnostics()
    first = node.pubs["/diagnostics"].sent[-1].status[0]
    assert first.level != 0  # nothing streaming yet
    node.service.step()
    node._publish_diagnostics()
    second = node.pubs["/diagnostics"].sent[-1].status[0]
    assert second.level == 0 and second.name == "buddy_camera"
    values = {kv.key: kv.value for kv in second.values}
    assert values["state"] == "streaming" and values["frames"] == "1"
    node.service.set_enabled(False)
    node._publish_diagnostics()
    assert node.pubs["/diagnostics"].sent[-1].status[0].level == 1


def test_refuses_second_camera_owner(cn):
    stub.base.Node.existing_nodes = ["buddy_camera"]
    with pytest.raises(RuntimeError, match="only one"):
        cn.CameraNode()
    stub.base.Node.overrides["allow_coexist"] = True
    cn.CameraNode()


def test_destroy_releases_the_camera(cn, tmp_path):
    node = cn.CameraNode()
    node.service.step()
    assert node.service.indicator
    node.destroy_node()
    assert not node.service.indicator and node.destroyed
