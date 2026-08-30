"""Tests for BUDDY safe neck control."""

from __future__ import annotations

from copy import deepcopy

import pytest

from buddy_core.config_loader import load_buddy_config
from buddy_core.neck.base import BuddyNeckError
from buddy_core.neck.controller import NeckController
from buddy_core.neck.factory import get_neck_backend
from buddy_core.neck.mock import MockNeckBackend


@pytest.fixture
def config():
    return load_buddy_config()


@pytest.fixture
def neck(config):
    backend = MockNeckBackend()
    controller = NeckController(backend, config)
    return controller, backend


def test_center_uses_calibrated_front_position(neck):
    controller, backend = neck

    result = controller.center()

    assert result == (135.0, 180.0)
    assert backend.pan_angle_deg == 135.0
    assert backend.tilt_angle_deg == 180.0


def test_pan_below_minimum_is_clamped(neck):
    controller, backend = neck

    result = controller.set_pan(-100)

    assert result == 50.0
    assert backend.pan_angle_deg == 50.0


def test_pan_above_maximum_is_clamped(neck):
    controller, backend = neck

    result = controller.set_pan(999)

    assert result == 180.0
    assert backend.pan_angle_deg == 180.0


def test_tilt_below_minimum_is_clamped(neck):
    controller, backend = neck

    result = controller.set_tilt(-100)

    assert result == 110.0
    assert backend.tilt_angle_deg == 110.0


def test_tilt_above_maximum_is_clamped(neck):
    controller, backend = neck

    result = controller.set_tilt(999)

    assert result == 180.0
    assert backend.tilt_angle_deg == 180.0


def test_look_clamps_both_axes(neck):
    controller, backend = neck

    result = controller.look(-500, 500)

    assert result == (50.0, 180.0)
    assert backend.pan_angle_deg == 50.0
    assert backend.tilt_angle_deg == 180.0


def test_non_numeric_pan_is_rejected(neck):
    controller, _ = neck

    with pytest.raises(BuddyNeckError):
        controller.set_pan("left")


def test_boolean_angle_is_rejected(neck):
    controller, _ = neck

    with pytest.raises(BuddyNeckError):
        controller.set_tilt(True)


def test_release_releases_mock_backend(neck):
    controller, backend = neck

    controller.center()
    controller.release()

    assert backend.released is True


def test_cleanup_cleans_mock_backend(neck):
    controller, backend = neck

    controller.cleanup()

    assert backend.cleaned_up is True
    assert backend.released is True


def test_factory_uses_mock_when_hardware_disabled(config):
    backend = get_neck_backend(config)

    assert isinstance(backend, MockNeckBackend)


def test_factory_uses_mock_during_simulation(config):
    test_config = deepcopy(config)
    test_config["hardware"]["enabled"] = True
    test_config["hardware"]["simulation"] = True
    test_config["hardware"]["neck"]["enabled"] = True

    backend = get_neck_backend(test_config)

    assert isinstance(backend, MockNeckBackend)


def test_factory_uses_mock_when_neck_disabled(config):
    test_config = deepcopy(config)
    test_config["hardware"]["enabled"] = True
    test_config["hardware"]["simulation"] = False
    test_config["hardware"]["neck"]["enabled"] = False

    backend = get_neck_backend(test_config)

    assert isinstance(backend, MockNeckBackend)


def test_config_rejects_pan_range_below_calibrated_minimum(config):
    from buddy_core.config_loader import (
        BuddyConfigError,
        validate_config,
    )

    test_config = deepcopy(config)
    test_config["hardware"]["neck"]["pan"]["min_angle_deg"] = 0

    with pytest.raises(
        BuddyConfigError,
        match="pan calibrated range",
    ):
        validate_config(test_config)


def test_config_rejects_tilt_range_below_calibrated_minimum(config):
    from buddy_core.config_loader import (
        BuddyConfigError,
        validate_config,
    )

    test_config = deepcopy(config)
    test_config["hardware"]["neck"]["tilt"]["min_angle_deg"] = 90

    with pytest.raises(
        BuddyConfigError,
        match="tilt calibrated range",
    ):
        validate_config(test_config)
