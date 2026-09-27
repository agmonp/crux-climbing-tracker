"""Technique metrics use known geometry and synthetic movement."""
import math

import numpy as np
import pytest
from mediapipe.tasks.python.components.containers import NormalizedLandmark

from climbing_m1 import FrameData
from climb_app.config import Config
from climb_app.models import Contact, Metric, Observation
from climb_app.technique import TechniqueEngine, joint_angle_deg


def landmark(x: float, y: float, visible: bool = True) -> NormalizedLandmark:
    return NormalizedLandmark(x=x, y=y, visibility=1.0 if visible else 0.0, presence=1.0)


def pose_with_angles(index: int, knee_angle: float, elbow_angle: float) -> FrameData:
    landmarks = [landmark(0.5, 0.5, False) for _ in range(33)]

    def place(indices: tuple[int, int, int], center_x: float, center_y: float,
              angle: float, length: float = 0.10) -> None:
        first, vertex, third = indices
        radians = math.radians(-90 + angle)
        landmarks[vertex] = landmark(center_x, center_y)
        landmarks[first] = landmark(center_x, center_y - length)
        landmarks[third] = landmark(center_x + length * math.cos(radians),
                                    center_y + length * math.sin(radians))

    place((11, 13, 15), 0.35, 0.40, elbow_angle)
    place((12, 14, 16), 0.65, 0.40, elbow_angle)
    place((23, 25, 27), 0.42, 0.62, knee_angle)
    place((24, 26, 28), 0.58, 0.62, knee_angle)
    return FrameData(index, round(index * 1000 / 30), tuple(landmarks))


def test_joint_angle_geometry() -> None:
    assert joint_angle_deg(np.array([0.0, 0.0]), np.array([1.0, 0.0]),
                           np.array([2.0, 0.0])) == pytest.approx(180)
    assert joint_angle_deg(np.array([1.0, 0.0]), np.array([0.0, 0.0]),
                           np.array([0.0, 1.0])) == pytest.approx(90)


def test_knee_extension_during_ascent_is_leg_led_signal() -> None:
    count = 60
    observations = [Observation(pose_with_angles(i, 90 + 80 * i / (count - 1), 170),
                                (0.5, 0.75 - 0.002 * i)) for i in range(count)]
    metrics = [Metric(i, i / 30, (0.5, 0.75 - 0.002 * i), 0.04, 0.0,
                      'Moving', 0) for i in range(count)]
    technique, _ = TechniqueEngine(Config(), 30, 1001, 1001).analyze(
        observations, metrics, [])
    assert technique['upward_phases'] == 1
    assert technique['drive_phase_counts']['leg_led'] == 1
    assert technique['leg_signal_share'] == pytest.approx(1.0, abs=1e-6)


def test_pause_runs_and_distinct_limb_hold_uses() -> None:
    states = ['Static'] * 20 + ['Moving'] * 10 + ['Static'] * 30
    observations = [Observation(FrameData(i, round(i * 1000 / 30), ()), (0.5, 0.5))
                    for i in range(len(states))]
    metrics = [Metric(i, i / 30, (0.5, 0.5), 0.0, 0.0, state, 0)
               for i, state in enumerate(states)]
    contacts = [
        Contact(10, 10 / 30, 15, (0.4, 0.4), 'blue', 0.8, 1),
        Contact(20, 20 / 30, 16, (0.4, 0.4), 'blue', 0.8, 1),
        Contact(30, 1.0, 27, (0.5, 0.7), 'red', 0.8, 2),
    ]
    technique, coaching = TechniqueEngine(Config(), 30, 640, 480).analyze(
        observations, metrics, contacts)
    assert technique['pause_count'] == 2
    assert technique['longest_pause_s'] == pytest.approx(1.0)
    assert technique['hand_hold_uses'] == 1
    assert technique['foot_hold_uses'] == 1
    assert any(item['code'] == 'recording' for item in coaching)


def test_empty_analysis_returns_serializable_unknowns() -> None:
    technique, coaching = TechniqueEngine(Config(), 30, 640, 480).analyze([], [], [])
    assert technique['available'] is False
    assert technique['leg_signal_share'] is None
    assert coaching == [{'code': 'recording', 'kind': 'recording'}]
