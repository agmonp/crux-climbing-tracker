"""Numerical and heuristic regression tests with known, independent outcomes."""
from dataclasses import replace

import cv2
import numpy as np
import pytest
from mediapipe.tasks.python.components.containers import NormalizedLandmark

from climbing_m1 import FrameData
from climb_app.config import Config
from climb_app.kinematics import KinematicsEngine
from climb_app.models import Observation
from climb_app.route import RouteAnalyzer


def observations(points, fps=30):
    return [Observation(FrameData(i, round(i*1000/fps), ()), p) for i, p in enumerate(points)]


@pytest.mark.parametrize('fps', [25, 30, 60])
def test_constant_velocity_is_time_scaled_and_has_zero_jerk(fps):
    t = np.arange(3*fps)/fps
    obs = observations([(0.2+0.1*x, 0.4) for x in t], fps)
    metrics, summary = KinematicsEngine(Config(), fps, 1001, 1001).analyze(obs)
    speeds = [m.speed for m in metrics if m.speed is not None]
    assert speeds == pytest.approx([0.1/np.sqrt(2)]*len(speeds), abs=1e-9)
    assert summary['rms_jerk'] < 1e-8
    assert summary['static_s'] == 0
    assert summary['moving_s'] > 2
    assert summary['path_directness'] == pytest.approx(1)


@pytest.mark.parametrize('fps', [30, 60])
def test_cubic_motion_recovers_analytic_third_derivative(fps):
    t = np.arange(3*fps)/fps
    coefficient = 0.002
    obs = observations([(0.2+coefficient*x**3, 0.4) for x in t], fps)
    _, summary = KinematicsEngine(Config(), fps, 1001, 1001).analyze(obs)
    assert summary['median_jerk'] == pytest.approx(6*coefficient/np.sqrt(2), rel=1e-7)


def test_gaps_are_unknown_and_do_not_create_velocity_spikes():
    points = [(0.2, 0.2)]*45+[None]*15+[(0.8, 0.8)]*45
    metrics, summary = KinematicsEngine(Config(), 30, 640, 480).analyze(observations(points))
    assert all(m.speed is None and m.state == 'Unknown' for m in metrics[45:60])
    assert summary['segments'] == 2
    assert summary['moving_s'] == 0
    assert summary['static_to_move_ratio'] is None
    assert summary['static_s']+summary['moving_s']+summary['unknown_s'] == pytest.approx(105/30)
    assert max(m.jerk for m in metrics if m.jerk is not None) < 1e-8


def test_no_pose_is_not_reported_as_static():
    _, summary = KinematicsEngine(Config(), 30, 640, 480).analyze(observations([None]*60))
    assert summary['pose_coverage'] == 0
    assert summary['static_fraction'] is None
    assert summary['median_jerk'] is None
    assert summary['unknown_s'] == 2


def test_jump_splits_segment_and_short_segment_has_no_derivatives():
    points = [(0.2, 0.2)]*10+[(0.8, 0.8)]*10
    metrics, summary = KinematicsEngine(Config(), 30, 640, 480).analyze(observations(points))
    assert summary['segments'] == 2
    assert all(m.speed is None for m in metrics)


def test_same_pixel_motion_on_each_axis_has_equal_speed():
    # 1280x720 frame: 100 pixels horizontally and vertically must measure equally.
    a = observations([(0.2+i/1279, 0.2) for i in range(60)])
    b = observations([(0.2, 0.2+i/719) for i in range(60)])
    engine = KinematicsEngine(Config(), 30, 1280, 720)
    assert engine.analyze(a)[1]['mean_speed'] == pytest.approx(engine.analyze(b)[1]['mean_speed'])


def solid_hsv(h, s=230, v=220):
    image = np.full((40, 40, 3), (h, s, v), dtype=np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)


@pytest.mark.parametrize('hue,name', [(0,'red'),(179,'red'),(15,'orange'),(28,'yellow'),
                                       (60,'green'),(92,'cyan'),(112,'blue'),(140,'purple'),(160,'pink')])
def test_standard_hold_colors(hue, name):
    result = RouteAnalyzer(Config(), 200, 200).classify_roi(solid_hsv(hue))
    assert result is not None and result[0] == name


def test_red_wraparound_clusters_correctly():
    roi = solid_hsv(1)
    roi[:, 20:] = solid_hsv(179)[:, 20:]
    assert RouteAnalyzer(Config(), 200, 200).classify_roi(roi)[0] == 'red'


@pytest.mark.parametrize('roi', [solid_hsv(0, 0, 200), solid_hsv(0, 0, 15), solid_hsv(12, 90, 180)])
def test_wall_dark_pixels_and_skin_do_not_force_color(roi):
    assert RouteAnalyzer(Config(), 200, 200).classify_roi(roi) is None


def frame_data(index, x=0.5, visible=True):
    landmarks = [NormalizedLandmark(x=x, y=0.5, visibility=0.0, presence=1.0) for _ in range(33)]
    landmarks[15] = NormalizedLandmark(x=x, y=0.5, visibility=1.0 if visible else 0.0, presence=1.0)
    return FrameData(index, round(index*1000/30), tuple(landmarks))


def test_contact_requires_more_than_half_second_and_votes_once():
    analyzer = RouteAnalyzer(Config(), 200, 200)
    frame = cv2.resize(solid_hsv(112), (200, 200))
    for i in range(16):
        assert not analyzer.update(frame, frame_data(i), 30)
    assert analyzer.update(frame, frame_data(16), 30)
    for i in range(17, 90):
        analyzer.update(frame, frame_data(i), 30)
    assert len(analyzer.events) == 1
    assert analyzer.summary()['distinct_holds'] == 1
    assert analyzer.summary()['color'] is None


def test_slow_drift_does_not_count_as_stationary():
    analyzer = RouteAnalyzer(Config(contact_radius_px=3), 200, 200)
    frame = cv2.resize(solid_hsv(112), (200, 200))
    for i in range(60):
        analyzer.update(frame, frame_data(i, x=0.1+i*0.003), 30)
    assert not analyzer.events


def test_lost_landmark_resets_contact_timer():
    analyzer = RouteAnalyzer(Config(), 200, 200)
    frame = cv2.resize(solid_hsv(112), (200, 200))
    for i in range(15):
        analyzer.update(frame, frame_data(i), 30)
    analyzer.update(frame, frame_data(15, visible=False), 30)
    for i in range(16, 30):
        analyzer.update(frame, frame_data(i), 30)
    assert not analyzer.events
