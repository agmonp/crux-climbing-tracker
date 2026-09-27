"""Explainable 2D technique cues derived from pose and pelvis motion.

The arm/leg result is an association between joint motion and upward pelvis
motion. A single RGB camera cannot measure force, load, grip pressure or muscle
work, so the result must never be described as a force split.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.signal import savgol_filter

from climbing_m1 import usable

from .config import Config
from .models import Contact, Metric, Observation


def joint_angle_deg(a: np.ndarray, vertex: np.ndarray, c: np.ndarray) -> float | None:
    """Return the smaller 2D angle at ``vertex`` or None for a zero-length limb."""
    first, second = a - vertex, c - vertex
    norm = float(np.linalg.norm(first) * np.linalg.norm(second))
    if norm <= np.finfo(float).eps:
        return None
    cosine = float(np.clip(np.dot(first, second) / norm, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


class TechniqueEngine:
    """Summarize repeatable technique cues without inventing a quality score."""

    JOINTS = {
        'left_elbow': (11, 13, 15),
        'right_elbow': (12, 14, 16),
        'left_knee': (23, 25, 27),
        'right_knee': (24, 26, 28),
    }

    def __init__(self, config: Config, fps: float, width: int, height: int) -> None:
        self.config = config
        self.fps = fps
        self.pixel_scale = np.array([width - 1, height - 1], dtype=float)
        self.diagonal = float(np.linalg.norm(self.pixel_scale))

    def analyze(self, observations: list[Observation], metrics: list[Metric],
                contacts: Iterable[Contact], kinematics: dict | None = None) -> tuple[dict, list[dict]]:
        """Return serializable measurements and evidence-based coaching codes."""
        count = min(len(observations), len(metrics))
        if count == 0:
            technique = self._empty_summary()
            return technique, [{'code': 'recording', 'kind': 'recording'}]

        angles = {name: self._angle_series(observations[:count], indices)
                  for name, indices in self.JOINTS.items()}
        derivatives = {name: self._differentiate(values) for name, values in angles.items()}
        upward_velocity = self._upward_velocity(metrics[:count])
        phases = self._runs(upward_velocity > self.config.upward_speed_threshold,
                           max(1, round(self.config.upward_phase_seconds * self.fps)),
                           metrics[:count])

        left_leg = np.maximum(derivatives['left_knee'], 0)
        right_leg = np.maximum(derivatives['right_knee'], 0)
        left_arm = np.maximum(-derivatives['left_elbow'], 0)
        right_arm = np.maximum(-derivatives['right_elbow'], 0)
        leg_signal = self._row_mean(left_leg, right_leg)
        arm_signal = self._row_mean(left_arm, right_arm)

        drive_counts = {'leg_led': 0, 'arm_led': 0, 'mixed': 0, 'unresolved': 0}
        total_leg = total_arm = 0.0
        phase_details: list[dict] = []
        for start, end in phases:
            leg = float(np.nansum(leg_signal[start:end]) / self.fps)
            arm = float(np.nansum(arm_signal[start:end]) / self.fps)
            total = leg + arm
            if total < self.config.min_drive_signal_degrees:
                classification = 'unresolved'
            elif leg > arm * self.config.drive_ratio:
                classification = 'leg_led'
            elif arm > leg * self.config.drive_ratio:
                classification = 'arm_led'
            else:
                classification = 'mixed'
            drive_counts[classification] += 1
            if classification != 'unresolved':
                total_leg += leg
                total_arm += arm
            phase_details.append({
                'start_s': start / self.fps,
                'end_s': (end - 1) / self.fps,
                'classification': classification,
                'leg_extension_deg': leg,
                'arm_flexion_deg': arm,
            })

        drive_total = total_leg + total_arm
        leg_share = total_leg / drive_total if drive_total else None
        elbow_values = np.column_stack((angles['left_elbow'], angles['right_elbow']))
        static_mask = np.array([m.state == 'Static' for m in metrics[:count]])
        static_elbows = elbow_values[static_mask]
        visible_static_elbows = static_elbows[np.isfinite(static_elbows)]
        straight_share = (float(np.mean(visible_static_elbows >= self.config.straight_arm_degrees))
                          if visible_static_elbows.size else None)

        pause_runs = self._state_runs(metrics[:count], 'Static', self.config.pause_seconds)
        move_runs = self._state_runs(metrics[:count], 'Moving', self.config.move_bout_seconds)
        pause_durations = [(end - start) / self.fps for start, end in pause_runs]
        move_durations = [(end - start) / self.fps for start, end in move_runs]
        finite_speed = np.array([m.speed for m in metrics[:count] if m.speed is not None], dtype=float)
        median_speed = float(np.median(finite_speed)) if finite_speed.size else None
        speed_variability = None
        if finite_speed.size >= 4 and median_speed and median_speed > np.finfo(float).eps:
            q1, q3 = np.percentile(finite_speed, [25, 75])
            speed_variability = float((q3 - q1) / median_speed)

        horizontal, vertical = self._axis_travel(metrics[:count])
        axis_total = horizontal + vertical
        horizontal_share = horizontal / axis_total if axis_total else None
        contact_list = list(contacts)
        hand_holds = {c.hold_id for c in contact_list if c.landmark in (15, 16)}
        foot_holds = {c.hold_id for c in contact_list if c.landmark in (27, 28)}
        technique = {
            'available': any(np.isfinite(values).any() for values in angles.values()),
            'interpretation': '2D joint-motion association; not force, load or energy share',
            'upward_phases': len(phases),
            'resolved_upward_phases': sum(value for key, value in drive_counts.items()
                                            if key != 'unresolved'),
            'drive_phase_counts': drive_counts,
            'leg_signal_share': leg_share,
            'arm_signal_share': 1 - leg_share if leg_share is not None else None,
            'phase_details': phase_details,
            'straight_arm_static_fraction': straight_share,
            'pause_count': len(pause_runs),
            'longest_pause_s': max(pause_durations, default=0.0),
            'move_bout_count': len(move_runs),
            'median_move_bout_s': float(np.median(move_durations)) if move_durations else None,
            'speed_variability_iqr_over_median': speed_variability,
            'horizontal_travel_share': horizontal_share,
            'path_directness': (kinematics or {}).get('path_directness'),
            'hand_hold_uses': len(hand_holds),
            'foot_hold_uses': len(foot_holds),
            'joint_angle_coverage': {
                name: float(np.isfinite(values).mean()) for name, values in angles.items()
            },
        }
        return technique, self._coaching(technique, metrics[:count])

    def _empty_summary(self) -> dict:
        return {
            'available': False,
            'interpretation': '2D joint-motion association; not force, load or energy share',
            'upward_phases': 0,
            'resolved_upward_phases': 0,
            'drive_phase_counts': {'leg_led': 0, 'arm_led': 0, 'mixed': 0, 'unresolved': 0},
            'leg_signal_share': None,
            'arm_signal_share': None,
            'phase_details': [],
            'straight_arm_static_fraction': None,
            'pause_count': 0,
            'longest_pause_s': 0.0,
            'move_bout_count': 0,
            'median_move_bout_s': None,
            'speed_variability_iqr_over_median': None,
            'horizontal_travel_share': None,
            'path_directness': None,
            'hand_hold_uses': 0,
            'foot_hold_uses': 0,
            'joint_angle_coverage': {name: 0.0 for name in self.JOINTS},
        }

    def _angle_series(self, observations: list[Observation], indices: tuple[int, int, int]) -> np.ndarray:
        raw = np.full(len(observations), np.nan)
        for frame, observation in enumerate(observations):
            landmarks = observation.pose.landmarks
            if len(landmarks) <= max(indices) or not all(
                    usable(landmarks[index], self.config) for index in indices):
                continue
            points = [np.array([landmarks[index].x, landmarks[index].y]) * self.pixel_scale
                      for index in indices]
            angle = joint_angle_deg(points[0], points[1], points[2])
            if angle is not None:
                raw[frame] = angle
        return self._smooth(raw)

    def _smooth(self, values: np.ndarray) -> np.ndarray:
        result = values.copy()
        desired = max(3, round(self.config.technique_smoothing_seconds * self.fps))
        window = desired if desired % 2 else desired + 1
        finite = np.isfinite(values)
        for start, end in self._boolean_runs(finite):
            length = end - start
            local_window = min(window, length if length % 2 else length - 1)
            if local_window >= 3:
                result[start:end] = savgol_filter(values[start:end], local_window,
                                                   min(2, local_window - 1), mode='interp')
        return result

    def _differentiate(self, values: np.ndarray) -> np.ndarray:
        derivative = np.full_like(values, np.nan)
        for start, end in self._boolean_runs(np.isfinite(values)):
            if end - start >= 2:
                derivative[start:end] = np.gradient(values[start:end], 1 / self.fps)
        return derivative

    def _upward_velocity(self, metrics: list[Metric]) -> np.ndarray:
        values = np.full(len(metrics), np.nan)
        for start, end in self._metric_runs(metrics):
            if end - start < 2:
                continue
            y = np.array([metrics[index].com[1] for index in range(start, end)])
            y_diagonal = y * self.pixel_scale[1] / self.diagonal
            values[start:end] = -np.gradient(y_diagonal, 1 / self.fps)
        return values

    def _axis_travel(self, metrics: list[Metric]) -> tuple[float, float]:
        horizontal = vertical = 0.0
        for start, end in self._metric_runs(metrics):
            if end - start < 2:
                continue
            points = np.array([metrics[index].com for index in range(start, end)])
            delta = np.abs(np.diff(points * self.pixel_scale / self.diagonal, axis=0))
            horizontal += float(delta[:, 0].sum())
            vertical += float(delta[:, 1].sum())
        return horizontal, vertical

    @staticmethod
    def _row_mean(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        stacked = np.column_stack((first, second))
        counts = np.isfinite(stacked).sum(axis=1)
        totals = np.nansum(stacked, axis=1)
        return np.divide(totals, counts, out=np.full(len(first), np.nan), where=counts > 0)

    @staticmethod
    def _boolean_runs(mask: np.ndarray) -> list[tuple[int, int]]:
        padded = np.r_[False, mask, False].astype(np.int8)
        changes = np.diff(padded)
        return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))

    def _metric_runs(self, metrics: list[Metric]) -> list[tuple[int, int]]:
        valid = np.array([m.com is not None for m in metrics])
        runs: list[tuple[int, int]] = []
        for start, end in self._boolean_runs(valid):
            part = start
            for index in range(start + 1, end):
                if metrics[index].segment != metrics[index - 1].segment:
                    runs.append((part, index))
                    part = index
            runs.append((part, end))
        return runs

    def _runs(self, mask: np.ndarray, minimum: int,
              metrics: list[Metric]) -> list[tuple[int, int]]:
        runs: list[tuple[int, int]] = []
        for start, end in self._boolean_runs(np.asarray(mask, dtype=bool)):
            part = start
            for index in range(start + 1, end):
                if metrics[index].segment != metrics[index - 1].segment:
                    if index - part >= minimum:
                        runs.append((part, index))
                    part = index
            if end - part >= minimum:
                runs.append((part, end))
        return runs

    def _state_runs(self, metrics: list[Metric], state: str,
                    minimum_seconds: float) -> list[tuple[int, int]]:
        mask = np.array([metric.state == state for metric in metrics])
        return self._runs(mask, max(1, round(minimum_seconds * self.fps)), metrics)

    def _coaching(self, technique: dict, metrics: list[Metric]) -> list[dict]:
        measured = [m for m in metrics if m.state != 'Unknown']
        measured_fraction = len(measured) / len(metrics) if metrics else 0.0
        advice: list[dict] = []
        if not technique['available'] or measured_fraction < 0.5:
            return [{'code': 'recording', 'kind': 'recording'}]
        if (technique['resolved_upward_phases'] >= self.config.coaching_min_upward_phases
                and technique['leg_signal_share'] is not None):
            advice.append({
                'code': ('leg_drive_focus' if technique['leg_signal_share']
                         < self.config.coaching_low_leg_share else 'leg_drive_positive'),
                'kind': ('focus' if technique['leg_signal_share']
                         < self.config.coaching_low_leg_share else 'positive'),
            })
        if technique['straight_arm_static_fraction'] is not None:
            advice.append({
                'code': ('straight_arm_focus' if technique['straight_arm_static_fraction']
                         < self.config.coaching_low_straight_arm_share else 'straight_arm_positive'),
                'kind': ('focus' if technique['straight_arm_static_fraction']
                         < self.config.coaching_low_straight_arm_share else 'positive'),
            })
        static_fraction = (sum(m.state == 'Static' for m in measured) / len(measured)
                           if measured else None)
        if (static_fraction is not None
                and (static_fraction > self.config.coaching_high_static_fraction
                     or technique['pause_count'] >= 3)):
            advice.append({'code': 'pauses_focus', 'kind': 'focus'})
        variability = technique['speed_variability_iqr_over_median']
        if variability is not None and variability > self.config.coaching_high_speed_variability:
            advice.append({'code': 'rhythm_focus', 'kind': 'focus'})
        directness = technique['path_directness']
        if directness is not None and directness < self.config.coaching_low_directness:
            advice.append({'code': 'path_focus', 'kind': 'focus'})
        if not advice:
            advice.append({'code': 'limited_evidence', 'kind': 'recording'})
        return advice
