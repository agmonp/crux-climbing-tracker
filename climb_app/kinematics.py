"""Offline, gap-aware kinematics in image-diagonal units."""
from dataclasses import asdict

import numpy as np
from scipy.signal import savgol_filter

from .config import Config
from .models import Metric, Observation


class KinematicsEngine:
    """Fit local cubics to contiguous observations, then differentiate the fits.

    Each pixel axis is divided by the SAME image diagonal, avoiding aspect-ratio
    distortion. Third derivatives amplify noise: use a fixed time window and
    exclude the half-window at both edges from speed/jerk summaries. Missing
    poses and implausible jumps split segments; no gap is interpolated.
    """

    def __init__(self, config: Config, fps: float, width: int, height: int) -> None:
        self.config, self.fps = config, fps
        self.scale = np.array([width - 1, height - 1]) / np.hypot(width - 1, height - 1)

    def analyze(self, observations: list[Observation]) -> tuple[list[Metric], dict]:
        count = len(observations)
        raw = np.array([o.raw_com if o.raw_com is not None else (np.nan, np.nan)
                        for o in observations], dtype=float)
        points = raw * self.scale
        valid = np.isfinite(points).all(axis=1)
        smooth = np.full_like(points, np.nan)
        speed, jerk = np.full(count, np.nan), np.full(count, np.nan)
        segments = np.full(count, -1, dtype=int)
        runs: list[list[int]] = []
        for i in range(count):
            if not valid[i]:
                continue
            if (i == 0 or not valid[i-1]
                    or np.linalg.norm(points[i] - points[i-1]) > self.config.max_step):
                runs.append([])
            runs[-1].append(i)
        desired = max(self.config.polynomial_order + 2,
                      round(self.config.smoothing_seconds * self.fps))
        window = desired if desired % 2 else desired + 1
        half = window // 2
        path_length, net_distance = 0.0, 0.0
        for segment, indices in enumerate(runs):
            segments[indices] = segment
            smooth[indices] = points[indices]
            if len(indices) < window:
                continue
            fitted = savgol_filter(points[indices], window, self.config.polynomial_order,
                                   axis=0, mode='interp')
            velocity = savgol_filter(points[indices], window, self.config.polynomial_order,
                                     deriv=1, delta=1/self.fps, axis=0, mode='interp')
            third = savgol_filter(points[indices], window, self.config.polynomial_order,
                                  deriv=3, delta=1/self.fps, axis=0, mode='interp')
            smooth[indices] = fitted
            interior = indices[half:-half]
            speed[interior] = np.linalg.norm(velocity[half:-half], axis=1)
            jerk[interior] = np.linalg.norm(third[half:-half], axis=1)
            tracked = fitted[half:-half]
            if len(tracked) > 1:
                path_length += float(np.linalg.norm(np.diff(tracked, axis=0), axis=1).sum())
                net_distance += float(np.linalg.norm(tracked[-1] - tracked[0]))
        metrics = []
        for i in range(count):
            has_speed = np.isfinite(speed[i])
            com = tuple(map(float, smooth[i] / self.scale)) if valid[i] else None
            metrics.append(Metric(i, i/self.fps, com,
                                  float(speed[i]) if has_speed else None,
                                  float(jerk[i]) if np.isfinite(jerk[i]) else None,
                                  ('Static' if speed[i] < self.config.static_speed else 'Moving')
                                  if has_speed else 'Unknown', int(segments[i])))
        static = sum(m.state == 'Static' for m in metrics) / self.fps
        moving = sum(m.state == 'Moving' for m in metrics) / self.fps
        measured = static + moving
        unknown = count/self.fps - measured
        finite_speed, finite_jerk = speed[np.isfinite(speed)], jerk[np.isfinite(jerk)]
        summary = {
            'duration_s': count/self.fps, 'frames': count, 'fps': self.fps,
            'pose_coverage': float(valid.mean()) if count else 0.0,
            'metric_coverage': measured / (count/self.fps) if count else 0.0,
            'static_s': static, 'moving_s': moving, 'unknown_s': max(0.0, unknown),
            'static_fraction': static/measured if measured else None,
            'static_to_move_ratio': static/moving if moving else None,
            'mean_speed': float(finite_speed.mean()) if finite_speed.size else None,
            'median_jerk': float(np.median(finite_jerk)) if finite_jerk.size else None,
            'rms_jerk': float(np.sqrt(np.mean(finite_jerk**2))) if finite_jerk.size else None,
            'path_length': path_length,
            'path_directness': net_distance/path_length if path_length > 0 else None,
            'segments': len(runs), 'smoothing_window_frames': window,
            'smoothing_window_seconds': window/self.fps,
            'speed_unit': 'image diagonals/s', 'jerk_unit': 'image diagonals/s^3',
            'static_threshold': self.config.static_speed,
        }
        return metrics, summary
