"""Two-pass video processing: observations, offline analytics, annotated export."""
from __future__ import annotations

import csv
import json
import math
import subprocess
import threading
import time
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import cv2
import imageio_ffmpeg
import numpy as np

from climbing_m1 import ClimberState, FrameData, FrameRenderer, PoseEstimator, TrailPoint, usable
from .config import Config, PALETTE
from .kinematics import KinematicsEngine
from .models import AnalysisResult, Observation
from .route import RouteAnalyzer
from .technique import TechniqueEngine

Progress = Callable[[str, float], None]
Crop = tuple[float, float, float, float]


class AnalysisCancelled(Exception):
    """Raised between frames when the user cancels a job."""


def video_metadata(path: Path, config: Config) -> dict:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError('Cannot open this video. Use a valid MP4, MOV, or WebM.')
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or not 0 < fps <= config.max_fps:
            raise ValueError(f'Video FPS must be between 0 and {config.max_fps:g}.')
        if min(width, height) < 2 or frames < 1:
            raise ValueError('Video has no valid frames or dimensions.')
        if frames > config.max_frames or frames/fps > config.max_seconds:
            raise ValueError(f'Use a clip up to {config.max_seconds:g} seconds / {config.max_frames} frames.')
        return {'fps': fps, 'frames': frames, 'width': width, 'height': height, 'duration_s': frames/fps}
    finally:
        capture.release()


def encode_preview(source: Path, target: Path, config: Config,
                   cancel: threading.Event | None = None) -> None:
    """Keep mp4v as the canonical export; create H.264 separately for browsers."""
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-v', 'error', '-i', str(source),
               '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', str(config.preview_crf),
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)]
    # File logging prevents a pipe filling during a long encode. Never use shell=True.
    with target.with_suffix('.log').open('wb') as log:
        flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=log, creationflags=flags)
        started = time.monotonic()
        try:
            while process.poll() is None:
                if cancel is not None and cancel.wait(0.1):
                    raise AnalysisCancelled()
                if cancel is None:
                    time.sleep(0.1)
                if time.monotonic()-started > config.preview_timeout_seconds:
                    raise RuntimeError('Browser preview encoding timed out.')
            if process.returncode:
                raise RuntimeError('Browser preview encoding failed. See preview.log.')
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()


class VideoProcessor:
    """Bounded clip analysis with no decoded frames retained in memory.

    Only landmarks and per-frame metrics are retained for the offline smoothing
    pass. Crop coordinates select the climber's full movement area. Coordinates
    are mapped back to the complete image for rendering and measurements.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def process(self, source: Path, output: Path, crop: Crop | None = None,
                progress: Progress | None = None,
                cancel: threading.Event | None = None) -> AnalysisResult:
        cfg = self.config
        progress = progress or (lambda stage, value: None)
        cancel = cancel or threading.Event()
        meta = video_metadata(source, cfg)
        if crop is not None:
            if (len(crop) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in crop)
                    or crop[2]-crop[0] < 0.05 or crop[3]-crop[1] < 0.05):
                raise ValueError('Crop must be x0,y0,x1,y1 in [0,1], at least 5% on each axis.')
        output.mkdir(parents=True, exist_ok=True)
        if any((output/name).exists() for name in ('annotated.mp4', 'report.json')):
            raise FileExistsError('Output already contains an analysis; choose a new directory.')
        factor = min(1.0, cfg.max_dimension/max(meta['width'], meta['height']))
        width = max(2, round(meta['width']*factor)//2*2)
        height = max(2, round(meta['height']*factor)//2*2)
        fps = meta['fps']
        route = RouteAnalyzer(cfg, width, height)
        capture = cv2.VideoCapture(str(source))
        estimator = None
        observations: list[Observation] = []
        camera_moves, camera_pairs = 0, 0
        previous_gray = None
        warnings: list[str] = []
        started = time.monotonic()
        try:
            estimator = PoseEstimator(cfg)
            while True:
                if cancel.is_set():
                    raise AnalysisCancelled()
                ok, frame = capture.read()
                if not ok:
                    break
                index = len(observations)
                if index >= cfg.max_frames or index/fps >= cfg.max_seconds:
                    raise ValueError('Decoded video exceeds the supported clip limit.')
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                timestamp = round(index*1000/fps)
                if crop is not None:
                    x0, y0, x1, y1 = (round(v*d) for v, d in zip(crop, (width, height, width, height)))
                    data = estimator.estimate(frame[y0:y1, x0:x1], index, timestamp)
                    landmarks = tuple(replace(lm, x=(x0+lm.x*(x1-x0-1))/(width-1),
                                              y=(y0+lm.y*(y1-y0-1))/(height-1))
                                      for lm in data.landmarks)
                    data = FrameData(index, timestamp, landmarks)
                else:
                    data = estimator.estimate(frame, index, timestamp)
                hips = (cfg.left_hip_index, cfg.right_hip_index)
                com = None
                if len(data.landmarks) > max(hips) and all(usable(data.landmarks[i], cfg) for i in hips):
                    a, b = (data.landmarks[i] for i in hips)
                    com = ((a.x+b.x)/2, (a.y+b.y)/2)
                contacts = route.update(frame, data, fps)
                observations.append(Observation(data, com, contacts))
                # Camera-motion hint only. The climber/lighting can also cause a
                # phase shift, so this is not presented as measured camera pose.
                gray = cv2.cvtColor(cv2.resize(frame, (cfg.camera_size, cfg.camera_size)), cv2.COLOR_BGR2GRAY).astype(np.float32)
                if previous_gray is not None:
                    shift, response = cv2.phaseCorrelate(previous_gray, gray)
                    if response >= cfg.camera_response_min:
                        camera_pairs += 1
                        camera_moves += np.linalg.norm(shift)/cfg.camera_size > cfg.camera_shift_warning
                previous_gray = gray
                if index % 10 == 0:
                    progress('pose', min(0.6, 0.6*(index+1)/meta['frames']))
        finally:
            capture.release()
            if estimator is not None:
                estimator.close()
        if not observations:
            raise ValueError('No frames could be decoded.')
        if len(observations) < meta['frames']:
            warnings.append('Input decoding ended before the reported frame count; results cover decoded frames only.')
        progress('metrics', 0.62)
        metrics, summary = KinematicsEngine(cfg, fps, width, height).analyze(observations)
        technique, coaching = TechniqueEngine(cfg, fps, width, height).analyze(
            observations, metrics, route.events, summary)
        summary.update({'source_name': source.name, 'width': width, 'height': height,
                        'source_width': meta['width'], 'source_height': meta['height'],
                        'route': route.summary(), 'technique': technique,
                        'coaching': coaching, 'crop': crop,
                        'camera_motion_fraction': camera_moves/camera_pairs if camera_pairs else None})
        if camera_pairs and camera_moves/camera_pairs > cfg.camera_fraction_warning:
            warnings.append('Possible camera movement: speed, jerk and hold deduplication may be unreliable. Use a fixed camera.')
        if summary['pose_coverage'] < 0.7:
            warnings.append('Low hip visibility: much of the clip is untracked. Try a clearer view or a climber crop.')
        if summary['metric_coverage'] < 0.5:
            warnings.append('Limited continuous tracking: less than half the clip supports reliable derivatives.')
        if summary['route']['color'] is None:
            warnings.append('Not enough consistent hold evidence to estimate the route color.')
        self._render(source, output/'annotated.mp4', observations, metrics, summary, cancel, progress)
        progress('preview', 0.92)
        encode_preview(output/'annotated.mp4', output/'preview.mp4', cfg, cancel)
        summary['processing_seconds'] = time.monotonic()-started
        result = AnalysisResult(summary, metrics, route.events, warnings)
        config_dict = asdict(cfg)
        config_dict['model_path'] = str(cfg.model_path)
        payload = {**asdict(result), 'config': config_dict,
                   'method': ('2D pelvis proxy, local cubic derivatives, 2D joint-motion associations, '
                              'HSV contact-color heuristic; no force measurement or calibrated fluidity score.')}
        (output/'report.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
        with (output/'metrics.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'time_s', 'com_x', 'com_y', 'speed_diagonal_s', 'jerk_diagonal_s3', 'state', 'segment'])
            for m in metrics:
                writer.writerow([m.frame_index, m.time_s, *(m.com or ('', '')), m.speed, m.jerk, m.state, m.segment])
        progress('complete', 1.0)
        return result

    def _render(self, source: Path, destination: Path, observations: list[Observation],
                metrics: list, summary: dict, cancel: threading.Event, progress: Progress) -> None:
        cfg = self.config
        width, height, fps = summary['width'], summary['height'], summary['fps']
        trail_frames = max(1, round(cfg.trail_seconds*fps))
        renderer = FrameRenderer(cfg, trail_frames)
        trail: deque[TrailPoint] = deque(maxlen=trail_frames)
        capture = cv2.VideoCapture(str(source))
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*cfg.codec), fps, (width, height))
        try:
            if not writer.isOpened():
                raise RuntimeError('Cannot create mp4v output.')
            for i, (obs, metric) in enumerate(zip(observations, metrics)):
                if cancel.is_set():
                    raise AnalysisCancelled()
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError('Input failed to decode during the rendering pass.')
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                if i and metrics[i-1].segment != metric.segment:
                    trail.clear()
                while trail and i-trail[0].frame_index >= trail_frames:
                    trail.popleft()
                if metric.com is not None:
                    trail.append(TrailPoint(i, metric.com))
                renderer.render(frame, obs.pose, ClimberState(metric.com, tuple(trail)))
                for contact in obs.contacts:
                    x, y = round(contact.position[0]*(width-1)), round(contact.position[1]*(height-1))
                    color = PALETTE[contact.color]
                    cv2.circle(frame, (x, y), cfg.route_circle_radius, color, 2, cv2.LINE_AA)
                    cv2.putText(frame, contact.color, (max(0, x-20), max(15, y-27)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                # Draw the dashboard at a canonical size, then scale to fit narrow video.
                panel = np.full((cfg.dashboard_height, cfg.dashboard_width, 3), cfg.dashboard_background, dtype=np.uint8)
                speed = f'{metric.speed:.3f} diag/s' if metric.speed is not None else '--'
                jerk = f'{metric.jerk:.2f} diag/s^3' if metric.jerk is not None else '--'
                fraction = summary['static_fraction']
                static = f'{fraction:.0%}' if fraction is not None else '--'
                lines = ['CRUX / CLIMB ANALYSIS', f'{metric.time_s:5.1f}s   {metric.state.upper()}',
                         f'Speed  {speed}', f'Jerk   {jerk}',
                         f'Static {static} (measured time)',
                         f"Route  {summary['route']['color'] or 'Unknown'} (clip estimate)"]
                for row, line in enumerate(lines):
                    cv2.putText(panel, line, (12, 23+row*26), cv2.FONT_HERSHEY_SIMPLEX,
                                cfg.dashboard_font_scale, cfg.text_color, 1, cv2.LINE_AA)
                size = min(1.0, (width-16)/cfg.dashboard_width, (height/3)/cfg.dashboard_height)
                panel = cv2.resize(panel, (max(1, round(cfg.dashboard_width*size)), max(1, round(cfg.dashboard_height*size))))
                ph, pw = panel.shape[:2]
                region = frame[8:8+ph, 8:8+pw]
                cv2.addWeighted(panel, cfg.dashboard_opacity, region, 1-cfg.dashboard_opacity, 0, region)
                writer.write(frame)
                if i == len(observations)//2:
                    cv2.imwrite(str(destination.parent/'poster.jpg'), frame)
                if i % 10 == 0:
                    progress('render', 0.64+0.26*(i+1)/len(observations))
        finally:
            capture.release()
            writer.release()
        # Verify the finalized file decodes, not just that VideoWriter opened.
        check = cv2.VideoCapture(str(destination))
        try:
            ok, _ = check.read()
            if not ok or int(check.get(cv2.CAP_PROP_FRAME_COUNT)) != len(observations):
                raise RuntimeError('Export validation failed: missing or unreadable frames.')
        finally:
            check.release()
