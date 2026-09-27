"""Milestone 1: single-climber pose and smoothed pelvis trajectory.

Python 3.10+. Suggested layout (paths are relative to your working directory):
    climbing_m1.py
    requirements.txt
    models/pose_landmarker_full.task
    input/climb.mp4
    output/annotated.mp4

Install: python -m pip install -r requirements.txt
Run: python climbing_m1.py input/climb.mp4 output/annotated.mp4
Download the official model using the command in README.md first.
OpenCV exports silent video; audio is not copied. Use constant-frame-rate input.
The hip midpoint is a 2D pelvis proxy, not a biomechanical whole-body CoM.
"""

from __future__ import annotations

import argparse
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.components.containers import NormalizedLandmark
from mediapipe.tasks.python import vision
from numpy.typing import NDArray


@dataclass(frozen=True)
class Config:
    """All tunable model, smoothing, and rendering parameters; colors are BGR."""

    model_path: Path = Path("models/pose_landmarker_full.task")
    visibility_threshold: float = 0.5
    presence_threshold: float = 0.5
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5
    pose_presence_confidence: float = 0.5
    smoothing_window: int = 7
    trail_seconds: float = 2.0
    trail_opacity: float = 0.9
    trail_color: tuple[int, int, int] = (0, 200, 255)
    skeleton_color: tuple[int, int, int] = (80, 220, 80)
    joint_color: tuple[int, int, int] = (255, 220, 100)
    com_color: tuple[int, int, int] = (0, 0, 255)
    skeleton_thickness: int = 2
    trail_thickness: int = 3
    joint_radius: int = 3
    com_radius: int = 6
    left_hip_index: int = 23
    right_hip_index: int = 24
    codec: str = "mp4v"
    progress_interval: int = 300

    def __post_init__(self) -> None:
        for value in (self.visibility_threshold, self.presence_threshold,
                      self.detection_confidence, self.tracking_confidence,
                      self.pose_presence_confidence, self.trail_opacity):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Confidence thresholds and opacity must be in [0, 1].")
        if self.smoothing_window < 1 or self.trail_seconds <= 0:
            raise ValueError("Smoothing window and trail duration must be positive.")
        if not np.isfinite(self.trail_seconds):
            raise ValueError("Trail duration must be finite.")
        if min(self.skeleton_thickness, self.trail_thickness, self.joint_radius,
               self.com_radius, self.progress_interval) < 1:
            raise ValueError("Drawing dimensions and progress interval must be positive.")
        if self.codec != "mp4v":
            raise ValueError("This milestone exports using mp4v.")


Point = tuple[float, float]
Image = NDArray[np.uint8]


@dataclass(frozen=True)
class FrameData:
    """Pose observations for one source frame; positions are normalized."""

    index: int
    timestamp_ms: int
    landmarks: tuple[NormalizedLandmark, ...]


@dataclass(frozen=True)
class TrailPoint:
    frame_index: int
    position: Point


@dataclass(frozen=True)
class ClimberState:
    """Current smoothed pelvis position and bounded historical observations."""

    com: Point | None
    trail: tuple[TrailPoint, ...]


def usable(landmark: NormalizedLandmark, config: Config) -> bool:
    """Reject uncertain or off-image landmarks instead of clamping them."""
    return bool(
        np.isfinite([landmark.x, landmark.y]).all()
        and 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1
        and landmark.visibility is not None
        and landmark.visibility >= config.visibility_threshold
        and landmark.presence is not None
        and landmark.presence >= config.presence_threshold
    )


class PoseEstimator:
    """Own the BlazePose-based MediaPipe Tasks video-mode landmarker."""

    def __init__(self, config: Config) -> None:
        if not config.model_path.is_file():
            raise FileNotFoundError(f"Missing pose model: {config.model_path}. See README.md.")
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(config.model_path.resolve())),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=config.detection_confidence,
            min_pose_presence_confidence=config.pose_presence_confidence,
            min_tracking_confidence=config.tracking_confidence,
            output_segmentation_masks=False,
        )
        self._model = vision.PoseLandmarker.create_from_options(options)

    def estimate(self, frame: Image, index: int, timestamp_ms: int) -> FrameData:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._model.detect_for_video(image, timestamp_ms)
        landmarks = tuple(result.pose_landmarks[0]) if result.pose_landmarks else ()
        return FrameData(index, timestamp_ms, landmarks)

    def close(self) -> None:
        self._model.close()


class PelvisTracker:
    """Causal moving average; a detection gap resets smoothing and breaks lines."""

    def __init__(self, config: Config, fps: float) -> None:
        self.config = config
        self.trail_frames = max(1, int(round(config.trail_seconds * fps)))
        self._samples: deque[Point] = deque(maxlen=config.smoothing_window)
        self._trail: deque[TrailPoint] = deque(maxlen=self.trail_frames + 1)

    def update(self, data: FrameData) -> ClimberState:
        while self._trail and data.index - self._trail[0].frame_index >= self.trail_frames:
            self._trail.popleft()
        indices = (self.config.left_hip_index, self.config.right_hip_index)
        if (len(data.landmarks) <= max(indices)
                or not all(usable(data.landmarks[i], self.config) for i in indices)):
            self._samples.clear()
            return ClimberState(None, tuple(self._trail))
        left, right = (data.landmarks[i] for i in indices)
        midpoint = ((left.x + right.x) / 2, (left.y + right.y) / 2)
        self._samples.append(midpoint)
        average = np.mean(self._samples, axis=0)
        com = (float(average[0]), float(average[1]))
        self._trail.append(TrailPoint(data.index, com))
        return ClimberState(com, tuple(self._trail))


class FrameRenderer:
    """Draw only reliable skeleton edges and age-faded trajectory segments."""

    def __init__(self, config: Config, trail_frames: int) -> None:
        self.config = config
        self.trail_frames = trail_frames

    def render(self, frame: Image, data: FrameData, state: ClimberState) -> Image:
        height, width = frame.shape[:2]

        def pixel(point: Point) -> tuple[int, int]:
            return round(point[0] * (width - 1)), round(point[1] * (height - 1))

        points = {
            i: pixel((landmark.x, landmark.y))
            for i, landmark in enumerate(data.landmarks)
            if usable(landmark, self.config)
        }
        for edge in vision.PoseLandmarksConnections.POSE_LANDMARKS:
            if edge.start in points and edge.end in points:
                cv2.line(frame, points[edge.start], points[edge.end],
                         self.config.skeleton_color, self.config.skeleton_thickness,
                         cv2.LINE_AA)
        for point in points.values():
            cv2.circle(frame, point, self.config.joint_radius,
                       self.config.joint_color, cv2.FILLED, cv2.LINE_AA)

        # Draw alpha into a single mask: old segments become transparent,
        # rather than darkening the underlying footage or accumulating ghosts.
        mask = np.zeros((height, width), dtype=np.uint8)
        for previous, current in zip(state.trail, state.trail[1:]):
            if current.frame_index != previous.frame_index + 1:
                continue  # Never connect observations across missing detections.
            age = data.index - current.frame_index
            opacity = self.config.trail_opacity * max(0.0, 1 - age / self.trail_frames)
            cv2.line(mask, pixel(previous.position), pixel(current.position),
                     round(255 * opacity), self.config.trail_thickness, cv2.LINE_AA)
        alpha = mask.astype(np.float32)[..., None] / 255
        color = np.asarray(self.config.trail_color, dtype=np.float32)
        frame[:] = np.clip(frame * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)
        if state.com is not None:
            cv2.circle(frame, pixel(state.com), self.config.com_radius,
                       self.config.com_color, cv2.FILLED, cv2.LINE_AA)
        return frame


class VideoProcessor:
    """Stream frames with bounded history; release all resources even on failure."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def process(self, source: Path, destination: Path) -> int:
        if not source.is_file():
            raise FileNotFoundError(f"Input video does not exist: {source}")
        if source.resolve() == destination.resolve():
            raise ValueError("Input and output paths must differ.")
        if destination.suffix.lower() != ".mp4":
            raise ValueError("Output must have an .mp4 extension.")
        if destination.exists():
            raise FileExistsError(f"Output already exists: {destination}")
        capture = cv2.VideoCapture(str(source))
        writer = None
        estimator = None
        count = 0
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open input video: {source}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not np.isfinite(fps) or not 0 < fps <= 1000:
                raise ValueError(f"Unsupported or missing video FPS: {fps}")
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("Input contains no decodable frames.")
            height, width = frame.shape[:2]
            # mp4v may silently crop odd dimensions; pad only the bottom/right.
            output_size = (width + width % 2, height + height % 2)
            estimator = PoseEstimator(self.config)
            tracker = PelvisTracker(self.config, fps)
            renderer = FrameRenderer(self.config, tracker.trail_frames)
            destination.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(destination),
                                     cv2.VideoWriter_fourcc(*self.config.codec),
                                     fps, output_size)
            if not writer.isOpened():
                raise RuntimeError("Cannot initialize mp4v writer; check codec and output path.")
            while ok and frame is not None:
                if frame.shape[:2] != (height, width):
                    raise RuntimeError("Input resolution changed during decoding.")
                timestamp_ms = round(count * 1000 / fps)
                data = estimator.estimate(frame, count, timestamp_ms)
                state = tracker.update(data)
                annotated = renderer.render(frame, data, state)
                if output_size != (width, height):
                    annotated = cv2.copyMakeBorder(
                        annotated, 0, height % 2, 0, width % 2, cv2.BORDER_REPLICATE)
                writer.write(annotated)
                count += 1
                if count % self.config.progress_interval == 0:
                    logging.info("Processed %d frames", count)
                ok, frame = capture.read()
            # OpenCV cannot reliably distinguish EOF from a decoding failure.
            expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if expected > count:
                logging.warning("Decoded %d of %d reported frames; input may be truncated.",
                                count, expected)
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if estimator is not None:
                estimator.close()
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError("Video writer produced no output.")
        logging.info("Saved %d frames to %s", count, destination)
        return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, default=Config.model_path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        VideoProcessor(Config(model_path=args.model)).process(args.input, args.output)
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        logging.error("Processing failed: %s", exc)
        return 1
    except KeyboardInterrupt:
        logging.warning("Interrupted; output may contain only the frames processed so far.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
