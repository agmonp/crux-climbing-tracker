"""Explicit configuration for the complete analysis pipeline."""
from dataclasses import dataclass
from pathlib import Path

from climbing_m1 import Config as PoseConfig

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config(PoseConfig):
    model_path: Path = ROOT / 'models' / 'pose_landmarker_full.task'
    max_dimension: int = 960
    max_seconds: float = 300.0
    max_frames: int = 36000
    max_fps: float = 120.0
    smoothing_seconds: float = 0.55
    polynomial_order: int = 3
    static_speed: float = 0.012  # Image diagonals / second, not metres / second.
    max_step: float = 0.12  # Break tracking at implausible inter-frame displacement.
    extremities: tuple[int, ...] = (15, 16, 27, 28)
    contact_seconds: float = 0.5
    contact_radius_px: float = 8.0  # Relative to the anchor, not just the previous frame.
    contact_retry_seconds: float = 0.4
    roi_size: int = 40
    min_color_pixels: int = 35
    min_saturation: int = 65
    min_value: int = 45
    skin_hue_low: int = 0
    skin_hue_high: int = 22
    skin_saturation_low: int = 30
    skin_saturation_high: int = 165
    skin_value_low: int = 65
    mask_skin: bool = True
    wall_border_px: int = 3
    wall_hue_distance: float = 8.0
    wall_saturation_distance: float = 35.0
    wall_value_distance: float = 40.0
    min_color_dominance: float = 0.55
    kmeans_k: int = 2
    kmeans_iterations: int = 30
    kmeans_epsilon: float = 0.1
    kmeans_attempts: int = 3
    random_seed: int = 42
    distinct_hold_radius: float = 0.025  # Image diagonals.
    min_route_contacts: int = 3
    min_route_share: float = 0.65
    route_circle_radius: int = 23
    camera_size: int = 160
    camera_response_min: float = 0.2
    camera_shift_warning: float = 0.002
    camera_fraction_warning: float = 0.12
    technique_smoothing_seconds: float = 0.35
    upward_speed_threshold: float = 0.006  # Image diagonals / second, upward.
    upward_phase_seconds: float = 0.20
    drive_ratio: float = 1.25
    min_drive_signal_degrees: float = 8.0
    straight_arm_degrees: float = 155.0
    pause_seconds: float = 0.50
    move_bout_seconds: float = 0.15
    coaching_min_upward_phases: int = 2
    coaching_low_leg_share: float = 0.40
    coaching_low_straight_arm_share: float = 0.45
    coaching_high_static_fraction: float = 0.35
    coaching_low_directness: float = 0.45
    coaching_high_speed_variability: float = 0.90
    dashboard_width: int = 370
    dashboard_height: int = 170
    dashboard_opacity: float = 0.8
    dashboard_font_scale: float = 0.55
    text_color: tuple[int, int, int] = (235, 238, 240)
    dashboard_background: tuple[int, int, int] = (25, 23, 20)
    preview_crf: int = 23
    preview_timeout_seconds: int = 600

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.15 <= self.smoothing_seconds <= 2:
            raise ValueError('Smoothing must be between 0.15 and 2 seconds.')
        if not 0 < self.static_speed <= 0.5:
            raise ValueError('Static speed threshold must be in (0, 0.5].')
        if not 4 <= self.roi_size <= 200:
            raise ValueError('ROI size must be between 4 and 200 pixels.')
        if not 1 <= self.contact_radius_px <= 50:
            raise ValueError('Contact radius must be between 1 and 50 pixels.')
        if self.polynomial_order < 3 or self.kmeans_k != 2:
            raise ValueError('Third derivatives require order >= 3; color clustering uses k=2.')
        shares = (self.coaching_low_leg_share, self.coaching_low_straight_arm_share,
                  self.coaching_high_static_fraction, self.coaching_low_directness)
        if not all(0 <= value <= 1 for value in shares):
            raise ValueError('Technique shares and fractions must be in [0, 1].')
        if min(self.technique_smoothing_seconds, self.upward_speed_threshold,
               self.upward_phase_seconds, self.min_drive_signal_degrees,
               self.pause_seconds, self.move_bout_seconds) <= 0:
            raise ValueError('Technique timing, speed and signal thresholds must be positive.')
        if self.drive_ratio <= 1 or not 90 <= self.straight_arm_degrees <= 180:
            raise ValueError('Drive ratio must exceed 1 and straight-arm angle must be in [90, 180].')
        if self.coaching_min_upward_phases < 1 or self.coaching_high_speed_variability <= 0:
            raise ValueError('Technique evidence thresholds must be positive.')


# OpenCV hue spans [0, 180); red wraps around the seam.
HUE_RANGES = (
    ('red', 0, 8), ('orange', 8, 22), ('yellow', 22, 36),
    ('green', 36, 85), ('cyan', 85, 100), ('blue', 100, 128),
    ('purple', 128, 150), ('pink', 150, 173), ('red', 173, 180),
)
PALETTE = {
    'red': (70, 65, 235), 'orange': (35, 150, 245), 'yellow': (40, 225, 245),
    'green': (80, 190, 75), 'cyan': (215, 205, 45), 'blue': (235, 120, 60),
    'purple': (195, 85, 155), 'pink': (180, 100, 245),
}
