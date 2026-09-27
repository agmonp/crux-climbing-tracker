"""Data exchanged between inference, analytics, rendering and exports."""
from dataclasses import dataclass, field
from typing import Any

from climbing_m1 import FrameData, Point


@dataclass(frozen=True)
class Contact:
    frame_index: int
    time_s: float
    landmark: int
    position: Point  # Normalized to output image width/height.
    color: str
    confidence: float
    hold_id: int


@dataclass
class Observation:
    pose: FrameData
    raw_com: Point | None
    contacts: tuple[Contact, ...] = ()


@dataclass(frozen=True)
class Metric:
    frame_index: int
    time_s: float
    com: Point | None
    speed: float | None
    jerk: float | None
    state: str  # Unknown is deliberately distinct from Static.
    segment: int


@dataclass
class AnalysisResult:
    summary: dict[str, Any]
    metrics: list[Metric]
    contacts: list[Contact]
    warnings: list[str] = field(default_factory=list)
