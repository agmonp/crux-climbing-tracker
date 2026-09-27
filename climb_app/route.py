"""Conservative color evidence from stationary wrists and ankles."""
from dataclasses import dataclass

import cv2
import numpy as np

from climbing_m1 import FrameData, Image, Point, usable
from .config import Config, HUE_RANGES
from .models import Contact


@dataclass
class ExtremityState:
    anchor: Point
    since_s: float
    last_attempt_s: float = float('-inf')
    contact: Contact | None = None


class RouteAnalyzer:
    """One vote per distinct hold, not one vote per video frame.

    Wrist/ankle coordinates are only proximity cues, not proof of gripping.
    Colored walls, clothes, skin, chalk, and adjacent holds can cause errors.
    Neutral colors are deliberately Unknown because wall/shoe separation is
    underdetermined in a tiny ROI. Never force a winning route without evidence.
    """

    def __init__(self, config: Config, width: int, height: int) -> None:
        self.config = config
        self.scale = np.array([width - 1, height - 1], dtype=float)
        self.diagonal = float(np.linalg.norm(self.scale))
        self.states: dict[int, ExtremityState] = {}
        self.events: list[Contact] = []
        self.holds: list[Contact] = []

    def classify_roi(self, roi: Image) -> tuple[str, float] | None:
        if roi.size == 0:
            return None
        cfg = self.config
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = (hsv[..., i].astype(float) for i in range(3))
        keep = (s >= cfg.min_saturation) & (v >= cfg.min_value)
        if cfg.mask_skin:
            skin = ((h >= cfg.skin_hue_low) & (h <= cfg.skin_hue_high)
                    & (s >= cfg.skin_saturation_low) & (s <= cfg.skin_saturation_high)
                    & (v >= cfg.skin_value_low))
            keep &= ~skin
        # Use ROI border as an approximate wall sample, only when its median is
        # sufficiently achromatic. A colorful border may itself be the hold.
        border = np.ones(h.shape, dtype=bool)
        b = cfg.wall_border_px
        border[b:-b, b:-b] = False
        wall = np.median(hsv[border], axis=0)
        if wall[1] < cfg.min_saturation:
            dh = np.minimum(abs(h-wall[0]), 180-abs(h-wall[0]))
            background = ((dh < cfg.wall_hue_distance)
                          & (abs(s-wall[1]) < cfg.wall_saturation_distance)
                          & (abs(v-wall[2]) < cfg.wall_value_distance))
            keep &= ~background
        pixels = hsv[keep].astype(np.float32)
        if len(pixels) < cfg.min_color_pixels:
            return None
        # Embed hue on a circle so reds on opposite sides of 0/180 cluster together.
        angle = pixels[:, 0] * (2*np.pi/180)
        saturation = pixels[:, 1]/255
        features = np.column_stack((np.cos(angle)*saturation, np.sin(angle)*saturation,
                                    pixels[:, 2]/255)).astype(np.float32)
        cv2.setRNGSeed(cfg.random_seed)
        _, labels, _ = cv2.kmeans(features, cfg.kmeans_k, None,
                                  (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                                   cfg.kmeans_iterations, cfg.kmeans_epsilon),
                                  cfg.kmeans_attempts, cv2.KMEANS_PP_CENTERS)
        # Merge two clusters if lighting split the same nominal hue into both.
        votes: dict[str, int] = {}
        for k in range(cfg.kmeans_k):
            group = pixels[labels.ravel() == k]
            if not len(group):
                continue
            a = group[:, 0] * (2*np.pi/180)
            hue = float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()) * 180/(2*np.pi)) % 180
            name = next(name for name, lo, hi in HUE_RANGES if lo <= hue < hi)
            votes[name] = votes.get(name, 0) + len(group)
        name = max(votes, key=votes.get)
        dominance = votes[name]/len(pixels)
        if dominance < cfg.min_color_dominance:
            return None
        # This is an evidence score, not a calibrated statistical probability.
        support = min(1.0, len(pixels)/(cfg.min_color_pixels * 3))
        return name, float(dominance * support)

    def update(self, frame: Image, data: FrameData, fps: float) -> tuple[Contact, ...]:
        now = data.index/fps
        active: list[Contact] = []
        for limb in self.config.extremities:
            if len(data.landmarks) <= limb or not usable(data.landmarks[limb], self.config):
                self.states.pop(limb, None)
                continue
            lm = data.landmarks[limb]
            point = (lm.x, lm.y)
            state = self.states.get(limb)
            if state is None or np.linalg.norm((np.array(point)-state.anchor)*self.scale) > self.config.contact_radius_px:
                state = ExtremityState(point, now)
                self.states[limb] = state
            if (now-state.since_s > self.config.contact_seconds and state.contact is None
                    and now-state.last_attempt_s >= self.config.contact_retry_seconds):
                state.last_attempt_s = now
                x, y = np.rint(np.array(point)*self.scale).astype(int)
                half = self.config.roi_size//2
                x0, y0 = max(0, x-half), max(0, y-half)
                roi = frame[y0:min(frame.shape[0], y-half+self.config.roi_size),
                            x0:min(frame.shape[1], x-half+self.config.roi_size)]
                detected = self.classify_roi(roi)
                if detected is not None:
                    color, confidence = detected
                    nearby = next((hold for hold in self.holds
                                   if np.linalg.norm((np.array(point)-hold.position)*self.scale)
                                   / self.diagonal < self.config.distinct_hold_radius), None)
                    hold_id = nearby.hold_id if nearby is not None else len(self.holds)
                    event = Contact(data.index, now, limb, point, color, confidence, hold_id)
                    if nearby is None:
                        self.holds.append(event)
                    self.events.append(event)
                    state.contact = event
            if state.contact is not None:
                active.append(state.contact)
        return tuple(active)

    def summary(self) -> dict:
        votes: dict[str, float] = {}
        for hold in self.holds:
            votes[hold.color] = votes.get(hold.color, 0.0) + hold.confidence
        total = sum(votes.values())
        candidate = max(votes, key=votes.get) if votes else None
        share = votes[candidate]/total if candidate else 0.0
        accepted = len(self.holds) >= self.config.min_route_contacts and share >= self.config.min_route_share
        return {'color': candidate if accepted else None, 'candidate': candidate,
                'evidence_share': share, 'distinct_holds': len(self.holds),
                'contact_events': len(self.events), 'votes': votes,
                'status': 'estimated' if accepted else 'insufficient_evidence'}
