"""The perception chain without DDS: rt/lowstate samples and depth frames in, the height scan out.

The scan is what Unitree-R1-Perceptive's actor observes (src/tasks/velocity/config/r1/env_cfgs.py):
17 x 11 points, 0.1 m apart, x -0.8..0.8 and y -0.5..0.5 m around the pelvis turned with its yaw,
ordered as mjlab's GridPatternCfg (rows along y, x within a row); each value is the pelvis height
minus the ground height there, in metres (the policy's deploy config scales it); unknown points
get the median of the known ones, as the training fills its holes.

Times: wall clock (time.time()) for matching a depth frame to the robot's pose, rt/lowstate's
tick for the odometry's foot speeds.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .elevation_map import ElevationMap
from .odometry import LegOdometry
from .robot_model import RobotModel

SCAN_X = np.arange(-0.8, 0.8 + 0.05, 0.1)
SCAN_Y = np.arange(-0.5, 0.5 + 0.05, 0.1)
_gx, _gy = np.meshgrid(SCAN_X, SCAN_Y, indexing="xy")
SCAN_OFFSETS = np.stack([_gx.ravel(), _gy.ravel()], axis=1)  # (187, 2)


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole, OpenCV convention (pixel centres at integers)."""
    fx: float
    fy: float
    cx: float
    cy: float


def depth_to_points(depth: np.ndarray, k: Intrinsics, stride: int = 4, max_range: float = 4.0) -> np.ndarray:
    """Depth along the optical axis in metres (0 or NaN: none) to (N, 3) points in MuJoCo's camera
    frame (x right, y up, looking along -z); every stride-th pixel."""
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[::stride, ::stride]
    ok = np.isfinite(z) & (z > 0) & (z < max_range)
    z, u, v = z[ok], u[ok], v[ok]
    return np.stack([(u - k.cx) / k.fx * z, -(v - k.cy) / k.fy * z, -z], axis=1)


def yaw_of(quat_wxyz) -> float:
    w, x, y, z = quat_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class HeightMapper:
    def __init__(self, model: RobotModel | None = None, elevation: ElevationMap | None = None,
                 history_s: float = 1.0, max_match_s: float = 0.05,
                 height_fix_gain: float = 0.5, height_fix_radius: float = 0.06):
        self.model = model or RobotModel()
        self.map = elevation or ElevationMap()
        self.odometry = LegOdometry()
        self.history: deque = deque()   # (wall, q, quat, pelvis position)
        self.history_s = history_s
        self.max_match_s = max_match_s
        self.height_fix_gain = height_fix_gain
        self.height_fix_radius = height_fix_radius
        self.height_fixes: list[float] = []   # corrections applied, for diagnostics
        self._stance: int | None = None

    # ---- rt/lowstate ---------------------------------------------------------------------------
    def on_lowstate(self, q, quat, tick_s: float, wall: float) -> bool:
        """One rt/lowstate sample; False (ignored) without a valid IMU quaternion, e.g. before the
        robot or simulator publishes real data."""
        if not 0.5 < float(np.linalg.norm(quat)) < 1.5:
            return False
        self.model.update(q, quat)
        self.odometry.update(self.model.foot_positions(), tick_s)
        if self.odometry.stance != self._stance:
            self._stance = self.odometry.stance
            self._fix_height()
        self.history.append((wall, np.array(q, dtype=float), np.array(quat, dtype=float),
                             self.odometry.position.copy()))
        while self.history and self.history[0][0] < wall - self.history_s:
            self.history.popleft()
        return True

    def _fix_height(self) -> None:
        """A foot just took the weight: its sole is on the ground the map saw there earlier. The
        difference is height the odometry drifted since; take out part of it."""
        foot = self.odometry.position + self.model.foot_positions()[self._stance]
        ground = self.map.ground_near(foot[:2], self.height_fix_radius)
        if ground is None:
            return
        error = (foot[2] - self.model.sole_below_ankle) - ground
        self.odometry.position[2] -= self.height_fix_gain * error
        self.height_fixes.append(error)

    # ---- depth -----------------------------------------------------------------------------------
    def _state_at(self, wall: float):
        if not self.history:
            return None
        best = min(self.history, key=lambda s: abs(s[0] - wall))
        return best if abs(best[0] - wall) <= self.max_match_s else None

    def on_depth(self, depth: np.ndarray, k: Intrinsics, wall: float) -> int:
        """One depth frame (metres) captured at wall; returns the cells it set (0: no matching pose)."""
        state = self._state_at(wall)
        if state is None:
            return 0
        _, q, quat, position = state
        latest = self.history[-1]
        self.model.update(q, quat)
        cam_pos, cam_rot = self.model.camera_pose()
        points = depth_to_points(depth, k) @ cam_rot.T + cam_pos   # pelvis-centred, world axes
        points = points[~self.model.on_body(points)]
        self.model.update(latest[1], latest[2])
        self.map.recentre(position)
        cells = self.map.insert(points + position, wall)
        self.map.expire(wall)
        return cells

    # ---- output ----------------------------------------------------------------------------------
    def scan(self) -> np.ndarray | None:
        """(187,) pelvis height minus ground height at the training's scan points; None before the
        first rt/lowstate sample."""
        if not self.history:
            return None
        _, _, quat, position = self.history[-1]
        yaw = yaw_of(quat)
        c, s = math.cos(yaw), math.sin(yaw)
        xy = position[:2] + SCAN_OFFSETS @ np.array([[c, s], [-s, c]])
        heights = position[2] - self.map.lookup(xy)
        known = np.isfinite(heights)
        if known.any():
            fill = float(np.median(heights[known]))
        else:
            # Nothing seen: flat ground at the stance sole.
            fill = -(self.model.foot_positions()[self.odometry.stance][2] - self.model.sole_below_ankle)
        heights[~known] = fill
        return heights
