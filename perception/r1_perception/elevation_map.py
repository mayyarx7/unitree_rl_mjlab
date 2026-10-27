"""A height map in the odometry frame that moves along with the robot.

Each cell holds the highest surface measured in it by the latest depth frame that saw it; a newer
frame replaces an older one, and a cell nobody saw for memory_s seconds is forgotten (NaN). The
window recentres on the robot by whole cells, so stored cells keep their place in the world.
"""

from __future__ import annotations

import numpy as np


class ElevationMap:
    def __init__(self, resolution: float = 0.04, size: float = 6.0, memory_s: float = 5.0):
        self.resolution = resolution
        self.n = int(round(size / resolution))
        self.memory_s = memory_s
        self.height = np.full((self.n, self.n), np.nan)  # [ix, iy]
        self.stamp = np.full((self.n, self.n), -np.inf)
        self.origin = np.array([-size / 2, -size / 2])   # odometry xy of cell (0, 0)'s corner

    def recentre(self, xy) -> None:
        """Shift the window so that xy is near its centre; cells outside it are dropped."""
        centre = self.origin + self.n * self.resolution / 2
        shift = np.round((np.asarray(xy[:2]) - centre) / self.resolution).astype(int)
        if np.all(np.abs(shift) < self.n // 4):
            return
        if np.any(np.abs(shift) >= self.n):
            self.height[:] = np.nan
            self.stamp[:] = -np.inf
            self.origin += shift * self.resolution
            return
        for axis, s in enumerate(shift):
            if s == 0:
                continue
            for grid, empty in ((self.height, np.nan), (self.stamp, -np.inf)):
                moved = np.roll(grid, -s, axis=axis)
                index = [slice(None), slice(None)]
                index[axis] = slice(self.n - s, None) if s > 0 else slice(None, -s)
                moved[tuple(index)] = empty
                grid[...] = moved
        self.origin += shift * self.resolution

    def cells(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cell indices of (N, 2) points and whether they lie in the window."""
        idx = np.floor((xy - self.origin) / self.resolution).astype(int)
        inside = np.all((idx >= 0) & (idx < self.n), axis=1)
        return idx[:, 0], idx[:, 1], inside

    def insert(self, points: np.ndarray, stamp: float) -> int:
        """(N, 3) odometry-frame points of one depth frame; returns how many cells they set."""
        ix, iy, inside = self.cells(points[:, :2])
        ix, iy, z = ix[inside], iy[inside], points[inside, 2]
        top = np.full((self.n, self.n), -np.inf)
        np.maximum.at(top, (ix, iy), z)
        seen = np.isfinite(top)
        self.height[seen] = top[seen]
        self.stamp[seen] = stamp
        return int(seen.sum())

    def expire(self, now: float) -> None:
        self.height[self.stamp < now - self.memory_s] = np.nan

    def lookup(self, xy: np.ndarray) -> np.ndarray:
        """Heights at (N, 2) points, NaN where unknown or outside the window."""
        ix, iy, inside = self.cells(np.asarray(xy, dtype=float).reshape(-1, 2))
        out = np.full(len(ix), np.nan)
        out[inside] = self.height[ix[inside], iy[inside]]
        return out

    def ground_near(self, xy, radius: float) -> float | None:
        """Median known height within radius of xy, or None."""
        r = int(np.ceil(radius / self.resolution))
        offsets = np.arange(-r, r + 1) * self.resolution
        gx, gy = np.meshgrid(offsets, offsets, indexing="ij")
        disc = np.hypot(gx, gy) <= radius
        pts = np.stack([gx[disc] + xy[0], gy[disc] + xy[1]], axis=1)
        h = self.lookup(pts)
        h = h[np.isfinite(h)]
        return float(np.median(h)) if h.size else None
