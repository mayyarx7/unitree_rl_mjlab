"""Leg odometry: the pelvis moves opposite to its stance foot, which stays put on the ground.

Feet come from RobotModel.foot_positions(), relative to the pelvis in world-aligned axes, so the
displacement is in the world frame directly; the IMU gives the orientation (yaw included). Each
update moves the pelvis by the stance foot's own change since the previous update.

The stance foot changes to the other foot once that one hardly moves up or down (below vz_max
relative to the pelvis) and is either the lower foot or the stance foot is lifting off. Without
the speed condition the foot still coming down counts as standing as soon as the stance heel
lifts: in the simulator that lost 20 % of the distance and gained 1 cm of height per second.
Without the lift-off case a foot already on a pallet never takes over from the lower one leaving
the floor. Knee torques picked the foot on the ground less often (83 %) than height did (100 %).

Feed one update per new rt/lowstate sample (stamp: its tick in seconds); repeated stamps are skipped.
"""

from __future__ import annotations

import numpy as np


class LegOdometry:
    def __init__(self, vz_max: float = 0.15):
        self.vz_max = vz_max
        self.position = np.zeros(3)  # pelvis in the odometry frame (start: origin)
        self.stance: int | None = None
        self._feet: np.ndarray | None = None
        self._stamp: float | None = None

    def update(self, feet: np.ndarray, stamp: float) -> np.ndarray:
        """feet: (2, 3) ankles relative to the pelvis, world-aligned axes. Returns the pelvis position."""
        feet = np.asarray(feet, dtype=float)
        if self._feet is None:
            self.stance = int(feet[1, 2] < feet[0, 2])
        elif stamp > self._stamp:
            vz = (feet[:, 2] - self._feet[:, 2]) / (stamp - self._stamp)
            other = 1 - self.stance
            still = abs(vz[other]) < self.vz_max
            lifting = vz[self.stance] > self.vz_max
            if still and (feet[other, 2] < feet[self.stance, 2] or lifting):
                self.stance = other
            self.position += self._feet[self.stance] - feet[self.stance]
        else:
            return self.position.copy()
        self._feet = feet
        self._stamp = stamp
        return self.position.copy()

    def reset(self) -> None:
        self.position[:] = 0.0
        self.stance = None
        self._feet = None
        self._stamp = None
