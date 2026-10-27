"""The R1's kinematics from the simulator's own MJCF: the same model in sim and on the robot.

Frame of everything returned here: origin at the pelvis, axes of the world (the IMU's orientation,
yaw included), so z is up. Joint angles and the IMU quaternion come from rt/lowstate.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

DEFAULT_MJCF = Path(__file__).resolve().parents[2] / "src/assets/robots/unitree_r1/xmls/R1_C++.xml"
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


class RobotModel:
    def __init__(self, mjcf: Path | str = DEFAULT_MJCF, camera: str = "depth_camera"):
        self.m = mujoco.MjModel.from_xml_path(str(mjcf))
        self.d = mujoco.MjData(self.m)
        # rt/lowstate's motor i is the model's i-th jointpos sensor (the simulator publishes them so).
        joints = [self.m.sensor_objid[s] for s in range(self.m.nsensor)
                  if self.m.sensor_type[s] == mujoco.mjtSensor.mjSENS_JOINTPOS]
        self.motor_names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints]
        self.motor_qadr = np.array([self.m.jnt_qposadr[j] for j in joints])
        self.camera = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if self.camera < 0:
            raise ValueError(f"camera {camera} not in {mjcf}")
        self.feet = np.array([mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, f) for f in FEET])
        # The robot's own geoms, to drop depth points on its body.
        self.geoms = np.array([g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] > 0])
        # Sole below the ankle origin: the bottom of the foot's contact spheres, standing upright.
        self.update(np.zeros(self.num_motors), (1.0, 0.0, 0.0, 0.0))
        spheres = [g for g in self.geoms if self.m.geom_bodyid[g] == self.feet[0]
                   and self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
        bottom = min(self.d.geom_xpos[g][2] - self.m.geom_size[g][0] for g in spheres)
        self.sole_below_ankle = float(self.d.xpos[self.feet[0]][2] - bottom)

    @property
    def num_motors(self) -> int:
        return len(self.motor_qadr)

    def update(self, q_motor, quat_wxyz) -> None:
        """Pose from rt/lowstate: motor angles (in motor order) and the IMU quaternion (w, x, y, z)."""
        q = np.asarray(q_motor, dtype=float)
        if q.shape != (self.num_motors,):
            raise ValueError(f"{q.shape[0]} motor angles, the model has {self.num_motors}")
        self.d.qpos[:] = self.m.qpos0
        self.d.qpos[:3] = 0.0
        self.d.qpos[3:7] = np.asarray(quat_wxyz, dtype=float) / np.linalg.norm(quat_wxyz)
        self.d.qpos[self.motor_qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_camlight(self.m, self.d)

    def camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Camera position and rotation (columns: MuJoCo camera x right, y up, looking along -z)."""
        return self.d.cam_xpos[self.camera].copy(), self.d.cam_xmat[self.camera].reshape(3, 3).copy()

    def foot_positions(self) -> np.ndarray:
        """(2, 3) left and right ankle-roll origins."""
        return self.d.xpos[self.feet].copy()

    def on_body(self, points: np.ndarray, margin: float = 0.03) -> np.ndarray:
        """Mask of the (N, 3) points inside one of the robot's geoms' boxes grown by margin."""
        inside = np.zeros(len(points), dtype=bool)
        for g in self.geoms:
            centre_local, half = self.m.geom_aabb[g, :3], self.m.geom_aabb[g, 3:] + margin
            rot = self.d.geom_xmat[g].reshape(3, 3)
            local = (points - self.d.geom_xpos[g]) @ rot - centre_local
            inside |= np.all(np.abs(local) <= half, axis=1)
        return inside
