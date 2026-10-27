"""The mapping chain on depth images rendered by MuJoCo: the robot slides towards a EUR pallet
and the scan it would hand the policy is compared with the true one. The pose is given exactly
here: this tests the mapping, not the odometry. Needs an EGL-capable GPU.

The true scan treats the pallet as a solid block, as the policy's training terrain is: the 4 cm
gaps between the top boards fall inside one map cell and the map closes them."""

import math
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from r1_perception.mapper import SCAN_OFFSETS, HeightMapper, Intrinsics
from r1_perception.robot_model import DEFAULT_MJCF

os.environ.setdefault("MUJOCO_GL", "egl")
PALLET = DEFAULT_MJCF.parent / "euro_pallet.xml"
W, H = 544, 448
PALLET_X = 1.9   # centre; turned 90 deg, so its 1200 mm side faces the robot at x = 1.5


def add_pallet(body):
    for geom in ET.parse(PALLET).find("worldbody").iter("geom"):
        x, y, z = map(float, geom.get("pos").split())
        sx, sy, sz = map(float, geom.get("size").split())
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-y, x, z], size=[sy, sx, sz])  # turned 90 deg


def scene(with_robot: bool):
    spec = mujoco.MjSpec.from_file(str(DEFAULT_MJCF)) if with_robot else mujoco.MjSpec()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[10, 10, 0.1])
    if with_robot:
        add_pallet(spec.worldbody.add_body(pos=[PALLET_X, 0, 0]))
    else:  # the pallet as a block, turned: 0.8 m deep in x, 1.2 m wide in y
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[PALLET_X, 0, 0.072], size=[0.4, 0.6, 0.072])
    return spec.compile()


def true_scan(m, d, pelvis):
    heights = []
    for ox, oy in SCAN_OFFSETS:
        geom = np.zeros(1, np.int32)
        dist = mujoco.mj_ray(m, d, np.array([pelvis[0] + ox, pelvis[1] + oy, 2.0]),
                             np.array([0, 0, -1.0]), None, 1, -1, geom)
        heights.append(pelvis[2] - (2.0 - dist))
    return np.array(heights)


def walk_to_pallet():
    """Scan, true scan and the mapper after sliding from x = 0 to 1.1 m (pallet edge 0.4 m ahead)."""
    m = scene(with_robot=True)
    d = mujoco.MjData(m)
    hm = HeightMapper()
    q = np.zeros(hm.model.num_motors)
    hm.model.update(q, (1, 0, 0, 0))
    pelvis_z = -hm.model.foot_positions()[0, 2] + hm.model.sole_below_ankle   # soles on the floor
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "depth_camera")
    f = 0.5 * H / math.tan(math.radians(m.cam_fovy[cam]) / 2)
    k = Intrinsics(f, f, (W - 1) / 2, (H - 1) / 2)
    renderer = mujoco.Renderer(m, H, W)
    renderer.enable_depth_rendering()
    for i, x in enumerate(np.arange(0.0, 1.1001, 0.05)):
        pelvis = np.array([x, 0.0, pelvis_z])
        d.qpos[:3], d.qpos[3:7] = pelvis, (1, 0, 0, 0)
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, camera="depth_camera")
        depth = renderer.render()
        t = i * 0.1
        hm.on_lowstate(q, (1, 0, 0, 0), tick_s=t, wall=t)
        hm.odometry.position[:] = pelvis               # exact pose (odometry is tested elsewhere)
        hm.history[-1] = (*hm.history[-1][:3], pelvis.copy())
        hm.on_depth(depth, k, wall=t)
    truth_m = scene(with_robot=False)
    truth_d = mujoco.MjData(truth_m)
    mujoco.mj_forward(truth_m, truth_d)
    return hm.scan(), true_scan(truth_m, truth_d, pelvis), hm


@pytest.fixture(scope="module")
def walked_up_to_the_pallet():
    return walk_to_pallet()


def test_the_pallet_edge_and_top_are_in_the_scan(walked_up_to_the_pallet):
    scan, truth, hm = walked_up_to_the_pallet
    pelvis = hm.history[-1][3]
    seen = np.isfinite(hm.map.lookup(pelvis[:2] + SCAN_OFFSETS))
    off_edge = np.abs(pelvis[0] + SCAN_OFFSETS[:, 0] - (PALLET_X - 0.4)) > 0.03
    on_pallet = truth < truth.max() - 0.1
    err = np.abs(scan - truth)[seen & off_edge]
    print(f"\nseen {seen.sum()} of 187 points ({(seen & on_pallet).sum()} of {on_pallet.sum()} on the pallet); "
          f"error there: median {np.median(err)*100:.1f} cm, max {err.max()*100:.1f} cm")
    assert np.all(seen[on_pallet & off_edge])     # every pallet point ahead is in the map
    assert np.median(err) < 0.01 and err.max() < 0.03


def test_the_robot_itself_is_not_in_the_map(walked_up_to_the_pallet):
    _, _, hm = walked_up_to_the_pallet
    known = hm.map.height[np.isfinite(hm.map.height)]
    assert known.max() < 0.2   # nothing above the pallet's 0.144 m: no arm or head in the map
