import math

import numpy as np
import pytest

from r1_perception.mapper import SCAN_OFFSETS, HeightMapper

PELVIS_Z = 0.74


def mapper_on_flat_ground_with_a_step(yaw=0.0):
    """Ground at 0, a 0.144 m step where the robot's forward x > 0.3 m."""
    hm = HeightMapper()
    quat = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    hm.on_lowstate(np.zeros(hm.model.num_motors), quat, tick_s=0.0, wall=0.0)
    hm.odometry.position[:] = [0.0, 0.0, PELVIS_Z]
    hm.history[-1] = (*hm.history[-1][:3], hm.odometry.position.copy())
    xy = np.mgrid[-1.5:1.5:0.02, -1.5:1.5:0.02].reshape(2, -1).T
    forward = xy @ np.array([math.cos(yaw), math.sin(yaw)])
    hm.map.insert(np.c_[xy, np.where(forward > 0.3, 0.144, 0.0)], stamp=0.0)
    return hm


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2])
def test_scan_order_and_values_follow_the_training(yaw):
    scan = mapper_on_flat_ground_with_a_step(yaw).scan()
    assert scan.shape == (187,)
    # Index = row (y) * 17 + column (x): the step is in the front columns of every row.
    expect = np.where(SCAN_OFFSETS[:, 0] > 0.32, PELVIS_Z - 0.144, PELVIS_Z)
    near_edge = np.abs(SCAN_OFFSETS[:, 0] - 0.3) < 0.05
    assert np.allclose(scan[~near_edge], expect[~near_edge], atol=1e-9)


def test_unknown_points_get_the_median():
    hm = mapper_on_flat_ground_with_a_step()
    hm.map.height[:] = np.nan
    hm.map.insert(np.array([[0.01, 0.01, 0.0]]), stamp=0.0)
    assert np.allclose(hm.scan(), PELVIS_Z)


def test_height_fix_pulls_the_odometry_towards_the_seen_ground():
    hm = mapper_on_flat_ground_with_a_step()
    hm.odometry.position[2] += 0.05            # drifted 5 cm up
    hm._stance = hm.odometry.stance
    hm.map.insert(np.c_[np.mgrid[-1:1:0.02, -1:1:0.02].reshape(2, -1).T, np.zeros(10000)], stamp=0.0)
    foot = hm.model.foot_positions()[hm._stance] + hm.odometry.position
    before = foot[2] - hm.model.sole_below_ankle
    hm._fix_height()
    after = hm.odometry.position[2] + hm.model.foot_positions()[hm._stance][2] - hm.model.sole_below_ankle
    assert abs(after) == pytest.approx(abs(before) * (1 - hm.height_fix_gain), abs=1e-6)


def test_samples_without_an_imu_quaternion_are_ignored():
    hm = HeightMapper()
    assert not hm.on_lowstate(np.zeros(hm.model.num_motors), (0, 0, 0, 0), tick_s=0.0, wall=0.0)
    assert hm.scan() is None
    assert hm.on_lowstate(np.zeros(hm.model.num_motors), (1, 0, 0, 0), tick_s=0.002, wall=0.002)
    assert np.all(np.isfinite(hm.odometry.position))
