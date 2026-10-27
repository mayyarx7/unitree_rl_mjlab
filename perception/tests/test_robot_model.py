import math

import numpy as np
import pytest

from r1_perception.robot_model import RobotModel

IDENTITY = (1.0, 0.0, 0.0, 0.0)


@pytest.fixture(scope="module")
def model():
    return RobotModel()


def test_motor_order_starts_with_the_legs(model):
    assert model.num_motors == 31
    assert model.motor_names[:3] == ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint"]
    assert "head_pitch_joint" in model.motor_names


def test_camera_above_and_ahead_of_the_pelvis_looking_forward(model):
    model.update(np.zeros(model.num_motors), IDENTITY)
    pos, rot = model.camera_pose()
    assert 0.3 < pos[2] < 0.5 and 0.0 < pos[0] < 0.2
    assert np.allclose(-rot[:, 2], [1, 0, 0], atol=1e-6)  # looks along +x


def test_head_pitch_tilts_the_camera_down(model):
    q = np.zeros(model.num_motors)
    q[model.motor_names.index("head_pitch_joint")] = 0.5
    model.update(q, IDENTITY)
    view = -model.camera_pose()[1][:, 2]
    assert view[2] == pytest.approx(-math.sin(0.5), abs=1e-6)


def test_imu_yaw_turns_everything(model):
    half = math.radians(90) / 2
    model.update(np.zeros(model.num_motors), (math.cos(half), 0, 0, math.sin(half)))
    view = -model.camera_pose()[1][:, 2]
    assert np.allclose(view, [0, 1, 0], atol=1e-6)


def test_feet_below_the_pelvis(model):
    model.update(np.zeros(model.num_motors), IDENTITY)
    feet = model.foot_positions()
    assert np.all(feet[:, 2] < -0.6) and feet[0, 1] > feet[1, 1]  # left foot on the left


def test_points_on_the_body_are_found(model):
    model.update(np.zeros(model.num_motors), IDENTITY)
    feet = model.foot_positions()
    points = np.array([feet[0], [1.5, 0.0, -0.7]])  # on the left foot, the floor ahead
    assert model.on_body(points).tolist() == [True, False]


def on_body_reference(model, points, margin=0.03):
    """The plain loop over the geoms, which on_body must match exactly."""
    inside = np.zeros(len(points), dtype=bool)
    for g in model.geoms:
        rot = model.d.geom_xmat[g].reshape(3, 3)
        local = (points - model.d.geom_xpos[g]) @ rot - model.m.geom_aabb[g, :3]
        inside |= np.all(np.abs(local) <= model.m.geom_aabb[g, 3:] + margin, axis=1)
    return inside


@pytest.mark.parametrize("pose", ["stand", "leg_lifted", "head_down_turned"])
def test_on_body_matches_the_reference(model, pose):
    q = np.zeros(model.num_motors)
    quat = IDENTITY
    if pose == "leg_lifted":
        q[model.motor_names.index("left_hip_pitch_joint")] = -0.8
        q[model.motor_names.index("left_knee_joint")] = 1.2
    elif pose == "head_down_turned":
        q[model.motor_names.index("head_pitch_joint")] = 0.45
        half = math.radians(30) / 2
        quat = (math.cos(half), 0.0, 0.0, math.sin(half))
    model.update(q, quat)
    rng = np.random.default_rng(0)
    # Far and near points, plus points just inside and outside each geom's grown box.
    points = [rng.uniform(-3, 3, (5000, 3)), rng.uniform(-0.5, 0.5, (5000, 3)) - [0, 0, 0.4]]
    for g in model.geoms:
        rot = model.d.geom_xmat[g].reshape(3, 3)
        half = model.m.geom_aabb[g, 3:] + 0.03
        local = rng.uniform(-1, 1, (50, 3)) * half * rng.choice([0.99, 1.01], (50, 1))
        points.append((local + model.m.geom_aabb[g, :3]) @ rot.T + model.d.geom_xpos[g])
    points = np.concatenate(points)
    expected = on_body_reference(model, points)
    assert 0 < expected.sum() < len(points)
    assert np.array_equal(model.on_body(points), expected)
