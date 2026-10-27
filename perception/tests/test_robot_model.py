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
