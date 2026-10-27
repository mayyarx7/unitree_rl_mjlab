import numpy as np

from r1_perception.odometry import LegOdometry

PELVIS_HEIGHT = 0.74


def walk(steps, step_length=0.25, rise_at=None, rise=0.0, samples=50, dt=0.002):
    """Feet relative to the pelvis for a straight walk; from step rise_at on the feet land rise higher."""
    feet_world = np.array([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]])
    pelvis = np.array([0.0, 0.0, PELVIS_HEIGHT])
    out = []
    for k in range(steps):
        swing = 1 - k % 2
        start, target = feet_world[swing].copy(), feet_world[swing].copy()
        target[0] += step_length if k else step_length / 2
        if rise_at is not None and k >= rise_at:
            target[2] = rise
        for i in range(samples):
            s = (i + 1) / samples
            feet_world[swing] = start + s * (target - start)
            feet_world[swing, 2] = start[2] + s * (target[2] - start[2]) + 0.08 * np.sin(np.pi * s)
            stance_z = feet_world[1 - swing, 2]
            pelvis = np.array([feet_world[:, 0].mean(), 0.0, PELVIS_HEIGHT + stance_z])
            out.append((feet_world - pelvis, pelvis.copy(), len(out) * dt))
    return out


def run(samples):
    odo = LegOdometry()
    for feet, pelvis, stamp in samples:
        est = odo.update(feet, stamp)
    return est, pelvis - samples[0][1]


def test_straight_walk_on_flat_ground():
    est, truth = run(walk(9))
    assert np.allclose(est, truth, atol=0.01)


def test_stepping_onto_a_pallet_raises_the_estimate():
    est, truth = run(walk(10, rise_at=5, rise=0.144))
    assert np.allclose(est, truth, atol=0.01)
    assert abs(est[2] - 0.144) < 0.01


def test_repeated_samples_change_nothing():
    samples = walk(4)
    doubled = [s for s in samples for _ in range(2)]
    assert np.allclose(run(doubled)[0], run(samples)[0])
