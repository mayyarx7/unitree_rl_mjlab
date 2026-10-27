import numpy as np

from r1_perception.elevation_map import ElevationMap


def test_insert_keeps_the_highest_point_per_cell_and_newer_frames_replace():
    m = ElevationMap(resolution=0.1, size=4.0)
    m.insert(np.array([[0.05, 0.05, 0.1], [0.06, 0.04, 0.3], [1.05, 0.05, 0.2]]), stamp=0.0)
    assert m.lookup(np.array([[0.05, 0.05], [1.05, 0.05]])).tolist() == [0.3, 0.2]
    m.insert(np.array([[0.05, 0.05, 0.0]]), stamp=1.0)
    assert m.lookup(np.array([[0.05, 0.05]]))[0] == 0.0


def test_cells_are_forgotten_after_the_memory():
    m = ElevationMap(resolution=0.1, size=4.0, memory_s=5.0)
    m.insert(np.array([[0.05, 0.05, 0.1]]), stamp=0.0)
    m.insert(np.array([[0.55, 0.05, 0.1]]), stamp=4.0)
    m.expire(now=6.0)
    assert np.isnan(m.lookup(np.array([[0.05, 0.05]]))[0])
    assert m.lookup(np.array([[0.55, 0.05]]))[0] == 0.1


def test_recentring_keeps_cells_in_place_in_the_world():
    m = ElevationMap(resolution=0.1, size=4.0)
    m.insert(np.array([[1.55, -0.45, 0.2]]), stamp=0.0)
    m.recentre([1.5, 0.0])
    assert m.lookup(np.array([[1.55, -0.45]]))[0] == 0.2
    m.recentre([30.0, 0.0])  # far away: everything dropped
    assert np.all(np.isnan(m.height))


def test_ground_near_is_the_median_around_a_point():
    m = ElevationMap(resolution=0.04, size=4.0)
    xy = np.mgrid[-0.2:0.2:0.04, -0.2:0.2:0.04].reshape(2, -1).T + 0.02
    m.insert(np.c_[xy, np.full(len(xy), 0.144)], stamp=0.0)
    assert abs(m.ground_near([0.0, 0.0], 0.06) - 0.144) < 1e-9
    assert m.ground_near([2.0, 2.0], 0.06) is None
