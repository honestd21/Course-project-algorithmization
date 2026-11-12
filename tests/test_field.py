import numpy as np
from simulator.field import Field, neighbors


def test_empty_field():
    f = Field(3, 4)
    assert f.weed.shape == (3, 4)
    assert f.obstacles.shape == (3, 4)
    assert not f.obstacles.any()


def test_random_same_seed_same_field():
    a = Field.random(20, 20, seed=1)
    b = Field.random(20, 20, seed=1)
    assert np.array_equal(a.weed, b.weed)
    assert np.array_equal(a.obstacles, b.obstacles)


def test_random_weed_in_range():
    f = Field.random(20, 20, seed=1)
    assert f.weed.min() >= 0
    assert f.weed.max() <= 1


def test_no_weed_on_obstacles():
    f = Field.random(20, 20, seed=1)
    assert (f.weed[f.obstacles] == 0).all()


def test_start_cell_is_free():
    for seed in range(100):
        f = Field.random(5, 5, n_obstacles=20, seed=seed, start=(0, 0))
        assert not f.obstacles[0, 0]


def test_in_bounds():
    f = Field(3, 4)
    assert f.in_bounds(0, 0)
    assert f.in_bounds(2, 3)
    assert not f.in_bounds(-1, 0)
    assert not f.in_bounds(3, 0)
    assert not f.in_bounds(0, 4)


def test_is_free():
    obstacles = np.zeros((3, 3), dtype=bool)
    obstacles[1, 1] = True
    f = Field(3, 3, obstacles=obstacles)
    assert f.is_free(0, 0)
    assert not f.is_free(1, 1)    # препятствие
    assert not f.is_free(-1, 0)   # за границей


def test_targets():
    weed = np.array([[0.9, 0.2],
                     [0.5, 0.7]])
    obstacles = np.array([[False, False],
                          [False, True]])
    f = Field(2, 2, weed, obstacles)
    # (1,1) имеет 0.7, но это препятствие, поэтому не цель
    assert sorted(f.targets(0.5)) == [(0, 0), (1, 0)]


def test_neighbors_in_corner():
    f = Field(3, 3)
    assert sorted(neighbors(f, 0, 0)) == [(0, 1), (1, 0)]


def test_neighbors_skip_obstacle():
    obstacles = np.zeros((3, 3), dtype=bool)
    obstacles[0, 1] = True
    f = Field(3, 3, obstacles=obstacles)
    assert neighbors(f, 0, 0) == [(1, 0)]