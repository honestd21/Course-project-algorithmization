"""Тесты routing/: запуск `python -m pytest tests/test_routing.py` или `python tests/test_routing.py`."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing import astar_path, bfs_path, route_nearest_neighbor, route_snake, route_two_opt  # noqa: E402
from simulator.field import Field  # noqa: E402  (Field участника B)

START = (0, 0)
ALGORITHMS = [route_snake, route_nearest_neighbor, route_two_opt]


def make_field(seed, h=30, w=30, p_obstacle=0.15):
    """Плотные случайные препятствия: жёсткая проверка обхода."""
    rng = np.random.default_rng(seed)
    obstacles = rng.random((h, w)) < p_obstacle
    obstacles[START] = False
    return Field(h, w, weed=rng.random((h, w)), obstacles=obstacles)


def make_targets(field, seed, threshold=0.8):
    cells = np.argwhere((field.weed >= threshold) & ~field.obstacles)
    return [tuple(int(x) for x in c) for c in cells]


def check_path(field, path, targets):
    assert path[0] == START
    for a, b in zip(path, path[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, f"не соседние клетки {a}->{b}"
    for r, c in path:
        assert not field.obstacles[r, c], f"путь через препятствие {(r, c)}"
    visited = set(path)
    reachable = set(c for c in targets if bfs_path(field, START, c) is not None)
    assert reachable <= visited, "не все достижимые цели посещены"


def test_all_algorithms_valid():
    for seed in range(5):
        field = make_field(seed)
        targets = make_targets(field, seed)
        for algo in ALGORITHMS:
            check_path(field, algo(field, targets, START), targets)


def test_empty_targets():
    field = make_field(0)
    for algo in ALGORITHMS:
        assert algo(field, [], START) == [START]


def test_field_not_modified():
    field = make_field(1)
    weed, obs = field.weed.copy(), field.obstacles.copy()
    for algo in ALGORITHMS:
        algo(field, make_targets(field, 1), START)
    assert np.array_equal(field.weed, weed) and np.array_equal(field.obstacles, obs)


def test_astar_equals_bfs_length():
    field = make_field(2)
    goals = make_targets(field, 2, threshold=0.9)[:30]
    for g in goals:
        p_bfs, p_astar = bfs_path(field, START, g), astar_path(field, START, g)
        assert (p_bfs is None) == (p_astar is None)
        if p_bfs:
            assert len(p_bfs) == len(p_astar)


def test_unreachable_target_skipped():
    obstacles = np.zeros((5, 5), dtype=bool)
    obstacles[1, :] = True  # стена, отрезающая нижние строки
    field = Field(5, 5, weed=np.ones((5, 5)), obstacles=obstacles)
    path = route_two_opt(field, [(0, 3), (4, 4)], START)
    assert (0, 3) in path and (4, 4) not in path


def test_with_field_random_and_targets():
    """Стык с настоящим генератором и Field.targets() участника B."""
    for seed in range(5):
        field = Field.random(30, 30, seed=seed, start=START)
        targets = field.targets(0.5)
        for algo in ALGORITHMS:
            check_path(field, algo(field, targets, START), targets)


def test_non_square_field():
    """Код не должен зависеть от размера (Контракт v1, п. 2)."""
    field = Field.random(12, 40, seed=3, start=START)
    targets = field.targets(0.5)
    for algo in ALGORITHMS:
        check_path(field, algo(field, targets, START), targets)


def test_quality_order():
    """2-opt не хуже NN, и обычно короче змейки."""
    total = {a.__name__: 0 for a in ALGORITHMS}
    for seed in range(5):
        field = make_field(seed)
        targets = make_targets(field, seed)
        lengths = {a.__name__: len(a(field, targets, START)) - 1 for a in ALGORITHMS}
        assert lengths["route_two_opt"] <= lengths["route_nearest_neighbor"]
        for k, v in lengths.items():
            total[k] += v
    print("суммарная длина по 5 полям:", total)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
