from routing.utils import manhattan, order_length, path_length
from routing.snake import snake_order
from routing.nearest import nearest_order


def test_manhattan():
    assert manhattan((0, 0), (3, 4)) == 7
    assert manhattan((2, 2), (2, 2)) == 0


def test_path_length():
    # 3 клетки подряд = 2 шага
    assert path_length([(0, 0), (0, 1), (0, 2)]) == 2


def test_order_length():
    # старт (0,0) -> (0,3) -> (0,5): 3 + 2 = 5 шагов
    assert order_length((0, 0), [(0, 3), (0, 5)]) == 5


def test_snake_keeps_all_targets():
    targets = [(0, 2), (1, 0), (0, 0), (1, 2)]
    order = snake_order(targets)
    assert sorted(order) == sorted(targets)   # ничего не потеряно и не добавлено


def test_nearest_order_simple():
    # цели на одной линии: ближайший сосед идёт по порядку
    targets = [(0, 5), (0, 1), (0, 3)]
    assert nearest_order((0, 0), targets) == [(0, 1), (0, 3), (0, 5)]


def test_nearest_keeps_all_targets():
    targets = [(4, 4), (0, 1), (2, 3), (1, 0)]
    order = nearest_order((0, 0), targets)
    assert sorted(order) == sorted(targets)