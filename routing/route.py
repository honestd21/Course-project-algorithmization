from routing.astar import astar_search
from routing.bfs import bfs_search
from routing.nearest import nearest_order
from routing.snake import snake_order
from routing.two_opt import two_opt

FINDERS = {"bfs": bfs_search, "astar": astar_search}


def make_order(start, targets, order_method):
    if order_method == "snake":
        return snake_order(targets)
    if order_method == "nearest":
        return nearest_order(start, targets)
    if order_method == "two_opt":
        return two_opt(start, nearest_order(start, targets))
    if order_method == "given":
        return list(targets)
    raise ValueError(f"Неизвестный порядок: {order_method}")


def route(field, start, targets, order_method="nearest", finder="astar"):
    """Полный маршрут: порядок целей + реальный путь между ними в обход препятствий.
    Возвращает (path, visited): список клеток и число просмотренных клеток."""
    order = make_order(start, targets, order_method)
    search = FINDERS[finder]
    path = [start]
    visited = 0
    current = start
    for goal in order:
        segment, v = search(field, current, goal)
        visited += v
        if not segment:  # цель недостижима, пропускаем
            continue
        path.extend(segment[1:])
        current = goal
    return path, visited