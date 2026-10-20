"""Маршрутизация (участник A).

route(field, targets, start) -> list[(row, col)] по Контракту v1.
Основной алгоритм: nearest neighbor + 2-opt, пути между целями через A*.
"""
from routing.pathfinding import astar_path, bfs_path
from routing.snake import route_snake
from routing.nearest_neighbor import route_nearest_neighbor
from routing.two_opt import route_two_opt

route = route_two_opt

__all__ = [
    "route", "route_snake", "route_nearest_neighbor", "route_two_opt",
    "astar_path", "bfs_path",
]
