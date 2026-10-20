"""Общие помощники для всех алгоритмов маршрутизации."""
import numpy as np

from routing.pathfinding import astar_path, bfs_distances


def clean_targets(field, targets, start):
    """Оставляет только осмысленные цели.

    Убирает дубликаты, саму стартовую клетку, клетки вне поля, препятствия
    и недостижимые от start клетки. Порядок первого появления сохраняется.
    """
    reachable = bfs_distances(field, start)
    result, seen = [], {start}
    for cell in targets:
        cell = (int(cell[0]), int(cell[1]))
        if cell in seen:
            continue
        seen.add(cell)
        if cell in reachable:
            result.append(cell)
    return result


def distance_matrix(field, nodes):
    """Матрица кратчайших расстояний между узлами (BFS из каждого узла).

    nodes[0] обычно старт. Недостижимых пар нет, если узлы прошли clean_targets.
    """
    n = len(nodes)
    index = {cell: i for i, cell in enumerate(nodes)}
    matrix = np.zeros((n, n), dtype=np.int64)
    for i, cell in enumerate(nodes):
        dist = bfs_distances(field, cell)
        for other, j in index.items():
            matrix[i, j] = dist[other]
    return matrix


def build_path(field, start, order):
    """Склеивает полный путь: start -> order[0] -> order[1] -> ...

    Между соседними целями идёт кратчайший путь A*, обходящий препятствия.
    """
    path = [start]
    current = start
    for cell in order:
        segment = astar_path(field, current, cell)
        path.extend(segment[1:])  # первая клетка уже в пути
        current = cell
    return path
