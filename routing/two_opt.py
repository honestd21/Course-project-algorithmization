"""Улучшение порядка обхода методом 2-opt поверх nearest neighbor.

Путь открытый: старт зафиксирован, конец свободный. Один шаг 2-opt разворачивает
кусок порядка order[i..j], если это сокращает общую длину.
"""
from routing.common import build_path, clean_targets, distance_matrix
from routing.nearest_neighbor import nn_order


def two_opt(dist, order):
    """Улучшает order (индексы узлов, 0 это старт) и возвращает новый список."""
    tour = [0] + list(order)
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                before = dist[tour[i - 1], tour[i]]
                after = dist[tour[i - 1], tour[j]]
                if j + 1 < n:  # есть ребро после сегмента
                    before += dist[tour[j], tour[j + 1]]
                    after += dist[tour[i], tour[j + 1]]
                if after < before:
                    tour[i:j + 1] = reversed(tour[i:j + 1])
                    improved = True
    return tour[1:]


def route_two_opt(field, targets, start):
    targets = clean_targets(field, targets, start)
    if not targets:
        return [start]
    nodes = [start] + targets
    dist = distance_matrix(field, nodes)
    order = two_opt(dist, nn_order(dist, len(targets)))
    return build_path(field, start, [nodes[i] for i in order])
