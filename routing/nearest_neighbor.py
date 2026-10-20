"""Жадный маршрут: всегда идти к ближайшей необработанной цели."""
from routing.common import build_path, clean_targets, distance_matrix


def nn_order(dist, n_targets):
    """Порядок обхода целей (индексы 1..n_targets; узел 0 это старт)."""
    unvisited = set(range(1, n_targets + 1))
    order, current = [], 0
    while unvisited:
        nxt = min(unvisited, key=lambda j: (dist[current, j], j))
        order.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return order


def route_nearest_neighbor(field, targets, start):
    targets = clean_targets(field, targets, start)
    if not targets:
        return [start]
    nodes = [start] + targets
    dist = distance_matrix(field, nodes)
    order = nn_order(dist, len(targets))
    return build_path(field, start, [nodes[i] for i in order])
