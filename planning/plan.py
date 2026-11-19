import numpy as np

from planning.dp import knapsack_dp
from planning.greedy import greedy_ratio, greedy_route
from planning.params import cell_cost


def find_targets(field, weed, threshold):
    rs, cs = np.where((weed >= threshold) & ~field.obstacles)
    return list(zip(rs.tolist(), cs.tolist()))


def plan(field, params, method="dp", weed=None):
    """Выбор клеток под бак. weed=None: планируем по настоящему полю,
    иначе по переданной карте (например, по прогнозу ML).
    Возвращает (список клеток, ёмкость бака)."""
    if weed is None:
        weed = field.weed
    cells = find_targets(field, weed, params.threshold)
    costs = [cell_cost(weed[r, c], params) for r, c in cells]
    capacity = int(sum(costs) * params.tank_fraction)

    if method == "greedy_ratio":
        chosen = greedy_ratio(cells, weed, params, capacity)
    elif method == "greedy_route":
        chosen = greedy_route(cells, weed, params, capacity, params.start)
    elif method == "dp":
        values = [weed[r, c] for r, c in cells]
        idx = knapsack_dp(values, costs, capacity)
        chosen = [cells[i] for i in idx]
    else:
        raise ValueError(f"Неизвестный метод: {method}")
    return chosen, capacity