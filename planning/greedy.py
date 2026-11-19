from planning.params import cell_cost
from routing.utils import manhattan


def greedy_ratio(cells, weed, params, capacity):
    """Жадный выбор: берём клетки по убыванию «польза / расход», пока влезает в бак."""
    scored = []
    for (r, c) in cells:
        value = weed[r, c]
        cost = cell_cost(value, params)
        scored.append((value / cost, cost, (r, c)))
    scored.sort(reverse=True)
    chosen, used = [], 0
    for _, cost, cell in scored:
        if used + cost <= capacity:
            chosen.append(cell)
            used += cost
    return chosen


def greedy_route(cells, weed, params, capacity, start):
    """Жадный по маршруту: едем к ближайшей цели, на которую ещё хватает бака.
    Сразу возвращает цели в порядке объезда."""
    remaining = list(cells)
    order, used, current = [], 0, start
    while remaining:
        fits = [t for t in remaining
                if used + cell_cost(weed[t[0], t[1]], params) <= capacity]
        if not fits:
            break
        nxt = min(fits, key=lambda t: manhattan(current, t))
        remaining.remove(nxt)
        order.append(nxt)
        used += cell_cost(weed[nxt[0], nxt[1]], params)
        current = nxt
    return order