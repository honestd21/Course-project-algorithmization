def manhattan(a, b):
    """Расстояние между клетками (r, c) при движении по сетке: |Δr| + |Δc|."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def order_length(start, order):
    """Длина маршрута: старт -> цели по порядку (без учёта препятствий)."""
    total = 0
    prev = start
    for cell in order:
        total += manhattan(prev, cell)
        prev = cell
    return total


def path_length(path):
    """Длина готового пути (списка клеток) в шагах."""
    return max(len(path) - 1, 0)

