"""Базовый маршрут «змейка» (boustrophedon).

Цели обходятся по строкам: чётная строка слева направо, нечётная справа налево.
Между целями путь строится через A*, поэтому препятствия обходятся.
Это baseline, с которым сравниваются остальные алгоритмы.
"""
from routing.common import build_path, clean_targets


def route_snake(field, targets, start):
    targets = clean_targets(field, targets, start)
    order = sorted(targets, key=lambda c: (c[0], c[1] if c[0] % 2 == 0 else -c[1]))
    return build_path(field, start, order)
