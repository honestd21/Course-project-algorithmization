def snake_order(targets):
    """Порядок «змейкой»: строки сверху вниз, чётные слева направо, нечётные справа налево."""
    return sorted(targets, key=lambda t: (t[0], t[1] if t[0] % 2 == 0 else -t[1]))