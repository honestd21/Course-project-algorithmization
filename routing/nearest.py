from routing.utils import manhattan


def nearest_order(start, targets):
    """Ближайший сосед: всегда едем к ближайшей из оставшихся целей."""
    remaining = list(targets)
    order = []
    current = start
    while remaining:
        nxt = min(remaining, key=lambda t: manhattan(current, t))
        remaining.remove(nxt)
        order.append(nxt)
        current = nxt
    return order