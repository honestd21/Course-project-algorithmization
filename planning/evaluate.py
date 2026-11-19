from routing.route import route
from routing.utils import path_length


def evaluate(field, params, chosen, order_method="nearest", finder="astar"):
    """Оценка плана по НАСТОЯЩЕМУ полю (field.weed), даже если планировали по прогнозу."""
    real = field.weed
    thr = params.threshold
    real_targets = {(r, c) for r in range(field.H) for c in range(field.W)
                    if real[r, c] >= thr and not field.obstacles[r, c]}
    chosen_set = set(chosen)

    total = sum(real[r, c] for r, c in real_targets)

    
    # остановился тут
    collected = sum(real[r, c] for r, c in chosen_set & real_targets)
    wasted = len(chosen_set - real_targets)      # опрыскали, а сорняков не было
    missed = len(real_targets - chosen_set)      # сорняки есть, а не опрыскали

    path, visited = route(field, params.start, chosen, order_method, finder)
    return {
        "coverage": collected / total if total > 0 else 0.0,  # доля собранной пользы
        "collected": collected,
        "wasted": wasted,
        "missed": missed,
        "steps": path_length(path),
        "visited": visited,
    }