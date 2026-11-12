from routing.utils import order_length


def two_opt(start, order):
    """2-opt: пробуем развернуть кусок маршрута; если стало короче, оставляем."""
    best = list(order)
    best_len = order_length(start, best)
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                candidate = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                cand_len = order_length(start, candidate)
                if cand_len < best_len:
                    best, best_len = candidate, cand_len
                    improved = True
    return best