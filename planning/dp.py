def knapsack_dp(values, costs, capacity):
    """Задача о рюкзаке, динамическое программирование. Возвращает индексы выбранных
    предметов с максимальной суммой values при сумме costs <= capacity.
    Сложность O(n * capacity)."""
    n = len(values)
    # best[i][w] = максимум пользы из первых i предметов при баке w
    best = [[0.0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        v, c = values[i - 1], costs[i - 1]
        # засорение и расход 1 клетки берем
        for w in range(capacity + 1):
            best[i][w] = best[i - 1][w]            # не берём предмет i
            if c <= w:                              # берём, если влезает
                take = best[i - 1][w - c] + v
                if take > best[i][w]:
                    best[i][w] = take

# короче мы проходимс по целям
# и смотрим на бак от 1 до значение бака
# и просто смотрим сколько целей залезет для конкретного значения бака
# Почти, но поправлю одну вещь: мы смотрим не «сколько целей залезет», а какая максимальная польза получится
# восстанавливаем, что именно взяли

    chosen, w = [], capacity
    for i in range(n, 0, -1):
        if best[i][w] != best[i - 1][w]:
            chosen.append(i - 1)
            w -= costs[i - 1]
    chosen.reverse()
    return chosen
# смотрим какие ьрали цели по итогу при максимальной пользе 