from collections import deque

from simulator.field import neighbors


def bfs_search(field, start, goal):
    """Кратчайший путь start -> goal. Возвращает (путь, число просмотренных клеток).
    Если пути нет, путь пустой []."""
    queue = deque([start])
    parent = {start: None}
    visited = 0
    while queue:
        cur = queue.popleft()
        visited += 1
        if cur == goal:
            break
        for nb in neighbors(field, cur[0], cur[1]):
            if nb not in parent:
                parent[nb] = cur
                queue.append(nb)
    if goal not in parent:
        return [], visited
    path = []
    cell = goal
    while cell is not None:
        path.append(cell)
        cell = parent[cell]
    path.reverse()
    return path, visited


def bfs(field, start, goal):
    return bfs_search(field, start, goal)[0]