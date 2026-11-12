import heapq

from routing.utils import manhattan
from simulator.field import neighbors


def astar_search(field, start, goal):
    """A*: как BFS, но сначала берём клетки с наименьшей f = g + h.
    g: сколько прошли, h: manhattan до цели. Возвращает (путь, просмотрено)."""
    heap = [(manhattan(start, goal), 0, start)]
    g = {start: 0}
    parent = {start: None}
    closed = set()
    visited = 0
    counter = 1  # чтобы кучу не сравнивала клетки при равных f
    while heap:
        _, _, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        closed.add(cur)
        visited += 1
        if cur == goal:
            break
        for nb in neighbors(field, cur[0], cur[1]):
            new_g = g[cur] + 1
            if nb not in g or new_g < g[nb]:
                g[nb] = new_g
                parent[nb] = cur
                f = new_g + manhattan(nb, goal)
                heapq.heappush(heap, (f, counter, nb))
                counter += 1
    if goal not in parent:
        return [], visited
    path = []
    cell = goal
    while cell is not None:
        path.append(cell)
        cell = parent[cell]
    path.reverse()
    return path, visited


def astar(field, start, goal):
    return astar_search(field, start, goal)[0]