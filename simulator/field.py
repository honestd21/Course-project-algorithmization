import numpy as np


class Field:
    """Поле: weed[r, c] в диапазоне 0..1, obstacles[r, c] == True значит клетка непроходима."""

    def __init__(self, H, W, weed=None, obstacles=None):
        self.H = H
        self.W = W
        self.weed = np.zeros((H, W)) if weed is None else weed
        self.obstacles = np.zeros((H, W), dtype=bool) if obstacles is None else obstacles

    @classmethod
    def random(cls, H, W, n_blobs=5, n_obstacles=15, seed=None, start=(0, 0)):
        rng = np.random.default_rng(seed)
        rows, cols = np.ogrid[:H, :W]

        # очаги сорняков: гауссовы «пятна»
        weed = np.zeros((H, W))
        for _ in range(n_blobs):
            r0, c0 = rng.integers(0, H), rng.integers(0, W)
            sigma = rng.uniform(2, 4)
            strength = rng.uniform(0.6, 1.0)
            weed += strength * np.exp(-((rows - r0) ** 2 + (cols - c0) ** 2) / (2 * sigma ** 2))
        weed = np.clip(weed, 0, 1)

        # препятствия: случайные клетки, но не стартовая
        obstacles = np.zeros((H, W), dtype=bool)
        for _ in range(n_obstacles):
            r, c = rng.integers(0, H), rng.integers(0, W)
            if (r, c) != start:
                obstacles[r, c] = True
        weed[obstacles] = 0  # на препятствии сорняков нет

        return cls(H, W, weed, obstacles)

    def in_bounds(self, r, c):
        return 0 <= r < self.H and 0 <= c < self.W

    def is_free(self, r, c):
        return self.in_bounds(r, c) and not self.obstacles[r, c]

    def targets(self, threshold):
        """Список клеток (r, c), которые нужно опрыскать."""
        rs, cs = np.where((self.weed >= threshold) & ~self.obstacles)
        return list(zip(rs.tolist(), cs.tolist()))


def neighbors(field, r, c):
    """Соседи клетки (вверх, вниз, влево, вправо), в поле и без препятствий."""
    result = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if field.is_free(nr, nc):
            result.append((nr, nc))
    return result



