from dataclasses import dataclass


@dataclass
class RobotParams:
    start: tuple = (0, 0)      # где стоит робот
    threshold: float = 0.3     # клетки с засорением >= порога считаются целями
    tank_fraction: float = 0.5 # бак = эта доля от стоимости обработки ВСЕХ целей
    cost_base: int = 1         # минимальный расход на клетку
    cost_scale: int = 4        # чем сильнее засорение, тем больше расход


def cell_cost(weed_value, params):
    """Расход бака на клетку: целое число, растёт с засорением."""
    return params.cost_base + int(round(weed_value * params.cost_scale))