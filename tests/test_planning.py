import pytest

from planning.dp import knapsack_dp
from planning.evaluate import evaluate
from planning.greedy import greedy_ratio, greedy_route
from planning.params import RobotParams, cell_cost
from planning.plan import find_targets, plan
from simulator.field import Field


def make_field(seed=1):
    return Field.random(30, 30, seed=seed)


def total_cost(cells, weed, params):
    return sum(cell_cost(weed[r, c], params) for r, c in cells)


# ---------- cell_cost ----------

def test_cell_cost_grows_with_weed():
    p = RobotParams()
    assert cell_cost(0.0, p) == 1
    assert cell_cost(1.0, p) == 5
    assert cell_cost(0.3, p) <= cell_cost(0.9, p)


# ---------- knapsack_dp ----------

def test_dp_known_example():
    # берём 0 и 2: расход 2+2=4, польза 3+4=7; клетка 1 дороже и хуже
    assert knapsack_dp([3, 5, 4], [2, 3, 2], 4) == [0, 2]


def test_dp_zero_capacity():
    assert knapsack_dp([3, 5], [1, 2], 0) == []


def test_dp_empty_input():
    assert knapsack_dp([], [], 10) == []


def test_dp_nothing_fits():
    assert knapsack_dp([5, 6], [4, 5], 3) == []


def test_dp_takes_all_if_capacity_large():
    assert knapsack_dp([1, 2, 3], [1, 1, 1], 10) == [0, 1, 2]


def test_dp_beats_greedy_trap():
    # жадный по «польза/расход» берёт клетку 0 (1.5) и больше ничего не вмещает;
    # оптимум: клетки 1 и 2, польза 10
    values, costs, cap = [3, 5, 5], [2, 3, 3], 6
    idx = knapsack_dp(values, costs, cap)
    assert sum(values[i] for i in idx) == 10
    assert sum(costs[i] for i in idx) <= cap


# ---------- plan ----------

@pytest.mark.parametrize("method", ["dp", "greedy_ratio", "greedy_route"])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_plan_respects_tank(method, seed):
    field, params = make_field(seed), RobotParams()
    chosen, cap = plan(field, params, method)
    assert total_cost(chosen, field.weed, params) <= cap


@pytest.mark.parametrize("method", ["dp", "greedy_ratio", "greedy_route"])
def test_plan_chooses_only_targets(method):
    field, params = make_field(), RobotParams()
    chosen, _ = plan(field, params, method)
    assert set(chosen) <= set(field.targets(params.threshold))
    assert len(chosen) == len(set(chosen))  # без повторов


def test_plan_unknown_method():
    with pytest.raises(ValueError):
        plan(make_field(), RobotParams(), "magic")


def test_plan_dp_not_worse_than_greedy_ratio():
    params = RobotParams()
    for seed in range(1, 8):
        field = make_field(seed)
        dp_cells, _ = plan(field, params, "dp")
        gr_cells, _ = plan(field, params, "greedy_ratio")
        dp_val = sum(field.weed[r, c] for r, c in dp_cells)
        gr_val = sum(field.weed[r, c] for r, c in gr_cells)
        assert dp_val >= gr_val - 1e-9


def test_plan_uses_given_weed_map():
    field, params = make_field(), RobotParams()
    fake = field.weed.copy()
    fake[:] = 0
    fake[5, 5] = 1.0  # в «прогнозе» одна цель
    chosen, cap = plan(field, params, "dp", weed=fake)
    # бак считается по карте прогноза: 50% от расхода одной цели
    assert cap == int(cell_cost(1.0, params) * params.tank_fraction)
    assert set(chosen) <= {(5, 5)}


def test_plan_does_not_change_field():
    field, params = make_field(), RobotParams()
    before = field.weed.copy()
    plan(field, params, "dp")
    assert (field.weed == before).all()


def test_find_targets_matches_field_targets():
    field = make_field()
    assert set(find_targets(field, field.weed, 0.3)) == set(field.targets(0.3))


# ---------- greedy ----------

def test_greedy_route_visits_all_with_big_tank():
    field, params = make_field(), RobotParams()
    cells = field.targets(params.threshold)
    order = greedy_route(cells, field.weed, params, 10 ** 6, params.start)
    assert sorted(order) == sorted(cells)  # все цели, каждая один раз


def test_greedy_ratio_zero_capacity():
    field, params = make_field(), RobotParams()
    cells = field.targets(params.threshold)
    assert greedy_ratio(cells, field.weed, params, 0) == []


# ---------- evaluate ----------

def test_evaluate_all_targets_gives_full_coverage():
    field, params = make_field(), RobotParams()
    chosen = field.targets(params.threshold)
    m = evaluate(field, params, chosen)
    assert m["coverage"] == pytest.approx(1.0)
    assert m["wasted"] == 0
    assert m["missed"] == 0


def test_evaluate_empty_plan():
    field, params = make_field(), RobotParams()
    m = evaluate(field, params, [])
    assert m["coverage"] == 0.0
    assert m["collected"] == 0
    assert m["steps"] == 0
    assert m["missed"] == len(field.targets(params.threshold))


def test_evaluate_counts_wasted_cell():
    field, params = make_field(), RobotParams()
    chosen = field.targets(params.threshold)
    empty = next((r, c) for r in range(field.H) for c in range(field.W)
                 if field.weed[r, c] < params.threshold and not field.obstacles[r, c])
    m = evaluate(field, params, chosen + [empty])
    assert m["wasted"] == 1


def test_evaluate_given_order_works_for_greedy_route():
    field, params = make_field(), RobotParams()
    chosen, _ = plan(field, params, "greedy_route")
    m = evaluate(field, params, chosen, order_method="given")
    assert m["steps"] > 0
    assert 0 <= m["coverage"] <= 1
