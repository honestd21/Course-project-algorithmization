import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error

import ml.train as train_module
from ml.clusters import cluster_targets
from ml.dataset import FEATURES, build_dataset, make_features
from ml.forecast import forecast
from ml.history import generate_history, neighbor_mean, step_weed
from simulator.field import Field


def small_history(n_fields=6, steps=3, seed=0):
    return generate_history(n_fields=n_fields, steps=steps, H=12, W=12, seed=seed)


# ---------- neighbor_mean ----------

def test_neighbor_mean_center_and_corner():
    weed = np.zeros((3, 3))
    weed[1, 1] = 1.0
    obstacles = np.zeros((3, 3), dtype=bool)
    nm = neighbor_mean(weed, obstacles)
    assert nm[0, 1] == pytest.approx(1 / 3)   # верхняя граница: 3 соседа
    assert nm[0, 0] == 0.0                     # угол, соседи нулевые
    assert nm[1, 1] == 0.0                     # соседи центра нулевые


def test_neighbor_mean_corner_uses_only_existing_neighbors():
    weed = np.zeros((3, 3))
    weed[0, 1] = 1.0
    obstacles = np.zeros((3, 3), dtype=bool)
    nm = neighbor_mean(weed, obstacles)
    assert nm[0, 0] == pytest.approx(1 / 2)    # у угла только 2 соседа


def test_neighbor_mean_obstacle_counts_as_zero():
    weed = np.full((3, 3), 1.0)
    obstacles = np.zeros((3, 3), dtype=bool)
    obstacles[0, 1] = True
    nm = neighbor_mean(weed, obstacles)
    # у (0, 0) соседи (0, 1) препятствие и (1, 0): (0 + 1) / 2
    assert nm[0, 0] == pytest.approx(0.5)


# ---------- step_weed ----------

def test_step_weed_range_and_obstacles():
    field = Field.random(20, 20, seed=1)
    rng = np.random.default_rng(0)
    new = step_weed(field.weed, field.obstacles, rng)
    assert new.shape == field.weed.shape
    assert new.min() >= 0 and new.max() <= 1
    assert (new[field.obstacles] == 0).all()


def test_step_weed_does_not_change_input():
    field = Field.random(20, 20, seed=1)
    before = field.weed.copy()
    step_weed(field.weed, field.obstacles, np.random.default_rng(0))
    assert (field.weed == before).all()


def test_step_weed_reproducible():
    field = Field.random(20, 20, seed=1)
    a = step_weed(field.weed, field.obstacles, np.random.default_rng(5))
    b = step_weed(field.weed, field.obstacles, np.random.default_rng(5))
    assert (a == b).all()


# ---------- generate_history ----------

def test_history_structure():
    history = small_history(n_fields=4, steps=3)
    assert len(history) == 4
    for fid, frames, obstacles in history:
        assert len(frames) == 4                      # steps + 1 кадр
        assert all(f.shape == (12, 12) for f in frames)
        assert obstacles.shape == (12, 12)


def test_history_reproducible():
    a = small_history(seed=7)
    b = small_history(seed=7)
    assert (a[0][1][-1] == b[0][1][-1]).all()


# ---------- make_features / build_dataset ----------

def test_make_features_shape_and_columns():
    field = Field.random(10, 15, seed=2)
    X = make_features(field.weed, field.obstacles)
    assert X.shape == (10 * 15, len(FEATURES))
    assert list(X.columns) == FEATURES


def test_make_features_row_major_order():
    field = Field.random(10, 15, seed=2)
    X = make_features(field.weed, field.obstacles)
    # строка таблицы i соответствует клетке (i // W, i % W)
    i = 3 * 15 + 7
    assert X.loc[i, "row"] == 3 and X.loc[i, "col"] == 7
    assert X.loc[i, "weed_now"] == field.weed[3, 7]


def test_build_dataset_size_and_columns():
    history = small_history(n_fields=4, steps=3)
    data = build_dataset(history)
    assert len(data) == 4 * 3 * 12 * 12              # поля * переходы * клетки
    for col in ["field_id", "step", "weed_next"] + FEATURES:
        assert col in data.columns


def test_build_dataset_target_is_next_frame():
    history = small_history(n_fields=2, steps=3)
    data = build_dataset(history)
    fid, frames, _ = history[1]
    part = data[(data["field_id"] == fid) & (data["step"] == 2)]
    assert np.allclose(part["weed_next"].to_numpy(), frames[3].ravel())
    assert np.allclose(part["weed_now"].to_numpy(), frames[2].ravel())


# ---------- train / forecast ----------

@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ml")
    csv = tmp / "data.csv"
    build_dataset(generate_history(n_fields=12, steps=4, H=15, W=15, seed=0)).to_csv(csv, index=False)
    old_path = train_module.MODEL_PATH
    train_module.MODEL_PATH = str(tmp / "model.joblib")   # не трогаем настоящую модель
    try:
        model = train_module.train(csv_path=str(csv))
    finally:
        train_module.MODEL_PATH = old_path
    return model, tmp


def test_train_returns_fitted_model(trained):
    model, _ = trained
    assert hasattr(model, "predict")
    assert model.n_features_in_ == len(FEATURES)


def test_train_saves_model_file(trained):
    _, tmp = trained
    assert (tmp / "model.joblib").exists()


def test_forecast_shape_range_obstacles(trained):
    model, _ = trained
    field = Field.random(30, 30, seed=3)
    pred = forecast(field, model)
    assert pred.shape == (30, 30)
    assert pred.min() >= 0 and pred.max() <= 1
    assert (pred[field.obstacles] == 0).all()


def test_forecast_does_not_change_field(trained):
    model, _ = trained
    field = Field.random(30, 30, seed=3)
    weed_before = field.weed.copy()
    obstacles_before = field.obstacles.copy()
    forecast(field, model)
    assert (field.weed == weed_before).all()
    assert (field.obstacles == obstacles_before).all()


def test_forecast_close_to_real_next_step(trained):
    """Прогноз должен быть не хуже «как сейчас» на новых полях (проверка качества)."""
    model, _ = trained
    errs_model, errs_base = [], []
    for seed in range(100, 105):
        field = Field.random(15, 15, seed=seed)
        nxt = step_weed(field.weed, field.obstacles, np.random.default_rng(seed))
        pred = forecast(field, model)
        errs_model.append(mean_absolute_error(nxt.ravel(), pred.ravel()))
        errs_base.append(mean_absolute_error(nxt.ravel(), field.weed.ravel()))
    assert np.mean(errs_model) < 0.1                 # разумная абсолютная ошибка
    assert np.mean(errs_model) <= np.mean(errs_base) * 1.5


# ---------- clusters ----------

def test_cluster_targets_labels_and_centers():
    field = Field.random(30, 30, seed=1)
    cells = field.targets(0.3)
    labels, centers = cluster_targets(cells, field.weed, k=3)
    assert len(labels) == len(cells)
    assert centers.shape == (3, 2)
    assert set(labels) <= {0, 1, 2}


def test_cluster_targets_fewer_points_than_k():
    cells = [(1, 1), (5, 5)]
    labels, centers = cluster_targets(cells, None, k=3)
    assert len(centers) == 2                          # k урезается до числа точек


def test_cluster_targets_separates_two_far_groups():
    cells = [(0, 0), (0, 1), (1, 0), (20, 20), (20, 21), (21, 20)]
    labels, _ = cluster_targets(cells, None, k=2)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]
