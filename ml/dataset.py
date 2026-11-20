import numpy as np
import pandas as pd
from ml.history import generate_history, neighbor_mean

FEATURES = ["weed_now", "neighbors_mean", "is_obstacle", "row", "col"]


def make_features(weed, obstacles):
    """Таблица признаков для всех клеток поля (порядок: построчно)."""
    H, W = weed.shape
    rows, cols = np.indices((H, W))
    return pd.DataFrame({
        "weed_now": weed.ravel(),
        "neighbors_mean": neighbor_mean(weed, obstacles).ravel(),
        "is_obstacle": obstacles.ravel().astype(int),
        "row": rows.ravel(),
        "col": cols.ravel(),
    })


def build_dataset(history):
    parts = []
    for fid, frames, obstacles in history:
        for step in range(len(frames) - 1):
            df = make_features(frames[step], obstacles)
            df.insert(0, "step", step)
            df.insert(0, "field_id", fid)
            df["weed_next"] = frames[step + 1].ravel()
            parts.append(df)
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    data = build_dataset(generate_history())
    data.to_csv("data/weed_data.csv", index=False)
    print(data.shape)