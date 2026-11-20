import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from ml.dataset import FEATURES

MODEL_PATH = "ml/model.joblib"


def train(csv_path="data/weed_data.csv", seed=0):
    data = pd.read_csv(csv_path)
    # делим по полям, а не по строкам, иначе будет утечка (одно поле в train и test)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, te = next(gss.split(data, groups=data["field_id"]))
    X_tr, y_tr = data.iloc[tr][FEATURES], data.iloc[tr]["weed_next"]
    X_te, y_te = data.iloc[te][FEATURES], data.iloc[te]["weed_next"]

    model = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=seed)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    print("MAE модели:      ", mean_absolute_error(y_te, pred))
    print("MAE «как сейчас»:", mean_absolute_error(y_te, X_te["weed_now"]))  # базовая линия
    print("R2 модели:       ", r2_score(y_te, pred))

    joblib.dump(model, MODEL_PATH)
    return model


if __name__ == "__main__":
    train()