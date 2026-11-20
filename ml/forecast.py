import joblib
import numpy as np
from ml.dataset import make_features
from ml.train import MODEL_PATH


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def forecast(field, model):
    """Прогноз засорения на следующий шаг. Возвращает новый массив (H, W), поле не меняет."""
    X = make_features(field.weed, field.obstacles)
    pred = model.predict(X).reshape(field.H, field.W)
    pred = np.clip(pred, 0, 1)
    pred[field.obstacles] = 0
    return pred