import pandas as pd
import numpy as np
import pytest

from src.models.ml_predictor import MLPredictor


def make_regression_data(n=50, n_features=3, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, n_features), columns=[f"f{i}" for i in range(n_features)])
    coef = rng.randn(n_features)
    y = X.values.dot(coef) + rng.randn(n) * 0.1
    return X, pd.Series(y)


def test_fit_predict_default_model():
    X, y = make_regression_data()
    pred = MLPredictor()
    pred.fit(X, y)
    preds = pred.predict(X)

    assert isinstance(preds, pd.Series)
    assert preds.shape[0] == X.shape[0]


def test_fit_predict_helper():
    X, y = make_regression_data()
    pred = MLPredictor(model_name="gbr")
    out = pred.fit_predict(X, y)
    assert isinstance(out, pd.Series)
    assert out.shape[0] == X.shape[0]


def test_predict_before_fit_raises():
    X, _ = make_regression_data()
    pred = MLPredictor()
    with pytest.raises(RuntimeError):
        pred.predict(X)


def test_unsupported_model_name_raises():
    with pytest.raises(ValueError):
        MLPredictor(model_name="not_a_model")._create_model()
