import pandas as pd
import pytest

from src.features.feature_engineer import FeatureEngineer


def test_fit_transform_basic():
    df = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "income": [2000, 3000, 4000, 5000],
        "gender": ["M", "F", "F", "M"],
    })

    eng = FeatureEngineer(
        numeric_features=["age", "income"],
        categorical_features=["gender"],
        scaling="standard",
        add_polynomial=False,
    )

    out = eng.fit_transform(df)

    # should return a DataFrame-like object
    assert hasattr(out, "shape")
    # rows should match input
    assert out.shape[0] == df.shape[0]


def test_pipeline_not_fitted_error():
    eng = FeatureEngineer(numeric_features=["age"], categorical_features=None)
    with pytest.raises(RuntimeError):
        eng.transform(pd.DataFrame({"age": [1, 2, 3]}))


def test_invalid_scaling_raises():
    df = pd.DataFrame({"age": [1, 2, 3], "gender": ["a", "b", "a"]})
    eng = FeatureEngineer(numeric_features=["age"], categorical_features=["gender"], scaling="bad")
    with pytest.raises(ValueError):
        eng.fit(df)
