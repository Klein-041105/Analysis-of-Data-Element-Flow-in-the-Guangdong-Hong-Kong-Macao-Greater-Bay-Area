"""
测试特征工程与特征选择模块
"""

import numpy as np
import pandas as pd

from src.features.feature_engineer import FeatureEngineer
from src.features.feature_selector import FeatureSelector


def test_feature_engineering():
    df = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [10, 20, 30]
    })

    fe = FeatureEngineer()
    df2 = fe.generate_interactions(df)

    assert "x_y" in df2.columns


def test_feature_selector_variance():
    X = np.array([[1, 0], [1, 0], [1, 0]])
    fs = FeatureSelector()

    selected = fs.remove_low_variance(X, threshold=0.01)
    assert selected.shape[1] == 0  # 所有特征方差为 0


def test_feature_selector_correlated():
    X = np.array([[1, 2], [2, 4], [3, 6]])  # 完全相关
    fs = FeatureSelector()

    selected = fs.remove_correlated_features(X, threshold=0.9)
    assert selected.shape[1] == 1
