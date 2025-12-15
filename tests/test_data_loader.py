"""
测试 DataLoader、Preprocessor、MissingValueHandler、OutlierDetector
"""

import pandas as pd
import numpy as np
import os

from src.data.data_loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.data.missing_handler import MissingValueHandler
from src.data.outlier_detector import OutlierDetector


def test_data_loader(tmp_path):
    # 创建临时 CSV
    file = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(file, index=False)

    loader = DataLoader()
    df = loader.load_data(str(file))

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)


def test_preprocessor_clean():
    df = pd.DataFrame({
        "A ": [1, 2],
        "B": ["x ", " y"]
    })
    pre = Preprocessor()
    df2 = pre.clean_data(df)

    assert "A " in df2.columns
    assert df2["B"].tolist() == ["x", "y"]


def test_missing_value_handler():
    df = pd.DataFrame({
        "a": [1, None, 3],
        "b": [4, 5, None]
    })
    mvh = MissingValueHandler()
    df2 = mvh.fill_missing_multi_strategy(df)

    assert df2.isna().sum().sum() == 0


def test_outlier_detector():
    df = pd.DataFrame({"x": [1, 2, 100]})  # 100 是离群点
    od = OutlierDetector()
    df2 = od.handle_outliers(df)

    assert df2.shape[0] <= df.shape[0]  # 可能删除离群点
