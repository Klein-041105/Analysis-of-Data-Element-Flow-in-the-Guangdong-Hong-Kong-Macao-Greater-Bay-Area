from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class ImputationResult:
    """缺失值填补结果容器。
    
    Attributes:
        df_filled: 填补后的DataFrame
        info: 包含填补方法参数的字典，用于追踪和重现结果
    """
    df_filled: pd.DataFrame
    info: dict


def mice_impute(
    df: pd.DataFrame,
    numeric_cols: Optional[Sequence[str]] = None,
    n_iter: int = 10,
    random_state: int = 42,
) -> ImputationResult:
    """使用多元插补链式方程（MICE）进行缺失值填补。
    
    MICE通过迭代方式，使用其他特征逐步预测缺失值。
    
    Args:
        df: 包含缺失值的数据框
        numeric_cols: 要填补的数值列，若为None则自动选择
        n_iter: 迭代次数（默认10）
        random_state: 随机种子，用于重现性
        
    Returns:
        ImputationResult对象，包含填补后的数据和方法参数
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    logger.info("MICE impute on columns: %s", numeric_cols)
    imp = IterativeImputer(max_iter=n_iter, random_state=random_state)
    arr = imp.fit_transform(df[numeric_cols])
    df_filled = df.copy()
    df_filled.loc[:, numeric_cols] = arr
    info = {"method": "mice", "n_iter": n_iter}
    return ImputationResult(df_filled, info)


def knn_impute(
    df: pd.DataFrame,
    numeric_cols: Optional[Sequence[str]] = None,
    n_neighbors: int = 5,
    weights: str = "uniform",
) -> ImputationResult:
    """使用K最近邻进行缺失值填补。
    
    对每个缺失值，使用其K个最近邻样本的对应特征值进行填补。
    
    Args:
        df: 包含缺失值的数据框
        numeric_cols: 要填补的数值列，若为None则自动选择
        n_neighbors: 最近邻个数（默认5）
        weights: 权重方式，'uniform'（等权重）或'distance'（距离反向权重）
        
    Returns:
        ImputationResult对象，包含填补后的数据和方法参数
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    logger.info("KNN impute on columns: %s", numeric_cols)
    imp = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    arr = imp.fit_transform(df[numeric_cols])
    df_filled = df.copy()
    df_filled.loc[:, numeric_cols] = arr
    info = {"method": "knn", "n_neighbors": n_neighbors, "weights": weights}
    return ImputationResult(df_filled, info)


def timeseries_impute(
    df: pd.DataFrame,
    time_col: Optional[str] = None,
    method: str = "spline",
    numeric_cols: Optional[Sequence[str]] = None,
    order: int = 3,
) -> ImputationResult:
    """使用时间序列插值进行缺失值填补。
    
    适用于具有时间维度的数据。支持样条插值和其他pandas插值方法。
    
    Args:
        df: 包含缺失值的数据框
        time_col: 时间列名称。若为None，则假设index为时间索引
        method: 插值方法，'spline'（样条）或pandas支持的其他方法
        numeric_cols: 要填补的数值列，若为None则自动选择
        order: 样条插值的阶数（当method='spline'时使用）
        
    Returns:
        ImputationResult对象，包含填补后的数据和方法参数
    """
    df_filled = df.copy()
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if time_col is not None:
        df_filled = df_filled.set_index(pd.to_datetime(df_filled[time_col]))
    else:
        if not isinstance(df_filled.index, pd.DatetimeIndex):
            logger.warning("No time_col and index is not DatetimeIndex — using default interpolate axis")
    try:
        for c in numeric_cols:
            if method == "spline":
                df_filled[c] = df_filled[c].interpolate(method="spline", order=order)
            else:
                df_filled[c] = df_filled[c].interpolate(method=method)
    finally:
        if time_col is not None:
            df_filled = df_filled.reset_index(drop=False)
    info = {"method": "timeseries", "strategy": method}
    return ImputationResult(df_filled, info)


def autoencoder_impute(
    df: pd.DataFrame,
    numeric_cols: Optional[Sequence[str]] = None,
    latent_dim: int = 8,
    epochs: int = 50,
    batch_size: int = 32,
    random_state: int = 42,
) -> ImputationResult:
    """使用自编码器神经网络进行缺失值填补。
    
    自编码器学习数据的潜在表示，通过重建来估计缺失值。
    如果TensorFlow不可用，自动回退到均值填补。
    
    Args:
        df: 包含缺失值的数据框
        numeric_cols: 要填补的数值列，若为None则自动选择
        latent_dim: 自编码器潜在层维度（默认8）
        epochs: 训练轮数（默认50）
        batch_size: 批大小（默认32）
        random_state: 随机种子
        
    Returns:
        ImputationResult对象。若TensorFlow不可用，使用均值填补并标注为'mean_fallback'
    """
    try:
        import tensorflow as tf  # type: ignore
        from tensorflow import keras  # type: ignore
    except Exception:
        warnings.warn(
            "TensorFlow not available — autoencoder_impute will fallback to simple mean imputation",
            UserWarning,
        )
        # fallback: mean impute
        df_filled = df.copy()
        numeric_cols = numeric_cols or df.select_dtypes(include=[np.number]).columns.tolist()
        for c in numeric_cols:
            df_filled[c] = df_filled[c].fillna(df_filled[c].mean())
        return ImputationResult(df_filled, {"method": "mean_fallback"})

    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    logger.info("Autoencoder impute on cols: %s", numeric_cols)
    X = df[numeric_cols].copy()
    mask = X.isnull().astype(float).values

    # initial fill
    X0 = X.fillna(X.mean()).values.astype(float)

    tf.random.set_seed(random_state)
    input_dim = X0.shape[1]

    # 简单 Autoencoder
    inputs = keras.layers.Input(shape=(input_dim,))
    encoded = keras.layers.Dense(max(4, latent_dim), activation="relu")(inputs)
    encoded = keras.layers.Dense(latent_dim, activation="relu")(encoded)
    decoded = keras.layers.Dense(max(4, latent_dim), activation="relu")(encoded)
    outputs = keras.layers.Dense(input_dim, activation=None)(decoded)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")

    # 自定义训练：通过样本权重屏蔽缺失项对 loss 的影响
    sample_weight = 1.0 - mask  # missing -> 1? we want to weight observed entries more
    # Fit
    model.fit(X0, X0, sample_weight=sample_weight, epochs=epochs, batch_size=batch_size, verbose=0)

    X_rec = model.predict(X0)
    X_final = X0.copy()
    # Replace only missing positions with reconstruction
    X_final[mask.astype(bool)] = X_rec[mask.astype(bool)]

    df_filled = df.copy()
    df_filled.loc[:, numeric_cols] = X_final
    info = {"method": "autoencoder", "latent_dim": latent_dim, "epochs": epochs}
    return ImputationResult(df_filled, info)


def evaluate_imputation(
    truth: pd.DataFrame, imputed: pd.DataFrame, numeric_cols: Optional[Sequence[str]] = None
) -> dict:
    """评估缺失值填补的效果。
    
    计算真值与填补值之间的误差指标（RMSE、MAE）。
    只在两个数据框都非缺失的位置进行评估。
    
    Args:
        truth: 真实值数据框
        imputed: 填补后的数据框
        numeric_cols: 要评估的数值列，若为None则自动选择
        
    Returns:
        字典，结构为：
        {
            'column_name': {'rmse': value, 'mae': value},
            ...,
            '_aggregate': {'rmse_mean': avg_rmse, 'mae_mean': avg_mae}
        }
    """
    if numeric_cols is None:
        numeric_cols = truth.select_dtypes(include=[np.number]).columns.tolist()
    metrics = {}
    for c in numeric_cols:
        # only evaluate positions where truth is not NaN and imputed is not NaN
        mask = truth[c].notna() & imputed[c].notna()
        if mask.sum() == 0:
            metrics[c] = {"rmse": None, "mae": None}
            continue
        y_true = truth.loc[mask, c].values
        y_pred = imputed.loc[mask, c].values
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        metrics[c] = {"rmse": float(rmse), "mae": float(mae)}
    # aggregate
    valid = [v for v in metrics.values() if v["rmse"] is not None]
    if valid:
        metrics["_aggregate"] = {
            "rmse_mean": float(np.mean([v["rmse"] for v in valid])),
            "mae_mean": float(np.mean([v["mae"] for v in valid])),
        }
    else:
        metrics["_aggregate"] = {"rmse_mean": None, "mae_mean": None}
    return metrics
