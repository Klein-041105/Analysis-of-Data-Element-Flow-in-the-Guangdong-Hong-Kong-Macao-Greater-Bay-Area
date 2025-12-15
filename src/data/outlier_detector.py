from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def grubbs_test(series: pd.Series, alpha: float = 0.05) -> Dict[str, object]:
    """使用Grubbs检验检测单变量离群值。
    
    Grubbs检验是单变量参数检验，适用于正态分布数据。
    每次检验只标记最极端的一个点。
    
    Args:
        series: 输入Series
        alpha: 显著性水平（默认0.05）
        
    Returns:
        字典，包含：
        - 'is_outlier': 离群值标记Series (与输入索引对齐)
        - 'g_stat': Grubbs检验统计量
        - 'crit': 临界值
    """
    x = series.dropna()
    n = x.size
    if n < 3:
        return {"is_outlier": pd.Series(False, index=series.index), "g_stat": None, "p_value": None}
    mean_x = x.mean()
    std_x = x.std(ddof=1)
    # 计算最大偏差观测
    abs_dev = (x - mean_x).abs()
    max_idx = abs_dev.idxmax()
    g_stat = abs_dev.loc[max_idx] / std_x
    # 临界值
    t = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    numerator = (n - 1) * math.sqrt(t ** 2)
    denominator = math.sqrt(n) * math.sqrt(n - 2 + t ** 2)
    crit = numerator / denominator
    is_outlier = pd.Series(False, index=series.index)
    if g_stat > crit:
        is_outlier.loc[max_idx] = True
    return {"is_outlier": is_outlier, "g_stat": float(g_stat), "crit": float(crit)}

def isolation_forest_detect(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    contamination: float = 0.05,
    random_state: int = 42,
) -> Dict[str, object]:
    """使用隔离森林进行多变量离群值检测。
    
    隔离森林是无参数算法，通过随机分割空间隔离异常点。
    不假设数据分布，适用于高维数据。
    
    Args:
        df: 输入DataFrame
        numeric_cols: 要检测的数值列，若为None则自动选择
        contamination: 期望的异常比例（默认0.05）
        random_state: 随机种子
        
    Returns:
        字典，包含：
        - 'flags': 离群值标记Series (True为离群值)
        - 'model': 训练好的IsolationForest模型
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols].fillna(0).values
    logger.info("IsolationForest on cols: %s", numeric_cols)
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    preds = clf.fit_predict(X)  # -1 outlier, 1 inlier
    flags = pd.Series(preds == -1, index=df.index, name="iforest_outlier")
    return {"flags": flags, "model": clf}

def lof_detect(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    n_neighbors: int = 20,
    contamination: float = 0.05,
) -> Dict[str, object]:
    """使用局部离群因子（LOF）进行多变量离群值检测。
    
    LOF评估每个点的局部密度，相对于其邻域的异常程度。
    对局部离群值敏感。
    
    Args:
        df: 输入DataFrame
        numeric_cols: 要检测的数值列，若为None则自动选择
        n_neighbors: 邻域大小（默认20）
        contamination: 期望的异常比例（默认0.05）
        
    Returns:
        字典，包含：
        - 'flags': 离群值标记Series (True为离群值)
        - 'model': 训练好的LocalOutlierFactor模型
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols].fillna(0).values
    logger.info("LOF on cols: %s", numeric_cols)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False, contamination=contamination)
    preds = lof.fit_predict(X)  # -1 outlier, 1 inlier
    flags = pd.Series(preds == -1, index=df.index, name="lof_outlier")
    return {"flags": flags, "model": lof}

def timeseries_residual_detect(
    series: pd.Series,
    method: str = "stl",
    z_thresh: float = 3.0,
) -> Dict[str, object]:
    """使用时间序列分解残差进行离群值检测。
    
    通过STL或ARIMA分解提取残差，离群值表现为异常残差。
    需要statsmodels库。
    
    Args:
        series: 时间序列Series
        method: 分解方法，'stl'（季节分解）或'arima'（AR模型）
        z_thresh: Z分数阈值（默认3.0，标准差倍数）
        
    Returns:
        字典，包含：
        - 'flags': 离群值标记Series
        - 'residuals': 提取的残差Series
        - 'zscores': 残差的Z分数
        
    Raises:
        ImportError: 当statsmodels不可用时
    """
    try:
        import statsmodels.api as sm  # type: ignore
    except Exception as e:
        logger.warning("statsmodels not available: %s", e)
        raise

    s = series.dropna()
    if s.empty:
        return {"flags": pd.Series(False, index=series.index)}

    if method == "stl":
        stl = sm.tsa.STL(s, robust=True)
        res = stl.fit()
        resid = res.resid
    else:
        # 简单 ARIMA 残差：使用 AR(1) as fallback
        arima = sm.tsa.ARIMA(s, order=(1, 0, 0)).fit()
        resid = arima.resid

    zscores = (resid - resid.mean()) / resid.std(ddof=1)
    out_idx = zscores.abs() > z_thresh
    flags = pd.Series(False, index=series.index)
    flags.loc[resid.index[out_idx]] = True
    return {"flags": flags, "residuals": resid, "zscores": zscores}

def aggregate_outlier_report(
    df: pd.DataFrame, detectors_results: List[Dict[str, object]]
) -> Dict[str, object]:
    """聚合多个离群值检测器的结果。
    
    结合不同检测方法的标记，通过投票（多数原则）生成最终离群值标签。
    
    Args:
        df: 原始DataFrame（用于获取索引）
        detectors_results: 检测器结果列表，每个为包含'flags'的字典
        
    Returns:
        字典，包含：
        - 'report': 统计摘要
            - 'method_count': 使用的检测方法数
            - 'detected_count': 标记为离群的样本数
            - 'threshold': 投票阈值
            - 'by_index': 离群值的详细标记（按索引）
        - 'flags_df': 完整的标记DataFrame
    """
    df_flags = pd.DataFrame(index=df.index)
    for res in detectors_results:
        if "flags" in res and isinstance(res["flags"], pd.Series):
            col = res["flags"].name or f"flag_{len(df_flags.columns)}"
            df_flags[col] = res["flags"].astype(int)
    if df_flags.shape[1] == 0:
        return {"report": {}, "flags_df": df_flags}

    df_flags["outlier_score"] = df_flags.sum(axis=1)
    # 设定阈值为多数方法判定为异常
    threshold = max(1, int(math.ceil(df_flags.shape[1] / 2)))
    df_flags["outlier_flag"] = df_flags["outlier_score"] >= threshold
    report = {
        "method_count": df_flags.shape[1] - 2 if "outlier_score" in df_flags.columns else df_flags.shape[1],
        "detected_count": int(df_flags["outlier_flag"].sum()),
        "threshold": threshold,
        "by_index": df_flags[df_flags["outlier_flag"]].to_dict(orient="index"),
    }
    return {"report": report, "flags_df": df_flags}
