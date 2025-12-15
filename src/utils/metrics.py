"""模型评估指标模块。

提供聚类、回归和分类模型的性能评估函数。
所有函数返回字典格式的指标，便于结果记录和对比。

支持的评估类型：
- 聚类: Silhouette系数、Davies-Bouldin指数
- 回归: MSE、RMSE、R²
- 分类: 准确率、加权F1

使用示例：
    >>> from metrics import regression_metrics
    >>> y_true = np.array([1, 2, 3])
    >>> y_pred = np.array([1.1, 2.1, 2.9])
    >>> metrics = regression_metrics(y_true, y_pred)
    >>> print(metrics['r2'])  # 0.999...
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score
)


def clustering_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """计算聚类模型的评估指标。
    
    计算Silhouette系数和Davies-Bouldin指数来评估聚类质量。
    
    Args:
        X: 特征矩阵，形状为 (n_samples, n_features)
        labels: 聚类标签，形状为 (n_samples,)，值为 [0, 1, ..., k-1]
        
    Returns:
        包含以下键的字典：
        - 'silhouette': Silhouette系数 (范围 [-1, 1])
            > 0.5: 聚类结构强
            0.3-0.5: 聚类结构合理
            < 0.3: 聚类结构弱
        - 'davies_bouldin': Davies-Bouldin指数 (越小越好)
            < 1.0: 聚类效果好
            > 2.0: 聚类效果差
        - 若只有一个聚类，两个指标为None
        
    Examples:
        >>> from sklearn.datasets import make_blobs
        >>> X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
        >>> from sklearn.cluster import KMeans
        >>> labels = KMeans(3).fit_predict(X)
        >>> metrics = clustering_metrics(X, labels)
        >>> print(f"Silhouette: {metrics['silhouette']:.3f}")
    """
    results = {}

    # 检查标签是否有效
    if len(np.unique(labels)) > 1:
        results["silhouette"] = float(silhouette_score(X, labels))
        results["davies_bouldin"] = float(davies_bouldin_score(X, labels))
    else:
        results["silhouette"] = None
        results["davies_bouldin"] = None

    return results


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """计算回归模型的评估指标。
    
    计算三个关键指标来评估回归模型性能。
    
    Args:
        y_true: 真实值，形状为 (n_samples,)
        y_pred: 预测值，形状为 (n_samples,)
        
    Returns:
        包含以下键的字典：
        - 'mse': 均方误差，绝对值越小越好
        - 'rmse': 均方根误差，与y的单位一致，易于解释
        - 'r2': R²系数 (范围 [0, 1]，或负值如果模型很差)
            1.0: 完美拟合
            0.7-1.0: 很好
            0.5-0.7: 可接受
            < 0.5: 较差
        
    Examples:
        >>> y_true = np.array([1, 2, 3, 4, 5])
        >>> y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        >>> metrics = regression_metrics(y_true, y_pred)
        >>> print(f"R²: {metrics['r2']:.4f}")  # R²: 0.9952
    """
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """计算分类模型的评估指标。
    
    计算两个关键指标来评估分类模型性能。
    
    Args:
        y_true: 真实标签，形状为 (n_samples,)
        y_pred: 预测标签，形状为 (n_samples,)
        
    Returns:
        包含以下键的字典：
        - 'accuracy': 准确率 (范围 [0, 1])
            1.0: 完全正确
            0.8-1.0: 很好
            0.6-0.8: 可接受
            < 0.6: 较差
        - 'f1': 加权F1分数 (范围 [0, 1]，考虑类别不平衡)
            1.0: 完美分类
            0.7-1.0: 很好
            0.5-0.7: 可接受
            < 0.5: 较差
        
    Examples:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 0, 0, 1])
        >>> metrics = classification_metrics(y_true, y_pred)
        >>> print(f"Accuracy: {metrics['accuracy']:.2%}")  # Accuracy: 80.00%
        >>> print(f"F1: {metrics['f1']:.4f}")  # F1: 0.8000
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="weighted")),
    }
