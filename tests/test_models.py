"""
测试 PCA、聚类、空间分析、网络分析、机器学习预测
"""

import numpy as np
import pandas as pd

from src.models.pca_analyzer import PCAAnalyzer
from src.models.cluster_analyzer import ClusterAnalyzer
from src.models.spatial_model import SpatialModel
from src.models.network_analyzer import NetworkAnalyzer
from src.models.ml_predictor import MLPredictor


def test_pca_analyzer():
    X = np.random.rand(20, 5)
    pca = PCAAnalyzer()
    results = pca.run_pca(X)

    assert "components" in results
    assert results["components"].shape[1] <= 5


def test_cluster_analyzer():
    X = np.random.rand(20, 3)
    cluster = ClusterAnalyzer()
    labels, metrics = cluster.cluster(X)

    assert len(labels) == 20
    assert isinstance(metrics, dict)


def test_spatial_model():
    df = pd.DataFrame({
        "x": [0, 1],
        "y": [1, 0],
        "value": [10, 20]
    })
    spatial = SpatialModel()
    results = spatial.analyze(df)

    assert isinstance(results, dict)
    assert "summary" in results


def test_network_analyzer():
    df = pd.DataFrame({
        "source": ["A", "A", "B"],
        "target": ["B", "C", "C"],
        "weight": [1, 2, 3]
    })
    network = NetworkAnalyzer()
    graph = network.build_graph(df)

    assert graph.number_of_nodes() >= 2
    assert graph.number_of_edges() == 3


def test_ml_predictor():
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4],
        "target": [10, 20, 30, 40]
    })

    ml = MLPredictor()
    y_true, y_pred, metrics = ml.run(df)

    assert len(y_true) == len(y_pred)
    assert "mse" in metrics
