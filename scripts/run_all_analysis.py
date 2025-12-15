import os
import pandas as pd

from src.models.pca_analyzer import PCAAnalyzer
from src.models.cluster_analyzer import ClusterAnalyzer
from src.models.spatial_model import SpatialModel
from src.models.network_analyzer import NetworkAnalyzer
from src.models.ml_predictor import MLPredictor

from src.visualization.plot_utils import PlotUtils
from src.utils.logger import logger


def run_all_analysis(data_path: str, results_dir: str = "results"):
    logger.info("===== 全部分析流程开始 =====")

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 读取数据（预处理后的结果）
    df = pd.read_csv(data_path)
    X = df.select_dtypes(include="number").values

    # PCA
    logger.info("开始 PCA 分析")
    pca = PCAAnalyzer()
    pca_results = pca.run_pca(X)
    pca.save_results(pca_results, os.path.join(results_dir, "pca_results.csv"))

    # 绘制 PCA 可视化图
    PlotUtils.plot_pca_2d(pca_results["components"], 
                          pca_results["explained_variance_ratio"],
                          os.path.join(results_dir, "pca_plot.png"))

    # 聚类
    logger.info("开始聚类分析")
    cluster = ClusterAnalyzer()
    labels, metrics = cluster.cluster(X)

    cluster.save_results(labels, metrics, os.path.join(results_dir, "cluster_results.json"))

    # 空间分析
    logger.info("开始空间分析")
    spatial = SpatialModel()
    spatial_results = spatial.analyze(df)
    spatial.save_results(spatial_results, os.path.join(results_dir, "spatial_results.json"))

    # 网络分析
    logger.info("开始网络分析")
    network = NetworkAnalyzer()
    graph = network.build_graph(df)
    network.save_graph(graph, os.path.join(results_dir, "network.graphml"))

    # 机器学习预测
    logger.info("开始机器学习预测")
    ml = MLPredictor()
    y_true, y_pred, metrics = ml.run(df)
    ml.save_results(y_true, y_pred, metrics, os.path.join(results_dir, "ml_results.json"))

    logger.info("===== 全部分析流程结束 =====")


if __name__ == "__main__":
    run_all_analysis("data/processed/cleaned_data.csv")
