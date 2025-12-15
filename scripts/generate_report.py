import os
import json
import pandas as pd

from src.utils.logger import logger


def generate_report(results_dir: str = "results", output_file: str = "report.md"):
    logger.info("===== 开始生成报告 =====")

    # 读取结果文件
    pca_file = os.path.join(results_dir, "pca_results.csv")
    cluster_file = os.path.join(results_dir, "cluster_results.json")
    ml_file = os.path.join(results_dir, "ml_results.json")

    pca_df = pd.read_csv(pca_file)

    with open(cluster_file, "r") as f:
        cluster_res = json.load(f)

    with open(ml_file, "r") as f:
        ml_res = json.load(f)

    # 生成 markdown 报告
    md_text = f"""
# 综合分析报告

## 1. PCA 主成分分析
解释方差比例：
{pca_df['explained_variance_ratio'].tolist()}

## 2. 聚类分析
- 聚类簇数量: {len(set(cluster_res['labels']))}
- 轮廓系数: {cluster_res['metrics'].get('silhouette')}
- DBI 指数: {cluster_res['metrics'].get('davies_bouldin')}

## 3. 机器学习预测
- MSE: {ml_res['metrics']['mse']}
- RMSE: {ml_res['metrics']['rmse']}
- R2: {ml_res['metrics']['r2']}

"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(md_text)

    logger.info(f"报告已生成: {output_file}")
    logger.info("===== 报告生成结束 =====")


if __name__ == "__main__":
    generate_report()
