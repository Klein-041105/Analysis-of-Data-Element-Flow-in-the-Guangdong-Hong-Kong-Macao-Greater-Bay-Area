import os
import pandas as pd

from src.data.data_loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.data.missing_handler import MissingValueHandler
from src.data.outlier_detector import OutlierDetector

from src.utils.logger import logger


def run_preprocessing(data_path: str, output_path: str):
    logger.info("===== 数据预处理流程开始 =====")

    # 步骤 1: 数据加载
    loader = DataLoader()
    df = loader.load_data(data_path)

    # 步骤 2: 数据基本预处理
    pre = Preprocessor()
    df_clean = pre.clean_data(df)
    df_clean = pre.normalize_column_names(df_clean)

    # 步骤 3: 缺失值处理
    mvh = MissingValueHandler()
    df_filled = mvh.fill_missing_multi_strategy(df_clean)

    # 步骤 4: 离群值检测
    od = OutlierDetector()
    df_filtered = od.handle_outliers(df_filled)

    # 步骤 5: 保存处理后的数据
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_file = os.path.join(output_path, "cleaned_data.csv")
    df_filtered.to_csv(output_file, index=False)

    logger.info(f"数据预处理完成，已保存到: {output_file}")
    logger.info("===== 数据预处理流程结束 =====")


if __name__ == "__main__":
    run_preprocessing("data/raw/data.csv", "data/processed")
