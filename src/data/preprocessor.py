from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer

from .data_loader import DataLoader
from .missing_handler import (
    ImputationResult,
    autoencoder_impute,
    knn_impute,
    mice_impute,
    timeseries_impute,
)
from .outlier_detector import (
    aggregate_outlier_report,
    grubbs_test,
    isolation_forest_detect,
    lof_detect,
    timeseries_residual_detect,
)

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class DataQualityReport:
    missing_rate: pd.Series
    outlier_summary: Dict
    distribution_summary: pd.DataFrame = field(default_factory=pd.DataFrame)


class Preprocessor:

    def __init__(self, base_dir: str):
        self.loader = DataLoader(base_dir)

    def compute_missing_rate(self, df: pd.DataFrame) -> pd.Series:
        """计算数据缺失率。
        
        Args:
            df: 输入数据框
            
        Returns:
            各列的缺失率（0-1之间的值）
        """
        return df.isna().mean()

    def run_imputation(
        self,
        df: pd.DataFrame,
        method: str = "mice",
        **kwargs,
    ) -> ImputationResult:
        """运行缺失值填补。
        
        Args:
            df: 含缺失值的数据框
            method: 填补方法，可选 ['mice', 'knn', 'timeseries', 'autoencoder']
            **kwargs: 传递给对应填补函数的参数
            
        Returns:
            ImputationResult 对象，包含填补后的数据和统计信息
            
        Raises:
            ValueError: 当method不在支持的方法列表中时
        """

        method = method.lower()
        if method == "mice":
            return mice_impute(df, **kwargs)
        if method == "knn":
            return knn_impute(df, **kwargs)
        if method == "timeseries":
            return timeseries_impute(df, **kwargs)
        if method == "autoencoder":
            return autoencoder_impute(df, **kwargs)
        raise ValueError(f"Unknown imputation method: {method}")

    def run_outlier_detection(
        self, df: pd.DataFrame, numeric_cols: Optional[List[str]] = None
    ) -> Dict[str, object]:
        """检测数据中的离群值。
        
        使用多种方法（Grubbs、Isolation Forest、LOF、时间序列残差）检测离群值。
        
        Args:
            df: 输入数据框
            numeric_cols: 数值列名列表，若为None则自动选择
            
        Returns:
            包含离群值检测报告的字典，主要包括离群值标记和检测方法统计
        """
        detectors = []
        # Grubbs: 对每个数值列单独运行并合并 flags
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for c in numeric_cols:
            res = grubbs_test(df[c])
            # rename the series so aggregate 能识别不冲突
            flags = res["is_outlier"].rename(f"grubbs_{c}")
            detectors.append({"flags": flags, "method": f"grubbs_{c}"})

        # Isolation Forest (multivariate)
        res_if = isolation_forest_detect(df, numeric_cols=numeric_cols)
        detectors.append({"flags": res_if["flags"], "method": "isolation_forest"})

        # LOF
        res_lof = lof_detect(df, numeric_cols=numeric_cols)
        detectors.append({"flags": res_lof["flags"], "method": "lof"})

        # For time-series columns, try residual-based detection
        # (if index is datetime-like)
        if isinstance(df.index, pd.DatetimeIndex):
            for c in numeric_cols:
                try:
                    tres = timeseries_residual_detect(df[c])
                    detectors.append({"flags": tres["flags"], "method": f"ts_resid_{c}"})
                except Exception:
                    logger.debug("Time series residual detect failed for %s", c)

        agg = aggregate_outlier_report(df, detectors)
        return agg

    def transform_data(
        self,
        df: pd.DataFrame,
        numeric_cols: Optional[List[str]] = None,
        method: str = "standard",
    ) -> Tuple[pd.DataFrame, object]:
        """进行数据变换（标准化、归一化、幂变换等）。
        
        Args:
            df: 输入数据框
            numeric_cols: 数值列名列表，若为None则自动选择
            method: 变换方法，可选 ['standard', 'minmax', 'boxcox', 'yeo-johnson', 'log']
            
        Returns:
            元组 (变换后的数据框, 变换器对象)，变换器可用于后续数据变换
            
        Raises:
            ValueError: 当method不在支持的方法列表中时
        """
        df_out = df.copy()
        if numeric_cols is None:
            numeric_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()
        if method == "standard":
            transformer = StandardScaler()
            df_out[numeric_cols] = transformer.fit_transform(df_out[numeric_cols])
        elif method == "minmax":
            transformer = MinMaxScaler()
            df_out[numeric_cols] = transformer.fit_transform(df_out[numeric_cols])
        elif method in ("boxcox", "yeo-johnson"):
            # Box-Cox 要求正值，这里使用 PowerTransformer
            if method == "boxcox":
                pt = PowerTransformer(method="box-cox")
            else:
                pt = PowerTransformer(method="yeo-johnson")
            df_out[numeric_cols] = pt.fit_transform(df_out[numeric_cols].fillna(0) + 1e-6)
            transformer = pt
        elif method == "log":
            for c in numeric_cols:
                df_out[c] = np.log1p(df_out[c].clip(lower=0))
            transformer = "log"
        else:
            raise ValueError(f"Unknown transform method: {method}")
        return df_out, transformer

    def generate_quality_report(self, df: pd.DataFrame) -> DataQualityReport:
        """生成数据质量评估报告。
        
        Args:
            df: 输入数据框
            
        Returns:
            DataQualityReport 对象，包含缺失率、离群值统计和分布摘要
        """
        missing_rate = self.compute_missing_rate(df)
        # distribution summary
        desc = df.describe(include="all").transpose()
        # outlier summary is empty here; user should run run_outlier_detection separately
        return DataQualityReport(missing_rate=missing_rate, outlier_summary={}, distribution_summary=desc)

    def preprocess_pipeline(
        self,
        df: pd.DataFrame,
        impute_method: str = "mice",
        transform_method: str = "standard",
        outlier_detection: bool = True,
        **impute_kwargs,
    ) -> Tuple[pd.DataFrame, DataQualityReport, Optional[Dict[str, object]]]:
        """执行完整的预处理流程。
        
        一站式预处理函数，顺序执行：
        1. 生成初始数据质量报告
        2. 缺失值填补（MICE、KNN、TimeSeries或Autoencoder）
        3. 可选的离群值检测（多种算法投票）
        4. 数据变换/标准化（StandardScaler、MinMaxScaler、PowerTransformer等）
        5. 生成最终质量报告
        
        Args:
            df: 原始输入数据框
            impute_method: 缺失值填补方法，可选 ['mice', 'knn', 'timeseries', 'autoencoder']
            transform_method: 数据变换方法，可选 ['standard', 'minmax', 'boxcox', 'yeo-johnson', 'log']
            outlier_detection: 是否执行离群值检测（默认True）
            **impute_kwargs: 传递给fill方法的参数
                - MICE: n_iter, random_state
                - KNN: n_neighbors, weights
                - TimeSeries: time_col, order
                - Autoencoder: latent_dim, epochs, batch_size
            
        Returns:
            元组 (预处理后的数据框, 最终质量报告, 离群值检测结果或None)
            
            其中：
            - 预处理后的数据框：已填补、变换且标准化
            - 最终质量报告：包含缺失率、分布统计
            - 离群值结果：若outlier_detection=True，为离群值统计字典；否则为None
            
        Examples:
            >>> preprocessor = Preprocessor('.')
            >>> df_clean, report, outliers = preprocessor.preprocess_pipeline(
            ...     df_raw,
            ...     impute_method='mice',
            ...     transform_method='standard',
            ...     outlier_detection=True
            ... )
            >>> print(report.missing_rate)  # 检查缺失率
            >>> print(df_clean.shape)  # (n_samples, n_features)
        """
        logger.info("Starting preprocessing pipeline")
        dq = self.generate_quality_report(df)
        imputed = self.run_imputation(df, method=impute_method, **impute_kwargs)
        outlier_report = None
        if outlier_detection:
            outlier_report = self.run_outlier_detection(imputed.df_filled)
        transformed_df, transformer = self.transform_data(imputed.df_filled, method=transform_method)
        # update quality report with post-imputation missing rate
        dq_post = self.generate_quality_report(transformed_df)
        return transformed_df, dq_post, outlier_report
