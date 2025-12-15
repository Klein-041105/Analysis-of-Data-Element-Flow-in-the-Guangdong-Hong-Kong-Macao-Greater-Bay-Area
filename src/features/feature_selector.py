from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_regression,
    RFE,
)
from sklearn.ensemble import RandomForestRegressor
import logging

logger = logging.getLogger(__name__)


class FeatureSelector:
    """特征选择器。
    
    支持多种特征筛选方法：方差阈值、相关性过滤、互信息、随机森林重要性、RFE。
    """

    def __init__(
        self,
        variance_threshold: float = 0.0,
        corr_threshold: float = 0.95,
        top_k: Optional[int] = None,
        use_rf_importance: bool = True,
        use_rfe: bool = False,
        rfe_n_features: Optional[int] = None,
    ):
        """初始化FeatureSelector。
        
        Args:
            variance_threshold: 方差阈值，低于该值的特征被过滤
            corr_threshold: 相关性阈值，高于该值的特征对仅保留一个
            top_k: 互信息和RF方法保留的特征数
            use_rf_importance: 是否使用随机森林重要性
            use_rfe: 是否使用递归特征消除
            rfe_n_features: RFE选择的特征数，若为None则选择特征总数的50%
        """
        self.variance_threshold = variance_threshold
        self.corr_threshold = corr_threshold
        self.top_k = top_k
        self.use_rf_importance = use_rf_importance
        self.use_rfe = use_rfe
        self.rfe_n_features = rfe_n_features

        self.selected_features_: List[str] = []

    def apply_variance_threshold(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用方差阈值过滤特征。
        
        移除方差低于阈值的特征（信息量少）。
        
        Args:
            df: 输入DataFrame
            
        Returns:
            过滤后的DataFrame
        """
        selector = VarianceThreshold(self.variance_threshold)
        arr = selector.fit_transform(df)
        kept = df.columns[selector.get_support()].tolist()

        logger.info(f"Variance filtering kept {len(kept)} features")
        return df[kept]

    def apply_correlation_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用相关性过滤移除高度相关的特征。
        
        对每对高度相关的特征，仅保留其中一个。
        
        Args:
            df: 输入DataFrame
            
        Returns:
            过滤后的DataFrame
        """
        logger.info("Applying correlation filtering...")
        corr = df.corr().abs()

        upper = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)
        )

        to_drop = [
            column for column in upper.columns
            if any(upper[column] > self.corr_threshold)
        ]

        logger.info(f"Correlation filter dropped {len(to_drop)} features")

        return df.drop(columns=to_drop)

    def apply_mutual_info(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """使用互信息进行特征排名。
        
        计算每个特征与目标变量之间的互信息，分数高表示重要性大。
        
        Args:
            X: 特征DataFrame
            y: 目标Series
            
        Returns:
            选中的特征名列表（若设置top_k则为前K个，否则全部）
        """
        logger.info("Computing mutual information...")
        mi = mutual_info_regression(X, y)
        mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)

        if self.top_k:
            selected = mi_series.iloc[: self.top_k].index.tolist()
        else:
            selected = mi_series.index.tolist()

        return selected

    def apply_rf_importance(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """使用随机森林特征重要性进行特征排名。
        
        训练随机森林并根据特征重要性排名。
        
        Args:
            X: 特征DataFrame
            y: 目标Series
            
        Returns:
            按重要性排序的特征名列表（若设置top_k则为前K个）
        """
        logger.info("Computing RandomForest feature importance...")
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X, y)
        importance = pd.Series(rf.feature_importances_, index=X.columns)
        importance = importance.sort_values(ascending=False)

        if self.top_k:
            selected = importance.iloc[: self.top_k].index.tolist()
        else:
            selected = importance.index.tolist()

        return selected

    def apply_rfe(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """使用递归特征消除（RFE）进行特征选择。
        
        通过迭代训练模型并消除最不重要的特征来进行选择。
        
        Args:
            X: 特征DataFrame
            y: 目标Series
            
        Returns:
            选中的特征名列表
        """
        logger.info("Performing RFE selection...")
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        n_features = self.rfe_n_features or max(1, X.shape[1] // 2)

        selector = RFE(rf, n_features_to_select=n_features)
        selector.fit(X, y)

        selected = X.columns[selector.support_].tolist()
        return selected

    def select_features(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """执行完整的特征选择流程。
        
        按顺序应用以下过滤：
        1. 方差阈值
        2. 相关性过滤
        3. 互信息选择
        4. 随机森林重要性（可选）
        5. RFE选择（可选）
        
        最终结果是所有选定方法的交集。
        
        Args:
            X: 特征DataFrame
            y: 目标Series
            
        Returns:
            选中的特征名列表
        """
        logger.info("Starting feature selection...")

        df = X.copy()

        # 1. Variance threshold
        df = self.apply_variance_threshold(df)

        # 2. Correlation filter
        df = self.apply_correlation_filter(df)

        # Candidates from MI / RF / RFE
        candidates = set(df.columns)

        # 3. Mutual Information
        mi_feats = self.apply_mutual_info(df, y)
        candidates &= set(mi_feats)

        # 4. RandomForest importance
        if self.use_rf_importance:
            rf_feats = self.apply_rf_importance(df, y)
            candidates &= set(rf_feats)

        # 5. RFE
        if self.use_rfe:
            rfe_feats = self.apply_rfe(df, y)
            candidates &= set(rfe_feats)

        self.selected_features_ = list(candidates)
        logger.info(f"Final selected features: {len(self.selected_features_)}")

        return self.selected_features_

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """使用选中的特征进行数据变换。
        
        Args:
            X: 输入DataFrame
            
        Returns:
            仅包含选中特征的DataFrame
            
        Raises:
            RuntimeError: 当未先调用select_features()时
        """
        if not self.selected_features_:
            raise RuntimeError("Must call select_features() before transform().")

        return X[self.selected_features_]
