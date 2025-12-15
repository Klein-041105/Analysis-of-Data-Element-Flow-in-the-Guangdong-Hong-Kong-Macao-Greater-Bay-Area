# 方法论说明（Methodology）

本项目旨在构建一个完整的数据分析与建模平台，包括：
- 数据预处理
- 特征工程
- 主成分分析（PCA）
- 聚类分析
- 空间分析（Spatial Analysis）
- 网络分析（Network Analysis）
- 机器学习建模（ML Modeling）
- 自动化报告生成

---

## 1. 数据预处理流程

### 1.1 数据加载  
使用 `DataLoader` 统一读取 CSV/Excel/Parquet 等文件。

### 1.2 数据清洗  
包含：
- 列名标准化
- 空白字符修剪
- 重复值处理
- 类别与数值类型修正

### 1.3 缺失值处理  
采用多策略融合：
- 均值/中位数填充
- KNN 插补
- 多重插补 MICE
- AutoEncoder 插补（深度学习）

### 1.4 离群值处理  
包括：
- Z-score
- IForest
- LOF（局部离群因子）
- Grubbs Test

---

## 2. 特征工程（Feature Engineering）

提供：
- 多项式特征
- 特征组合（交互项）
- 归一化、标准化

特征选择模块使用：
- 方差过滤
- 相关性过滤
- 基于模型的特征重要性（可扩展）

---

## 3. PCA 主成分分析

使用 `sklearn.decomposition.PCA`，目标：
- 降维
- 提取高贡献主成分
- 绘制散点图、贡献率图

输出包含：
- 成分矩阵
- 方差解释比例
- 转换后的低维空间坐标

---

## 4. 聚类分析

采用如下模型：
- KMeans
- DBSCAN
- 层次聚类（Agglomerative Clustering）

评估指标：
- Silhouette Score
- Davies-Bouldin Index

---

## 5. 空间分析（Spatial Analysis）

提供：
- 基于 DataFrame 的空间趋势分析
- 邻域效应分析
- 空间分布 summary（中心、离散度、边界）

后续可扩展：
- 空间自相关（Moran’s I）
- 栅格化网格分析

---

## 6. 网络分析（Network Analysis）

使用 `networkx` 进行图分析：
- 构建有向/无向图
- 节点/边统计
- 路径分析
- 网络特征：度、介数中心性等

导出格式：
- GraphML（支持 Gephi 等工具可视化）

---

## 7. 机器学习建模（ML Predictor）

默认采用线性回归，可扩展为：
- XGBoost
- RandomForest
- CatBoost
- Deep Learning

输出指标：
- MSE
- RMSE
- R²
- 分类模型则提供 Accuracy / F1

---

## 8. 自动化报告生成

读取 results/ 中的数据，生成 Markdown 格式报告，包括：
- PCA 结果
- 聚类结果
- ML 指标
- 可扩展空间和网络分析总结

---

## 9. 系统整体流程图

原始数据 → 预处理 → 特征工程 → PCA → 聚类 → 空间分析 → 网络分析 → ML 预测 → 报告生成

---

## 10. 可扩展性

本项目支持：
- 增加新模型
- 扩展新分析方法
- 动态更换配置（通过 config/）
- 插槽式 pipeline

可用于科研、业务分析、大模型数据预处理、因果推断前置分析等场景。



