# 用户使用手册（User Guide）

本手册将指导您如何运行此项目，包括环境准备、目录结构、脚本使用方法等。

---

## 1. 环境准备

### 1.1 安装依赖

确保您已安装 Python 3.9+。

安装依赖：
```bash
pip install -r requirements.txt
```

## 2. 项目结构

project/
│── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── utils/
│   └── visualization/
│── scripts/
│── tests/
│── docs/
│── data/
│   ├── raw/
│   └── processed/
│── results/
│── config/

## 3. 数据处理指南

### 3.1 放置原始数据

将原始数据放入：
    data/raw/

### 3.2 运行预处理脚本

    python scripts/run_preprocessing.py
处理完成后，新数据将保存到：
    data/processed/cleaned_data.csv

## 4. 运行完整分析流程

执行项目中所有分析：
    python scripts/run_all_analysis.py
生成内容包含：
PCA 结果
聚类分析结果
空间分析结果
网络图文件（graphml）
机器学习预测结果
所有输出将保存到：
    results/

## 5. 自动生成报告
执行：
    python scripts/generate_report.py
输出：
    report.md
包括模型结果、图表、指标汇总等。

## 6. 测试模块说明
使用 pytest：
    pytest tests/
可快速验证所有模块功能是否正常。

## 7. 可视化输出
所有图像将存储在：
    results/
例如：
PCA 可视化
聚类散点图
网络可视化（如导出后在 Gephi 查看）

## 8. 常见问题 FAQ

Q: 输入数据必须是什么格式？

A: 任何能读取为 DataFrame 的 CSV/Excel 文件。

Q: 可否自定义分析步骤？

A: 可以，通过修改 scripts 中的流程脚本即可。

Q: 能否扩展模型？

A: 完全可以，例如在 MLPredictor 中加入 XGBoost 等。


