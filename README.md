# 粤港澳大湾区数据要素流动多元统计分析

## 概述
本项目旨在对粤港澳大湾区（11 城市）数据要素跨境流动开展系统化、多方法的量化分析。涵盖数据预处理、统计分析、空间计量、网络分析、机器学习预测与因果推断等方法，并强调代码工程化与可复现性。

## 快速开始

1. 克隆仓库并进入目录  
git clone <repo-url>
cd 粤港澳数据要素流动实验

2. 创建虚拟环境并安装依赖（使用 conda 或 venv）
conda env create -f environment.yml
conda activate gdai
pip install -r requirements.txt

3. 运行预处理脚本
python scripts/run_preprocessing.py --config config/config.yaml

## 开发规范
代码风格：PEP8，使用black格式化。
类型检查：mypy（类型注解覆盖率 ≥ 90%）。
测试：pytest（覆盖率目标 ≥ 80%）。
文档：Sphinx 自动生成 API 文档。

## 贡献
请遵循 Git 分支策略（main/dev/feature），提交信息遵循 Conventional Commits。

## 许可证
MIT
