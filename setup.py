from setuptools import setup, find_packages

setup(
    name="gdai_greater_bay_dataflow",
    version="0.1.0",
    description="粤港澳大湾区数据要素流动多元统计分析实验",
    author="彭景辉",
    author_email="1456850136@qq.com",
    url="https://github.com/Klein-041105/gba_data_flow.git",
    license="MIT",
    packages=find_packages(exclude=("tests", "notebooks", "outputs")),
    include_package_data=True,
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.1",
        "scikit-learn>=1.2",
        "statsmodels>=0.14",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "networkx>=3.1",
        "pyyaml>=6.0",
    ],
    extras_require={
        "spatial": ["geopandas>=0.13", "shapely>=2.0"],
        "ml": ["xgboost>=1.7", "lightgbm>=4.4"],
        "deep": ["tensorflow>=2.12", "torch>=2.0"],
        "dev": ["pytest>=7.3", "black>=24.3", "mypy>=1.5"],
    },
    python_requires=">=3.10",
)
