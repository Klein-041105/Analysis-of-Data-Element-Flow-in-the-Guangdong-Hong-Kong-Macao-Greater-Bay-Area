from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

# 使用标准 logging 以便 CI/测试环境捕获
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class DataLoader:
    """数据加载器类。
    
    负责从多种格式文件中读取数据，管理数据目录结构。
    支持的格式：CSV、Excel、Parquet、JSON。
    """

    def __init__(self, base_dir: Union[str, Path]) -> None:
        """初始化DataLoader。
        
        Args:
            base_dir: 项目根目录路径
            
        Attributes:
            base_dir: 项目根目录Path对象
            raw_dir: 原始数据目录(data/raw)
            processed_dir: 处理后数据目录(data/processed)
            interim_dir: 中间数据目录(data/interim)
        """
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "data" / "raw"
        self.processed_dir = self.base_dir / "data" / "processed"
        self.interim_dir = self.base_dir / "data" / "interim"

    def _check_file(self, path: Path) -> None:
        """检查文件是否存在。
        
        Args:
            path: 文件路径
            
        Raises:
            FileNotFoundError: 当文件不存在时
        """
        if not path.exists():
            logger.error("File not found: %s", path)
            raise FileNotFoundError(f"{path} not found")

    def read_table(
        self, filename: str, parse_dates: Optional[List[str]] = None, **kwargs
    ) -> pd.DataFrame:
        """读取数据表文件。
        
        支持CSV、Excel、Parquet、JSON格式文件。
        
        Args:
            filename: 文件名或绝对路径。若为相对路径，则从raw_dir读取
            parse_dates: 指定要解析为日期的列名列表
            **kwargs: 传递给对应pandas读取函数的参数
                - CSV/Excel: sep, encoding, sheet_name等
                - JSON: orient, lines等
            
        Returns:
            读取后的DataFrame
            
        Raises:
            FileNotFoundError: 当文件不存在时
            ValueError: 当文件格式不支持时
            
        Examples:
            >>> loader = DataLoader('.')
            >>> df = loader.read_table('data.csv')
            >>> df_dates = loader.read_table('data.csv', parse_dates=['date_col'])
        """
        path = Path(filename)
        if not path.is_absolute():
            path = self.raw_dir / filename
        self._check_file(path)

        suffix = path.suffix.lower()
        logger.info("Loading file %s", path)
        if suffix in [".csv", ".txt"]:
            df = pd.read_csv(path, parse_dates=parse_dates, **kwargs)
        elif suffix in [".xls", ".xlsx"]:
            df = pd.read_excel(path, parse_dates=parse_dates, **kwargs)
        elif suffix in [".parquet"]:
            df = pd.read_parquet(path, **kwargs)
        elif suffix in [".json"]:
            df = pd.read_json(path, **kwargs)
        else:
            raise ValueError(f"Unsupported file suffix: {suffix}")
        logger.info("Loaded shape: %s", df.shape)
        return df

    def load_od_matrices(self, od_config: Dict) -> Dict[int, Dict[str, pd.DataFrame]]:
        """加载出入流（OD）矩阵数据。
        
        从配置中读取年份和矩阵类型，按约定的命名规则加载OD矩阵文件。
        缺失的文件会被跳过并记录警告信息。
        
        Args:
            od_config: 配置字典，应包含：
                - 'years': 年份列表 (如 [2019, 2020, 2021])
                - 'matrix_type': 矩阵类型列表 (如 ['inflow', 'outflow'])
            
        Returns:
            嵌套字典 {年份: {类型: DataFrame}}。
            结构如：{2019: {'inflow': df1, 'outflow': df2}, 2020: {...}}
            
        Examples:
            >>> config = {'years': [2019, 2020], 'matrix_type': ['inflow', 'outflow']}
            >>> od_data = loader.load_od_matrices(config)
            >>> od_2019_inflow = od_data[2019]['inflow']
        """
        years = od_config.get("years", [])
        types = od_config.get("matrix_type", [])
        results: Dict[int, Dict[str, pd.DataFrame]] = {}
        for y in years:
            results[y] = {}
            for t in types:
                fname = f"od_{y}_{t}.csv"
                try:
                    df = self.read_table(fname)
                except FileNotFoundError:
                    logger.warning("OD file not found: %s (skipped)", fname)
                    continue
                results[y][t] = df
        return results

    def save_processed(self, df: pd.DataFrame, name: str) -> Path:
        """保存处理后的数据为Parquet格式。
        
        自动创建processed目录（若不存在）。
        
        Args:
            df: 要保存的DataFrame
            name: 文件名（不包含扩展名，自动添加.parquet）
            
        Returns:
            保存文件的完整Path对象
            
        Examples:
            >>> loader = DataLoader('.')
            >>> path = loader.save_processed(df, 'processed_data')
            >>> print(path)  # data/processed/processed_data.parquet
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.processed_dir / f"{name}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Saved processed data to %s", out_path)
        return out_path

    def load_config(self, path: Union[str, Path]) -> Dict:
        """加载JSON配置文件。
        
        Args:
            path: 配置文件路径（绝对或相对）
            
        Returns:
            解析后的配置字典
            
        Raises:
            FileNotFoundError: 当配置文件不存在时
            ValueError: 当文件格式不是JSON时
            json.JSONDecodeError: 当JSON解析失败时
            
        Examples:
            >>> loader = DataLoader('.')
            >>> config = loader.load_config('config/config.json')
            >>> years = config['years']
        """
        p = Path(path)
        self._check_file(p)
        if p.suffix.lower() in [".json"]:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        raise ValueError("Unsupported config type for load_config; use JSON")


# 简易测试钩子
def _example_usage() -> None:
    dl = DataLoader(".")
    try:
        _ = dl.read_table("example.csv")
    except FileNotFoundError:
        logger.info("example.csv not present — this is just a usage demo.")


if __name__ == "__main__":
    _example_usage()
