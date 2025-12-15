from __future__ import annotations
from typing import Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, out_path: str, dpi: int = 150, tight: bool = True) -> None:
    """保存图表到文件。
    
    自动创建目录结构（如果不存在）。
    
    Args:
        fig: matplotlib图表对象
        out_path: 输出文件路径
        dpi: 分辨率（默认150）
        tight: 是否使用tight_layout（默认True）
        
    Raises:
        Exception: 当文件保存失败时
    """
    p = Path(out_path)
    ensure_dir(p)
    try:
        if tight:
            fig.tight_layout()
        fig.savefig(p, dpi=dpi)
        logger.info("Saved figure to %s", p)
    except Exception as e:
        logger.exception("Failed to save figure %s: %s", out_path, e)
        raise


def new_figure(figsize: Tuple[float, float] = (8, 4)) -> plt.Figure:

    fig = plt.figure(figsize=figsize)
    return fig


def plot_timeseries(
    df,
    cols: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 4),
    save_to: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """绘制时间序列数据。
    
    Args:
        df: 包含时间序列数据的DataFrame
        cols: 要绘制的列名列表，为None时绘制所有列
        title: 图表标题
        figsize: 图表大小
        save_to: 保存路径，为None时不保存
        
    Returns:
        (图表对象, 坐标轴对象)
    """
    fig = new_figure(figsize)
    ax = fig.add_subplot(111)
    if cols is None:
        df.plot(ax=ax)
    else:
        df[list(cols)].plot(ax=ax)
    if title:
        ax.set_title(title)
    ax.grid(True)

    if save_to:
        save_figure(fig, save_to)
    return fig, ax


def plot_bar(
    series,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 4),
    save_to: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """绘制条形图。
    
    Args:
        series: pandas Series或dict，包含要绘制的数据
        title: 图表标题
        xlabel: x轴标签
        ylabel: y轴标签
        figsize: 图表大小
        save_to: 保存路径，为None时不保存
        
    Returns:
        (图表对象, 坐标轴对象)
    """
    fig = new_figure(figsize)
    ax = fig.add_subplot(111)
    series.plot(kind="bar", ax=ax)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y")
    if save_to:
        save_figure(fig, save_to)
    return fig, ax
