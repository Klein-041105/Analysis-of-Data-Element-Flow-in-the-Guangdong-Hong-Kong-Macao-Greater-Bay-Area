from __future__ import annotations
from typing import Optional
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from .plot_utils import save_figure, new_figure

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def plot_city_choropleth(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    cmap: str = "viridis",
    title: Optional[str] = None,
    figsize: tuple = (10, 8),
    save_to: Optional[str] = None,
) -> plt.Figure:

    fig = new_figure(figsize)
    ax = fig.add_subplot(111)
    gdf.plot(column=value_col, ax=ax, legend=True, cmap=cmap, edgecolor="black")
    if title:
        ax.set_title(title)
    ax.axis("off")
    if save_to:
        save_figure(fig, save_to)
    return fig


def plot_lisa_clusters(
    gdf: gpd.GeoDataFrame,
    lisa_cluster_col: str = "lisa_cluster",
    title: Optional[str] = None,
    figsize: tuple = (10, 8),
    save_to: Optional[str] = None,
):

    fig = new_figure(figsize)
    ax = fig.add_subplot(111)
    # ensure it's categorical for consistent coloring
    gdf = gdf.copy()
    if lisa_cluster_col not in gdf.columns:
        raise ValueError(f"{lisa_cluster_col} not in GeoDataFrame")

    gdf[lisa_cluster_col] = gdf[lisa_cluster_col].astype("category")
    gdf.plot(column=lisa_cluster_col, categorical=True, legend=True, ax=ax, edgecolor="black")
    if title:
        ax.set_title(title)
    ax.axis("off")
    if save_to:
        save_figure(fig, save_to)
    return fig
