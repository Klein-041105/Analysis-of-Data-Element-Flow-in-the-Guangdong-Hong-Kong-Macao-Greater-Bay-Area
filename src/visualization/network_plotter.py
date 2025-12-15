from __future__ import annotations
from typing import Dict, Optional, Sequence, Any
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from .plot_utils import new_figure, save_figure
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def draw_force_layout(
    G: nx.Graph,
    node_size_map: Optional[Dict[Any, float]] = None,
    node_color_map: Optional[Dict[Any, float]] = None,
    with_labels: bool = True,
    figsize: tuple = (10, 8),
    save_to: Optional[str] = None,
):

    fig = new_figure(figsize)
    ax = fig.add_subplot(111)

    pos = nx.spring_layout(G, seed=42)

    nodes = list(G.nodes())
    sizes = [node_size_map.get(n, 300) if node_size_map else 300 for n in nodes]
    colors = [node_color_map.get(n, 0.5) if node_color_map else 0.5 for n in nodes]

    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.7, ax=ax)
    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)

    ax.axis("off")
    if save_to:
        save_figure(fig, save_to)
    return fig


def draw_circular_layout(
    G: nx.Graph,
    node_order: Optional[Sequence[Any]] = None,
    with_labels: bool = True,
    figsize: tuple = (10, 8),
    save_to: Optional[str] = None,
):
    fig = new_figure(figsize)
    ax = fig.add_subplot(111)
    if node_order:
        pos = nx.circular_layout(G)
        # reorder nodes by node_order for consistent label placement (networkx manages positions)
    else:
        pos = nx.circular_layout(G)

    nx.draw(G, pos, with_labels=with_labels, ax=ax)
    ax.axis("off")
    if save_to:
        save_figure(fig, save_to)
    return fig


def draw_time_series_networks(
    graphs: Dict[int, nx.Graph],
    out_dir: Optional[str] = None,
    prefix: str = "network_",
    figsize: tuple = (10, 8),
):

    saved_files = []
    for year, G in sorted(graphs.items()):
        fname = None
        if out_dir:
            Path = __import__("pathlib").Path
            p = Path(out_dir) / f"{prefix}{year}.png"
            fname = str(p)
        fig = draw_force_layout(G, figsize=figsize, save_to=fname)
        saved_files.append(fname or f"{prefix}{year}.png")
    return saved_files
