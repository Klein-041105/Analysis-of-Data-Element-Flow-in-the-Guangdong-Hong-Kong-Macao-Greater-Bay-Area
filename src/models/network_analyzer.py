from __future__ import annotations
import pandas as pd
import networkx as nx
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class NetworkAnalyzer:

    def __init__(self, edges: pd.DataFrame):

        if "source" not in edges or "target" not in edges:
            raise ValueError("Edges DataFrame must have columns: source, target")

        self.edges = edges
        self.graph = nx.from_pandas_edgelist(
            edges,
            source="source",
            target="target",
            edge_attr=True,
            create_using=nx.Graph(),
        )

    def basic_metrics(self) -> Dict[str, Any]:
        """计算基本的网络拓扑指标。
        
        计算包括：
        - 节点度数
        - 中介中心性
        - 接近中心性
        
        Returns:
            包含各类指标的字典
        """
        logger.info("Computing network metrics...")

        degree = dict(self.graph.degree())
        betweenness = nx.betweenness_centrality(self.graph)
        closeness = nx.closeness_centrality(self.graph)

        return {
            "degree": degree,
            "betweenness": betweenness,
            "closeness": closeness,
        }

    def communities(self) -> Dict[str, Any]:
        """检测网络中的社区结构。
        
        使用贪心模块度优化算法进行社区检测。
        
        Returns:
            包含以下键的字典：
            - 'communities': 社区节点列表
            - 'count': 检测到的社区数量
        """
        logger.info("Detecting communities...")
        com = nx.algorithms.community.greedy_modularity_communities(self.graph)

        return {
            "communities": [
                list(sorted(c)) for c in com
            ],
            "count": len(com),
        }
