import pandas as pd

from src.models.network_analyzer import NetworkAnalyzer


def test_basic_metrics_and_communities():
    edges = pd.DataFrame({
        "source": ["A", "A", "B", "C"],
        "target": ["B", "C", "C", "D"],
    })

    na = NetworkAnalyzer(edges)
    metrics = na.basic_metrics()

    assert isinstance(metrics, dict)
    assert "degree" in metrics and "betweenness" in metrics and "closeness" in metrics

    comm = na.communities()
    assert isinstance(comm, dict)
    assert "communities" in comm and "count" in comm
    assert isinstance(comm["communities"], list)
