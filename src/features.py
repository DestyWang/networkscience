from __future__ import annotations

from typing import List, Tuple

import networkx as nx
import torch

from src.process import get_ACC, get_ASP, get_BC


def _average_neighbor_degree(
    graph: nx.Graph,
    node_list: List[str],
) -> torch.Tensor:
    """
    Helper that keeps the feature computation tidy.
    """

    avg_neighbor_degree_dict = nx.average_neighbor_degree(graph, nodes=node_list)
    values = [float(avg_neighbor_degree_dict.get(node, 0.0)) for node in node_list]
    return torch.tensor(values, dtype=torch.float32)


def get_node_features(graph: nx.Graph) -> Tuple[torch.Tensor, List[str], List[str]]:
    """
    Assemble per-node structural descriptors for downstream alignment.

    Feature order: degree, average neighbor degree, local clustering coefficient,
    average shortest-path length, betweenness centrality.

    Returns
    -------
    Tuple[torch.Tensor, List[str], List[str]]
        Feature matrix shaped (n, 5), human-readable names, and node ordering.
    """

    node_list = sorted(graph.nodes())
    if not node_list:
        return torch.zeros((0, 0), dtype=torch.float32), [], []

    degrees = torch.tensor(
        [float(graph.degree(node)) for node in node_list],
        dtype=torch.float32,
    )
    avg_neighbor_degree = _average_neighbor_degree(graph, node_list)

    clustering_dict = get_ACC(graph)
    clustering = torch.tensor(
        [float(clustering_dict.get(node, 0.0)) for node in node_list],
        dtype=torch.float32,
    )

    asp_dict = get_ASP(graph)
    avg_shortest = torch.tensor(
        [float(asp_dict.get(node, 0.0)) for node in node_list],
        dtype=torch.float32,
    )

    betweenness_dict = get_BC(graph)
    betweenness = torch.tensor(
        [float(betweenness_dict.get(node, 0.0)) for node in node_list],
        dtype=torch.float32,
    )

    features = torch.stack(
        (
            degrees,
            avg_neighbor_degree,
            clustering,
            avg_shortest,
            betweenness,
        ),
        dim=1,
    )
    feature_names = [
        "degree",
        "avg_neighbor_degree",
        "avg_clustering_coefficient",
        "avg_shortest_path_length",
        "betweenness_centrality",
    ]
    return features, feature_names, node_list
