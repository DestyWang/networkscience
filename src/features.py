from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np


def build_fo_vocabulary(annotations_list: Sequence[Dict[str, List[str]]], top_k: int = 64,) -> Dict[str, int]:
    """
    根据所有网络的 FO 频次，构建前 top_k 的词汇表。

    Parameters
    ----------
    annotations_list : Sequence[Dict[str, List[str]]]
        来自多个网络的 node->FO 映射集合。
    top_k : int
        选取的高频 FO 数（shape: 标量）。

    Returns
    -------
    Dict[str, int]
        词汇表，键为 FO 字符串，值为索引（shape: (<=top_k,)）。
    """

    counter = Counter()
    for annotations in annotations_list:
        for values in annotations.values():
            counter.update(values)
    most_common = counter.most_common(top_k)
    return {term: idx for idx, (term, _) in enumerate(most_common)}

def _compute_shortest_path_profile(graph: nx.Graph, node_list: List[str], sample_size: int = 256,) -> np.ndarray:
    """
    近似计算每个节点到随机采样节点的平均最短路长度。

    Returns
    -------
    np.ndarray
        形状 (n,) 的向量，表示每个节点的平均距离。
    """

    n = len(node_list)
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    sampled = random.sample(node_list, k=min(sample_size, n))
    idx = {node: i for i, node in enumerate(node_list)}
    total = np.zeros(n, dtype=np.float32)
    counts = np.zeros(n, dtype=np.float32)

    for source in sampled:
        lengths = nx.single_source_shortest_path_length(graph, source)
        for node, dist in lengths.items():
            j = idx[node]
            total[j] += float(dist)
            counts[j] += 1.0

    counts[counts == 0.0] = 1.0
    return total / counts

def _encode_fo_multi_hot(annotations: Dict[str, List[str]], node_list: List[str], vocab: Dict[str, int],) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 FO 标签编码为多热向量。

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        multi_hot: 形状 (n, |vocab|) 的 0/1 矩阵；
        fo_count: 形状 (n,) 的标签数量向量。
    """

    n = len(node_list)
    dim = len(vocab)
    multi_hot = np.zeros((n, dim), dtype=np.float32)
    fo_count = np.zeros(n, dtype=np.float32)
    for i, node in enumerate(node_list):
        fo_terms = annotations.get(node, [])
        fo_count[i] = float(len(fo_terms))
        for term in fo_terms:
            j = vocab.get(term)
            if j is not None:
                multi_hot[i, j] = 1.0
    return multi_hot, fo_count

def compute_node_features(graph: nx.Graph, annotations: Dict[str, List[str]], fo_vocab: Dict[str, int],
    *, shortest_path_sample: int = 256, betweenness_k: int = 256,) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    计算节点特征矩阵，确保与 FUGW 的 L2 cost 兼容。

    Features（按列顺序）：
    - degree（度，归一化）
    - avg_neighbor_degree（邻居平均度）
    - clustering（聚类系数）
    - closeness（紧密度中心性）
    - betweenness_approx（近似中介中心性，k 采样）
    - avg_shortest_path_to_sample（到采样子集的平均最短路）
    - fo_count（FO 标签数量）
    - fo_onehot_*（前 |vocab| 个 FO 的多热编码）

    Parameters
    ----------
    graph : nx.Graph
        输入无向图。
    annotations : Dict[str, List[str]]
        节点到 FO 列表的映射。
    fo_vocab : Dict[str, int]
        FO 词汇表（来自源/靶网络的联合高频项）。
    shortest_path_sample : int
        近似平均最短路的采样节点数。
    betweenness_k : int
        近似 betweenness 的采样节点数。

    Returns
    -------
    Tuple[np.ndarray, List[str], List[str]]
        features: 形状 (n, d) 的特征矩阵；
        feature_names: 每一维的名称列表（长度 d）；
        node_list: 节点顺序（后续与耦合矩阵索引对齐）。
    """

    node_list = sorted(graph.nodes())
    n = len(node_list)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32), [], []

    degrees = np.array([graph.degree(node) for node in node_list], dtype=np.float32)
    max_deg = max(float(degrees.max()), 1.0)
    degrees /= max_deg

    avg_neighbor_degree_dict = nx.average_neighbor_degree(graph, nodes=node_list)
    avg_neighbor_degree = np.array(
        [avg_neighbor_degree_dict.get(node, 0.0) for node in node_list],
        dtype=np.float32,
    )

    clustering = np.array(
        [nx.clustering(graph, node) for node in node_list],
        dtype=np.float32,
    )

    closeness = np.array(
        [nx.closeness_centrality(graph, u=node) for node in node_list],
        dtype=np.float32,
    )

    betweenness = nx.betweenness_centrality(
        graph,
        k=min(betweenness_k, n),
        normalized=True,
        seed=42,
    )
    betweenness_vec = np.array(
        [betweenness.get(node, 0.0) for node in node_list],
        dtype=np.float32,
    )

    avg_shortest = _compute_shortest_path_profile(
        graph=graph,
        node_list=node_list,
        sample_size=shortest_path_sample,
    )

    fo_multi_hot, fo_count = _encode_fo_multi_hot(
        annotations=annotations,
        node_list=node_list,
        vocab=fo_vocab,
    )

    base_features = [
        degrees,
        avg_neighbor_degree,
        clustering,
        closeness,
        betweenness_vec,
        avg_shortest,
        fo_count,
    ]
    features = np.concatenate(
        [np.stack(base_features, axis=1), fo_multi_hot],
        axis=1,
    )

    feature_names: List[str] = [
        "degree",
        "avg_neighbor_degree",
        "clustering",
        "closeness",
        "betweenness_approx",
        "avg_shortest_path_to_sample",
        "fo_count",
    ]
    feature_names.extend([f"fo_onehot_{term}" for term in fo_vocab.keys()])

    return features.astype(np.float32), feature_names, node_list

def standardize_pair(source_features: np.ndarray, target_features: np.ndarray, weights: np.ndarray | None = None, eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对源/靶特征做联合标准化并施加维度权重。

    Parameters
    ----------
    source_features : np.ndarray
        形状 (n_s, d)。
    target_features : np.ndarray
        形状 (n_t, d)。
    weights : np.ndarray | None
        形状 (d,) 的权重向量；None 时使用全 1。
    eps : float
        避免除 0 的平滑项。

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        归一化后的源/靶特征（同输入形状），以及实际使用的权重。
    """

    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError("source_features 与 target_features 维度不一致")

    combined = np.vstack([source_features, target_features])
    mean = combined.mean(axis=0)
    std = combined.std(axis=0) + eps

    if weights is None:
        weights = np.ones_like(mean, dtype=np.float32)
    weights = weights.astype(np.float32)

    source_norm = (source_features - mean) / std * weights
    target_norm = (target_features - mean) / std * weights
    return source_norm.astype(np.float32), target_norm.astype(np.float32), weights

